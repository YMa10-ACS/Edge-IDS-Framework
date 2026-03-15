#!/usr/bin/env python3
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import inspect
from flask import Flask, request, jsonify

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NANOVLM_PATH = os.path.join(PROJECT_ROOT, "nanoVLM")
sys.path.insert(0, NANOVLM_PATH)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer

import torch
import torch.nn as nn

@torch.no_grad()
def generate_from_embeddings(
    decoder: nn.Module,
    lm_head: nn.Module,
    token_emb: nn.Embedding,
    tokenizer,
    inputs: torch.Tensor,          # (1, S0, D)  S0=inputs_len
    inputs_len: int,
    max_new_tokens: int = 50,
    greedy: bool = True,
    temperature: float = 1.0,
):
    """
    Correct decoding when lm_use_tokens=False and you already have embedding inputs.
    Uses: prefill (forward) + token-space decoding (lm_head -> id -> token_emb -> forward with kv_cache).
    Returns: decoded string + generated_ids list
    """
    device = inputs.device

    # ---------- 1) Prefill ----------
    out, kv_cache = decoder.forward(
        inputs,
        attention_mask=None,
        kv_cache=None,
        start_pos=0
    )
    last_hidden = out[:, -1, :]     # (1, D)

    generated_ids = []

    # ---------- 2) Decode loop ----------
    for t in range(max_new_tokens):
        logits = lm_head(last_hidden)   # (1, vocab)

        if temperature != 1.0:
            logits = logits / temperature

        if greedy:
            next_id = torch.argmax(logits, dim=-1)         # (1,)
        else:
            probs = torch.softmax(logits, dim=-1)          # (1, vocab)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)

        generated_ids.append(int(next_id.item()))

        # token id -> token embedding (1, 1, D)
        next_emb = token_emb(next_id).unsqueeze(1)

        # IMPORTANT: start_pos must continue after the prefill length
        out, kv_cache = decoder.forward(
            next_emb,
            attention_mask=None,
            kv_cache=kv_cache,
            start_pos=inputs_len + t
        )
        last_hidden = out[:, -1, :]     # (1, D)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids

# ---------- helpers ----------
def find_token_embedding(decoder: nn.Module) -> nn.Embedding:
    # common names first
    for name in ["tok_embeddings", "token_embedding", "embed_tokens", "wte", "embedding"]:
        if hasattr(decoder, name) and isinstance(getattr(decoder, name), nn.Embedding):
            return getattr(decoder, name)
    # fallback: first embedding
    for m in decoder.modules():
        if isinstance(m, nn.Embedding):
            return m
    raise AttributeError("No nn.Embedding found in model.decoder")

def find_lm_head(decoder: nn.Module, vocab_size: int) -> nn.Linear:
    cands = []
    for name, m in decoder.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == vocab_size:
            cands.append((name, m.in_features, m.out_features))
    print("[cloud] lm_head candidates:", cands[:10])
    if not cands:
        raise AttributeError("No Linear layer with out_features==vocab_size found in decoder.")
    best_name = cands[0][0]
    lm_head = dict(decoder.named_modules())[best_name]
    print(f"[cloud] using lm_head: {best_name} -> {lm_head}")
    return lm_head

# ---------- init ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("[cloud] device:", device)

model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-230M-8k").eval().to(device)

tokenizer = get_tokenizer(
    model.cfg.lm_tokenizer,
    model.cfg.vlm_extra_tokens,
    model.cfg.lm_chat_template
)

# IMPORTANT: find these AFTER model.to(device)
token_emb = find_token_embedding(model.decoder)
print("[cloud] token_emb:", token_emb, "dtype:", token_emb.weight.dtype, "device:", token_emb.weight.device)

vocab_size = token_emb.num_embeddings
lm_head = find_lm_head(model.decoder, vocab_size)
print("[cloud] lm_head dtype:", lm_head.weight.dtype, "device:", lm_head.weight.device)

app = Flask(__name__)

@app.route("/", methods=["POST"])
def receive_embedding():
    t0 = time.perf_counter()

    raw = request.data
    meta_json = request.headers.get("Visual-Meta")

    metadata = json.loads(meta_json)
    shape = metadata["shape"]
    dtype = metadata["dtype"]

    np_dtype = np.float16 if dtype == "float16" else np.float32
    arr = np.frombuffer(raw, dtype=np_dtype)

    arr = arr.reshape(shape)  # (13, 64, 576)
    print(f"arr = {arr.shape}")
    # rebuild projected tensor -> (1, 832, 576)
    proj = torch.tensor(arr, device=device).reshape(1, -1, shape[-1])

    # prompt -> input_ids
    prompt = "Describe the image."
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    # ids -> text embeddings
    text_emb = token_emb(input_ids)

    # unify dtype to token_emb dtype (best practice)
    target_dtype = token_emb.weight.dtype
    proj = proj.to(target_dtype)
    text_emb = text_emb.to(target_dtype)

    # concat (image first)
    inputs = torch.cat([proj, text_emb], dim=1)
    inputs_len = inputs.shape[1]

    # ✅ correct decoding
    out_text, gen_ids = generate_from_embeddings(
        decoder=model.decoder,
        lm_head=lm_head,
        token_emb=token_emb,
        tokenizer=tokenizer,
        inputs=inputs,
        inputs_len=inputs_len,
        max_new_tokens=196,
        greedy=True,          # 先用 greedy 跑通
        temperature=1.0
    )

    print("[cloud] first 30 gen_ids:", gen_ids[:])

    # 看看这些 id 对应的 token 是啥
    tokens_str = tokenizer.convert_ids_to_tokens(gen_ids[:30])
    print("[cloud] first 30 tokens:", tokens_str)

    # 不跳过 special tokens 解码一次（看看是不是全被 skip 掉了）
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    print("[cloud] raw decoded (no skip):", repr(raw_text[:200]))

    print(f"[cloud] inputs_len={inputs_len}, gen_len={len(gen_ids)}")
    print("[cloud] generated:", out_text)

    return jsonify({"result": out_text})

if __name__ == "__main__":
    app.run(port=8000)