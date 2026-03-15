#!/usr/bin/env python3
"""
Monitor CPU usage for edge.py processes on macOS/Linux using `ps`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time


def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def find_pids(pattern):
    proc = run_cmd(["pgrep", "-f", pattern])
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    out = []
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            out.append(int(line))
    return sorted(set(out))


def read_stats(pid):
    # %cpu and rss(KB)
    proc = run_cmd(["ps", "-p", str(pid), "-o", "%cpu=", "-o", "rss="])
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        cpu = float(parts[0])
        rss_kb = int(parts[1])
    except ValueError:
        return None
    return cpu, rss_kb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None, help="Monitor a specific PID")
    parser.add_argument(
        "--pattern",
        default="Framework/edge.py",
        help="Process match pattern when --pid is not provided",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit")
    args = parser.parse_args()

    if args.pid is not None:
        pids = [args.pid]
    else:
        pids = find_pids(args.pattern)

    if not pids:
        print(f"No process found. pattern={args.pattern!r}")
        return

    print("Monitoring PIDs:", ", ".join(map(str, pids)))
    print("timestamp,pid,cpu_percent,rss_mb")

    while True:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alive = 0
        for pid in pids:
            stats = read_stats(pid)
            if stats is None:
                continue
            alive += 1
            cpu, rss_kb = stats
            rss_mb = rss_kb / 1024.0
            print(f"{now},{pid},{cpu:.2f},{rss_mb:.2f}")

        if alive == 0:
            print("All monitored processes have exited.")
            return

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
