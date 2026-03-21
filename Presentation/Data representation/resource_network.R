number_record <- 2219201

latency_base <- df2 %>%
  mutate(
    dimension = as.integer(dimension),
    encode_duration_s = as.numeric(encode_duration_s / number_record),
    network_latency_s = as.numeric(network_latency_s / number_record),
    inference_per_record_s = as.numeric(inference_per_record_s)
  ) %>%
  filter(
    !is.na(dimension),
    !is.na(encode_duration_s),
    !is.na(network_latency_s),
    !is.na(inference_per_record_s)
  ) %>%
  mutate(
    # 秒 -> 纳秒
    encode_latency_ns = encode_duration_s * 1e9,
    network_latency_ns = network_latency_s * 1e9,
    inference_latency_ns = inference_per_record_s * 1e9,
    total_latency_ns = encode_latency_ns + network_latency_ns + inference_latency_ns
  )

latency_long <- latency_base %>%
  select(encodertype, dimension, total_latency_ns,
         encode_latency_ns, network_latency_ns, inference_latency_ns) %>%
  pivot_longer(
    cols = c(encode_latency_ns, network_latency_ns, inference_latency_ns),
    names_to = "component",
    values_to = "latency_ns"
  ) %>%
  mutate(
    component = factor(
      component,
      levels = c("encode_latency_ns", "network_latency_ns", "inference_latency_ns"),
      labels = c("Encode", "Network", "Inference/record")
    )
  )

totals_df <- latency_base %>%
  select(encodertype, dimension, total_latency_ns)

g_latency <- ggplot(latency_long, aes(x = dimension, y = latency_ns, fill = component)) +
  geom_col(width = 0.75) +
  scale_x_continuous(breaks = dims, labels = dims) +
  geom_text(
    data = totals_df,
    aes(x = dimension, y = total_latency_ns, label = sprintf("%.0f", total_latency_ns)),
    inherit.aes = FALSE, vjust = -0.35, size = 3
  ) +
  facet_wrap(~ encodertype, scales = "free_y") +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Total Latency",
    x = "Feature Dimension",
    y = "Latency (ns)",
    fill = "Component"
  ) +
  theme_minimal()

g_latency
