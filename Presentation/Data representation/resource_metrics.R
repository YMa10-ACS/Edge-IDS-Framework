library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)


setwd('/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/records/')

df <- read_csv("./encoder_metrics_20260319_142750.csv")

df2 <- df %>%
  extract(
    encoder,
    into = c("encodertype", "dimension"),
    regex = "([A-Za-z_]+)(\\d+)?",
    remove = FALSE
  ) %>%
  mutate(dimension = as.integer(dimension))

dims <- c(4, 8, 12, 16, 20, 24, 28, 32)
fs_base <- df2 %>%
  filter(encodertype == "feature_selection") %>%
  slice(1) %>%
  select(-dimension)

fs_expanded <- crossing(fs_base, dimension = dims)

df2 <- df2 %>%
  filter(encodertype != "feature_selection") %>%
  bind_rows(fs_expanded) %>%
  arrange(encodertype, dimension)

mem_data <- df2 %>%
  select(encodertype, dimension, rss_net_growth_mb) %>%
  mutate(
    rss_net_growth_mb = if_else(rss_net_growth_mb <= 0, 1e-6, rss_net_growth_mb),
    mem_label = scales::label_number(accuracy = 0.01, big.mark = "")(rss_net_growth_mb)
  ) %>%
  group_by(dimension) %>%
  mutate(
    mem_rank_in_dim = rank(rss_net_growth_mb, ties.method = "first"),
    mem_label_mult = c(1.08, 1.18, 1.30, 1.44)[mem_rank_in_dim],
    mem_label_y = rss_net_growth_mb * mem_label_mult
  ) %>%
  ungroup()

# memory metrics
g_mem <- ggplot(mem_data, aes(x = dimension, y = rss_net_growth_mb, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_label(
    aes(y = mem_label_y, label = mem_label),
    size = 2.4,
    label.size = 0.15,
    fill = "white",
    show.legend = FALSE
  ) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_log10(
    labels = scales::label_number(accuracy = 0.01, big.mark = "")
  ) +
  scale_x_continuous(breaks = dims, labels = dims) +
  coord_cartesian(clip = "off") +  
  labs(
    y = "Memory (MB, log10 scale)",
    color = "Encoder Type"
  ) +
  theme_minimal() + 
  theme(
    axis.text.x = element_text(size = 11),
    plot.margin = margin(10, 20, 10, 10) 
  )

cpu_data <- df2 %>% 
  select(encodertype, dimension, cpu_avg_pct, encode_duration_s) %>%
  mutate(
    cpu_time_sec = (cpu_avg_pct / 100) * encode_duration_s,
    cpu_time_sec = if_else(cpu_time_sec <= 0, 1e-6, cpu_time_sec),
    cpu_label = scales::label_number(accuracy = 0.01, big.mark = "")(cpu_time_sec)
  ) %>%
  group_by(dimension) %>%
  mutate(
    cpu_rank_in_dim = rank(cpu_time_sec, ties.method = "first"),
    cpu_label_mult = c(1.08, 1.18, 1.30, 1.44)[cpu_rank_in_dim],
    cpu_label_y = cpu_time_sec * cpu_label_mult
  ) %>%
  ungroup()

g_cpu <- ggplot(cpu_data, aes(x = dimension, y = cpu_time_sec, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_label(
    aes(y = cpu_label_y, label = cpu_label),
    size = 2.4,
    label.size = 0.15,
    fill = "white",
    show.legend = FALSE
  ) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_log10(
    labels = scales::label_number(accuracy = 0.01, big.mark = "")
  ) +
  scale_x_continuous(breaks = dims, labels = dims) +
  coord_cartesian(clip = "off") +  
  labs(
    y = "CPU Time (core-seconds, log10 scale)",
    color = "Encoder Type"
  ) +
  theme_minimal() + 
  theme(
    axis.title.x = element_text(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

g_cpu / g_mem
