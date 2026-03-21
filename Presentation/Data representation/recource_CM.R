library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)


setwd('/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/records/')

df <- read_csv("./encoder_metrics_20260321_152436.csv")

df2 <- df %>%
  extract(
    encoder,
    into = c("encodertype", "dimension"),
    regex = "([A-Za-z_]+)(\\d+)?",
    remove = FALSE
  ) %>%
  mutate(dimension = as.integer(dimension))

dims <- sort(unique(df2$dimension[!is.na(df2$dimension)]))

cpu_data <- df2 %>%
  select(encodertype, dimension, cpu_avg_pct, encode_duration_s) %>%
  filter(!is.na(dimension)) %>%
  mutate(cpu_time_sec = (cpu_avg_pct / 100))

g_cpu_percentange <- ggplot(cpu_data, aes(x = dimension, y = cpu_time_sec, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = dims, labels = dims) +
  labs(
    title = "CPU Time",
    y = "CPU Time (core-seconds)",
    fill = "Encoder Type"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

g_cpu_duration <- ggplot(cpu_data, aes(x = factor(dimension), y = encode_duration_s, fill = encodertype)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "CPU Time",
    y = "CPU Time (core-seconds)",
    color = "Encoder Type"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

mem_data <- df2 %>%
  select(encodertype, dimension, rss_net_growth_mb) %>%
  filter(!is.na(dimension))

g_mem <- ggplot(mem_data, aes(x = dimension, y = rss_net_growth_mb, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = dims, labels = dims) +
  labs(
    title = "Memory",
    x = "Feature Dimension",
    y = "Memory (MB)",
    color = "Encoder Type"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

(g_cpu_percentange | g_cpu_duration) / g_mem
