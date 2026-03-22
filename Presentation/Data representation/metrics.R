library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)


setwd('/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/records/')

df <- read_csv("./encoder_metrics_20260322_112410.csv")

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

accuracy_data <- df2 %>%
  select(encodertype, dimension, test_accuracy) %>%
  filter(!is.na(dimension))

g_acc <- ggplot(accuracy_data, aes(x = dimension, y = test_accuracy, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_text(
    aes(label = scales::percent(test_accuracy, accuracy = 0.1)),
    vjust = -0.6, size = 3, show.legend = FALSE
  ) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_continuous(labels = scales::label_percent(accuracy = 0.1)) +
  scale_x_continuous(breaks = dims, labels = dims) +
  coord_cartesian(clip = "off") +
  labs(
    title = "Accuracy",
    color = "Encoder Type"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

g_f1 <- ggplot(f1_data, aes(x = dimension, y = test_f1_score, color = encodertype)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_text(
    aes(label = scales::percent(test_f1_score, accuracy = 0.1)),
    vjust = -0.6, size = 3, show.legend = FALSE
  ) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_continuous(labels = scales::label_percent(accuracy = 0.1)) +
  scale_x_continuous(breaks = dims, labels = dims) +
  coord_cartesian(clip = "off") +
  labs(
    title = "F1 Score",
    x = "Feature Dimension",
    y = NULL,
    color = "Encoder Type"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(),
    axis.text.x  = element_text(size = 11, colour = "black"),
    axis.ticks.x = element_line(colour = "black")
  )

g_acc / g_f1

