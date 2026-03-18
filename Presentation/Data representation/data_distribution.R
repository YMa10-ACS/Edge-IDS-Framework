library(readr)
library(dplyr)
library(ggplot2)
library(scales)
setwd('/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/Presentation/Data representation')
df <- read_csv("./edge_iiotset_distribution.csv")

head(df)
plot_df <- df %>%
  filter(!is.na(`attack type`), !is.na(Number)) %>%
  mutate(Percentage = Number / sum(Number)) %>%
  arrange(Percentage)

plot_df <- plot_df %>%
  filter(`attack type` != "Normal")

ggplot(plot_df, aes(x = Percentage, y = reorder(`attack type`, Percentage))) +
  geom_col(fill = "#4A90E2") +
  geom_text(aes(label = percent(Percentage, accuracy = 0.1)),
            hjust = -0.1, size = 4) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Edge-IIoTset",
    x = "Percentage",
    y = "Attack Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
  ) + 
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    
    axis.line = element_blank()
  )