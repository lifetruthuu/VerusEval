#!/usr/bin/env Rscript
# Render the RQ2 figures: model-matched workflow comparisons and StarVerus model and prompt contrasts.
# Run with: conda run -n r Rscript RQs/RQ2/scripts/plot_rq2_figures.R
# Reads results/rq2_configuration_profiles.csv and results/rq2_prompt_deltas.csv written by
# analyze_rq2_configurations.py; dimensions and metrics follow the order of Table 1.

plot_cache <- file.path(tempdir(), "veruseval-rq2-figures-cache")
dir.create(plot_cache, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(XDG_CACHE_HOME = plot_cache)

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
project_root <- normalizePath(
  file.path(dirname(sub("^--file=", "", script_arg)), "..", "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
rq2_dir <- file.path(project_root, "RQs", "RQ2")
figure_dir <- file.path(rq2_dir, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

font_family <- "Liberation Sans"
ink <- "#212121"
muted <- "#5F5F5C"
rule <- "#E1E1DC"
dimensions <- c(
  "Text similarity" = "Text\nsimilarity",
  "Intent judgement" = "Intent\njudgement",
  "Verifier validity" = "Verifier\nvalidity",
  "Semantic triviality" = "Semantic\ntriviality",
  "Specification correctness" = "Specification\ncorrectness",
  "Behavior correctness" = "Behavior\ncorrectness"
)
metrics <- c(
  "BLEU" = "BLEU", "ROUGE-L" = "ROUGE-L", "KeySpecMatch" = "KeySpecMatch",
  "Spec-code judgement" = "Spec–code", "Spec-spec judgement" = "Spec–spec",
  "Syntactic validity" = "Syntactic validity", "Type validity" = "Type validity",
  "Verifier acceptance" = "Acceptance",
  "No false precondition" = "False pre. not proved", "No true postcondition" = "True post. not proved",
  "Soundness" = "Soundness", "Completeness" = "Completeness", "Equivalence" = "Equivalence",
  "Correct-I/O acceptance" = "Correct I/O", "Wrong-output rejection" = "Wrong output",
  "Invalid-input rejection" = "Invalid input"
)
models <- c("llama" = "Llama", "gpt-4o" = "GPT-4o", "deepseek-chat" = "DeepSeek-Chat",
            "deepseek-reasoner" = "DeepSeek-Reasoner", "qwen-coder" = "Qwen-Coder")
model_colours <- c("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c0457a")
model_shapes <- c(16, 17, 15, 18, 4)
workflows <- c("AlphaVerus" = "AlphaVerus", "AutoVerus" = "AutoVerus", "StarVerus" = "StarVerus",
               "VerusAge" = "VeruSAGE")

read_result <- function(name) {
  read.csv(file.path(rq2_dir, "results", name), stringsAsFactors = FALSE, check.names = FALSE) %>%
    mutate(dimension = factor(dimensions[dimension], levels = dimensions),
           metric = factor(metrics[metric], levels = rev(metrics)))
}

theme_rq2 <- function() {
  theme_minimal(base_size = 8, base_family = font_family) +
    theme(
      text = element_text(color = ink),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = rule, linewidth = 0.3),
      panel.spacing.x = unit(14, "pt"),
      panel.spacing.y = unit(4, "pt"),
      strip.text.y = element_blank(),
      strip.text.x = element_text(face = "bold", size = 7.2, lineheight = 1.1, margin = margin(b = 3)),
      axis.text.y = element_text(size = 6.8, color = ink),
      axis.text.x = element_text(size = 6.3, color = muted),
      axis.title.x = element_text(size = 7, color = muted, margin = margin(t = 2)),
      axis.title.y = element_blank(),
      plot.title = element_text(size = 7.5, face = "bold", margin = margin(b = 1)),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 7.2),
      legend.key.width = unit(10, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(2, "pt"),
      plot.margin = margin(3, 5, 2, 2)
    )
}

percent_axis <- function() {
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25), expand = expansion(mult = c(0.03, 0.04)))
}

save_figure <- function(plot, name, width, height) {
  ggsave(file.path(figure_dir, paste0(name, ".pdf")), plot, width = width, height = height, units = "in",
         device = cairo_pdf)
  ggsave(file.path(figure_dir, paste0(name, ".png")), plot, width = width, height = height, units = "in",
         dpi = 300, bg = "white")
}

profiles <- read_result("rq2_configuration_profiles.csv") %>% mutate(value = 100 * mean)
deltas <- read_result("rq2_prompt_deltas.csv") %>%
  mutate(across(c(delta, ci_low, ci_high), ~ 100 * .x))

# Figure 1: under few-shot prompting, StarVerus against the other workflow that shares each model.
frames <- data.frame(model = c("llama", "gpt-4o", "deepseek-chat", "qwen-coder"),
                     other = c("AlphaVerus", "AutoVerus", "VerusAge", "VerusAge"))
frames$frame <- paste0(models[frames$model], "\nvs. ", workflows[frames$other])
workflow_rows <- profiles %>%
  filter(shot == "few-shot") %>%
  inner_join(frames, by = "model") %>%
  filter(workflow == "StarVerus" | workflow == other) %>%
  mutate(frame = factor(frame, levels = frames$frame),
         role = factor(ifelse(workflow == "StarVerus", "StarVerus", "Compared workflow"),
                       levels = c("StarVerus", "Compared workflow")))
stopifnot(nrow(workflow_rows) == 4 * 2 * length(metrics))
gaps <- workflow_rows %>%
  group_by(frame, dimension, metric) %>%
  summarise(low = min(value), high = max(value), .groups = "drop")

workflow_plot <- ggplot(workflow_rows, aes(x = value, y = metric)) +
  geom_segment(data = gaps, aes(x = low, xend = high, y = metric, yend = metric),
               color = "#C4C8CC", linewidth = 0.9, lineend = "round") +
  geom_point(aes(color = role, shape = role), size = 1.5) +
  facet_grid(dimension ~ frame, scales = "free_y", space = "free_y") +
  scale_color_manual(values = c("StarVerus" = "#2a78d6", "Compared workflow" = "#eb6834")) +
  scale_shape_manual(values = c("StarVerus" = 16, "Compared workflow" = 17)) +
  percent_axis() +
  labs(x = "Configuration mean (%)") +
  theme_rq2() +
  # Right-align the legend and lift it into the axis-title row, which it leaves free beside the title.
  theme(legend.justification = c(1, 0), legend.box.spacing = unit(-10, "pt"))
save_figure(workflow_plot, "rq2_workflow_comparison", 6.3, 2.6)

# Figure 2: StarVerus models under few-shot prompting, and task-paired few-shot minus zero-shot changes.
dodge <- position_dodge(width = 0.8, orientation = "y")

# Shade every other metric row, starting from the top of each dimension, to separate the dodged points.
row_bands <- function(rows) {
  rows %>%
    distinct(dimension, metric) %>%
    group_by(dimension) %>%
    mutate(position = rank(as.integer(metric))) %>%
    filter((max(position) - position) %% 2 == 0) %>%
    ungroup()
}
band_layer <- function(rows) {
  geom_rect(data = row_bands(rows), aes(xmin = -Inf, xmax = Inf, ymin = position - 0.5, ymax = position + 0.5),
            inherit.aes = FALSE, fill = "#F0F0EC")
}
# Grid lines drawn above the bands, since the theme grid sits beneath every layer.
grid_layer <- function(breaks) geom_vline(xintercept = breaks, color = rule, linewidth = 0.3)
delta_breaks <- seq(0, 40, 10)
model_rows <- profiles %>%
  filter(workflow == "StarVerus", shot == "few-shot") %>%
  mutate(model = factor(models[model], levels = models))
delta_rows <- deltas %>%
  filter(workflow == "StarVerus") %>%
  mutate(model = factor(models[model], levels = models))
stopifnot(nrow(model_rows) == 5 * length(metrics), nrow(delta_rows) == 5 * length(metrics))

model_scales <- list(scale_color_manual(values = setNames(model_colours, models)),
                     scale_shape_manual(values = setNames(model_shapes, models)))
model_plot <- ggplot(model_rows, aes(x = value, y = metric, color = model, shape = model)) +
  geom_blank() +
  band_layer(model_rows) +
  grid_layer(seq(0, 100, 25)) +
  geom_point(position = dodge, size = 0.95) +
  facet_grid(dimension ~ ., scales = "free_y", space = "free_y") +
  model_scales +
  percent_axis() +
  labs(title = "(a) Few-shot means", x = "Configuration mean (%)") +
  theme_rq2() +
  theme(panel.grid.major.x = element_blank())
delta_plot <- ggplot(delta_rows, aes(x = delta, y = metric, color = model, shape = model)) +
  geom_blank() +
  band_layer(delta_rows) +
  grid_layer(delta_breaks[delta_breaks != 0]) +
  geom_vline(xintercept = 0, color = muted, linewidth = 0.4) +
  geom_linerange(aes(xmin = ci_low, xmax = ci_high), position = dodge, linewidth = 0.35, show.legend = FALSE) +
  geom_point(position = dodge, size = 0.95) +
  facet_grid(dimension ~ ., scales = "free_y", space = "free_y") +
  model_scales +
  scale_x_continuous(breaks = delta_breaks, labels = function(x) ifelse(x > 0, paste0("+", x), x)) +
  labs(title = "(b) Few-shot minus zero-shot", x = "Paired change (percentage points)") +
  theme_rq2() +
  theme(axis.text.y = element_blank(), panel.grid.major.x = element_blank())
combined <- (model_plot | delta_plot) +
  plot_layout(guides = "collect", widths = c(1, 1)) &
  theme(legend.position = "bottom")
save_figure(combined, "rq2_model_prompt_comparison", 6.3, 2.95)
message("RQ2 figures written to ", figure_dir)
