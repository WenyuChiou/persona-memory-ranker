#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(script), "common.R"))
root <- pmr_root()
pmr_use_library(root)
pmr_main(function() {
  args <- pmr_args(c("split", "output"))
  if (!args$split %in% c("cv", "val", "benchmark")) stop("Unknown evaluation split.")
  pmr_require(c("ggplot2", "jsonlite"))
  python <- Sys.which("python")
  if (!nzchar(python)) stop("Python is required for evaluation provenance verification.")
  check <- system2(python, c("-m", "persona_memory_ranker.cli", "--root", shQuote(root),
                            "verify", "--split", args$split), stdout = TRUE, stderr = TRUE)
  status <- attr(check, "status")
  if (!is.null(status) && status != 0L) stop(paste(check, collapse = "\n"))
  values <- pmr_read_csv(file.path(root, "reports", paste0(args$split, "_summary.csv")))
  values <- values[values$metric == "budget_recall", , drop = FALSE]
  for (name in c("mean", "ci_low", "ci_high")) values[[name]] <- pmr_finite_number(values[[name]], name)
  alignment <- pmr_read_csv(file.path(root, "reports/alignment.csv"))
  source_split <- if (args$split == "cv") "train" else args$split
  alignment <- alignment[alignment$split == source_split, , drop = FALSE]
  retained <- sum(alignment$status == "aligned")
  excluded <- nrow(alignment)-retained
  if (retained != as.numeric(values$queries[1])) stop("Alignment and evaluation populations differ.")
  methods <- c("bm25", "semantic", "rrf", "logistic", "logistic_no_position", "random_forest")
  labels <- c("BM25", "Vector", "Hybrid RRF", "Logistic", "Logistic without position", "Random forest")
  if (nrow(values) != 6L || !setequal(values$method, methods)) stop("Incomplete method comparison.")
  values$method_name <- factor(values$method, levels = rev(methods), labels = rev(labels))
  values$model_family <- ifelse(values$method %in% methods[1:3], "Retrieval baseline", "R-trained model")
  plot <- ggplot2::ggplot(values, ggplot2::aes(method_name, mean, color = model_family)) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = ci_low, ymax = ci_high), width = 0.16, linewidth = 0.8) +
    ggplot2::geom_point(size = 3.4) + ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(limits = c(0, 1), labels = function(x) paste0(round(x*100), "%")) +
    ggplot2::scale_color_manual(values = c("Retrieval baseline" = "#687580", "R-trained model" = "#207D83")) +
    ggplot2::labs(title = paste("Evidence retrieval:", args$split),
      subtitle = paste(retained, "of", nrow(alignment), "cleaned questions · 2,000-token evidence budget"),
      x = NULL, y = "Annotated evidence recall", color = NULL,
      caption = paste(excluded, "alignment exclusions. 95% intervals: 1,000 persona bootstrap resamples.\nSynthetic PersonaMem-v2 (CC BY 4.0); incomplete annotations. Budget uses MiniLM WordPiece.")) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "bottom", panel.grid.major.y = ggplot2::element_blank(),
                   plot.title = ggplot2::element_text(face = "bold"))
  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(args$output, plot, width = 10, height = 5.5, dpi = 180, bg = "white")
  cat("Saved verified R result figure:", args$output, "\n")
})
