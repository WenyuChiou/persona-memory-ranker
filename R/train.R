#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(script), "common.R"))
source(file.path(dirname(script), "models.R"))
pmr_use_library(pmr_root())
pmr_main(function() {
  args <- pmr_args(c("train", "val", "out"))
  pmr_require(c("jsonlite", "ranger"))
  result <- pmr_train(pmr_read_csv(args$train), pmr_read_csv(args$val), args$out)
  cat("Trained models on", result$report$train_rows, "candidate rows and",
      result$report$train_personas, "personas; wrote", args$out, "\n")
})
