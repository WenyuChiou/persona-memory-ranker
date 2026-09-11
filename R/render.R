#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(script), "common.R"))
root <- pmr_root()
pmr_use_library(root)
pmr_main(function() {
  args <- pmr_args("output", "pandoc")
  pmr_require(c("rmarkdown", "knitr", "ggplot2", "jsonlite"))
  pmr_use_pandoc(root)
  if (!is.null(args$pandoc)) Sys.setenv(RSTUDIO_PANDOC = args$pandoc)
  if (!rmarkdown::pandoc_available()) stop("Pandoc is required. Install Pandoc or pass --pandoc DIRECTORY.")
  dir.create(dirname(args$output), recursive = TRUE, showWarnings = FALSE)
  output <- normalizePath(args$output, winslash = "/", mustWork = FALSE)
  rmarkdown::render(file.path(root, "R", "eda.Rmd"), output_file = basename(output),
    output_dir = dirname(output), params = list(project_root = root),
    envir = new.env(parent = globalenv()), quiet = TRUE)
  cat("Rendered", output, "\n")
})
