#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
selector_project_root <- normalizePath(if (length(args)) args[[1]] else getwd(), winslash = "/", mustWork = TRUE)
source(file.path(selector_project_root, "R", "common.R"))
pmr_use_library(selector_project_root); pmr_use_pandoc(selector_project_root)
pmr_require(c("rmarkdown", "knitr", "jsonlite", "digest"))
if (!rmarkdown::pandoc_available()) stop("Pandoc is required to render the selector notebook.")
# Bind paths and parameters before knitting changes its working directory.
selector_params <- list(project_root = selector_project_root)
selector_output <- file.path(selector_project_root, "reports", "selector")
dir.create(selector_output, recursive = TRUE, showWarnings = FALSE)
rmarkdown::render(file.path(selector_project_root, "R", "selector", "eda.Rmd"),
  output_file = "eda.html", output_dir = selector_output, params = selector_params,
  envir = new.env(parent = globalenv()), quiet = TRUE)
cat("Rendered selector EDA in", selector_output, "\n")
