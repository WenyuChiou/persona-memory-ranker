#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
root <- if (length(args)) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else getwd()
source(file.path(root, "R", "common.R"))
pmr_use_library(root)
source(file.path(root, "R", "selector", "core.R"))
cat(jsonlite::toJSON(selector_runtime(), auto_unbox = TRUE))
