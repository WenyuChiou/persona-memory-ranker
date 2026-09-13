#!/usr/bin/env Rscript
# Dependencies used by the Persona Memory Selector R analysis.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
root <- normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = TRUE)
source(file.path(root, "R", "common.R"))
lib <- file.path(root, ".Rlib"); dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
packages <- c("glmnet", "ranger", "jsonlite", "digest", "rmarkdown", "knitr")
absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(absent)) install.packages(absent, repos = "https://cloud.r-project.org", lib = lib)
pmr_require(packages)
print(data.frame(package = packages,
  version = vapply(packages, function(x) as.character(packageVersion(x)), character(1))), row.names = FALSE)
