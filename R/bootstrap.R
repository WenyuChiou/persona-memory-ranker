#!/usr/bin/env Rscript
# Install explicit direct dependencies into the project, never a global library.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(script), "common.R"))
root <- pmr_root()
lib <- file.path(root, ".Rlib")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
packages <- c("jsonlite", "ranger", "rmarkdown", "knitr", "ggplot2")
absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(absent)) {
  options(timeout = 300)
  install.packages(absent, repos = "https://cloud.r-project.org", lib = lib)
}
pmr_require(packages)
pmr_use_pandoc(root)
versions <- data.frame(package = packages,
                       version = vapply(packages, function(p) as.character(utils::packageVersion(p)), character(1)))
print(versions, row.names = FALSE)
cat("Project library:", lib, "\n")
cat("Pandoc available:", rmarkdown::pandoc_available(), "\n")
if (!rmarkdown::pandoc_available()) {
  cat("HTML rendering also requires Pandoc; set RSTUDIO_PANDOC to its containing directory.\n")
}
