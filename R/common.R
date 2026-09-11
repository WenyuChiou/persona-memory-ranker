# Shared, dependency-light helpers for the R course pipeline.
pmr_root <- function() {
  script <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (!length(script)) stop("Run this entry point with Rscript.", call. = FALSE)
  normalizePath(file.path(dirname(sub("^--file=", "", script[[1]])), ".."),
                winslash = "/", mustWork = TRUE)
}

pmr_use_library <- function(root) {
  pmr_utf8_locale()
  lib <- file.path(root, ".Rlib")
  if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
  invisible(lib)
}

pmr_utf8_locale <- function() {
  if (isTRUE(l10n_info()[["UTF-8"]])) return(invisible(TRUE))
  # A Unix-style C.UTF-8 environment is not a valid Windows locale. If R falls
  # back to C, fileEncoding converts Unicode into the native encoding and may
  # truncate reads or write literal <U+....> substitutions. Use a UTF-8 CTYPE
  # for this R process before parsing, serializing, or operating on source text.
  choices <- if (.Platform$OS.type == "windows") ".UTF-8" else c("C.UTF-8", "en_US.UTF-8", "")
  for (choice in choices) {
    suppressWarnings(Sys.setlocale("LC_CTYPE", choice))
    if (isTRUE(l10n_info()[["UTF-8"]])) return(invisible(TRUE))
  }
  stop("A UTF-8 LC_CTYPE locale is required to preserve source text; none is available.", call. = FALSE)
}

pmr_use_pandoc <- function(root) {
  if (!nzchar(Sys.getenv("RSTUDIO_PANDOC"))) {
    installed <- file.path(root, ".Rlib", "pandoc")
    if (dir.exists(installed)) {
      executable <- if (.Platform$OS.type == "windows") "pandoc.exe" else "pandoc"
      paths <- list.files(installed, pattern = paste0("^", gsub(".", "\\.", executable, fixed = TRUE), "$"),
                          recursive = TRUE, full.names = TRUE)
      if (length(paths)) Sys.setenv(RSTUDIO_PANDOC = dirname(paths[[1]]))
    }
  }
  invisible(Sys.getenv("RSTUDIO_PANDOC"))
}

pmr_require <- function(packages) {
  absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent)) stop("Missing R packages: ", paste(absent, collapse = ", "),
                           ". Run Rscript R/bootstrap.R first.", call. = FALSE)
}

pmr_args <- function(required, optional = character()) {
  args <- commandArgs(TRUE)
  if (length(args) %% 2L) stop("Arguments must be --name VALUE pairs.", call. = FALSE)
  result <- list()
  if (length(args)) for (i in seq.int(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[[i]])
    if (!startsWith(args[[i]], "--") || !key %in% c(required, optional) ||
        !is.null(result[[key]])) stop("Unknown or duplicate argument: ", args[[i]], call. = FALSE)
    result[[key]] <- args[[i + 1L]]
  }
  missing <- setdiff(required, names(result))
  if (length(missing)) stop("Missing arguments: ", paste(paste0("--", missing), collapse = ", "), call. = FALSE)
  result
}

pmr_read_csv <- function(path) {
  if (!file.exists(path)) stop("Input does not exist: ", path, call. = FALSE)
  pmr_utf8_locale()
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
           colClasses = "character", na.strings = NULL, encoding = "UTF-8")
}

pmr_write_csv <- function(data, path) {
  pmr_utf8_locale()
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(data, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

pmr_write_json <- function(object, path) {
  pmr_utf8_locale()
  pmr_require("jsonlite")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(object, path, auto_unbox = TRUE, pretty = TRUE,
                       digits = NA, null = "null")
}

pmr_require_columns <- function(data, required, exact = FALSE) {
  if (anyDuplicated(names(data))) stop("Duplicate column names are forbidden.", call. = FALSE)
  missing <- setdiff(required, names(data))
  if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "), call. = FALSE)
  extra <- setdiff(names(data), required)
  if (exact && length(extra)) stop("Unexpected columns: ", paste(extra, collapse = ", "), call. = FALSE)
}

pmr_nonempty <- function(x) !is.na(x) & nzchar(trimws(x))

pmr_finite_number <- function(x, field) {
  value <- suppressWarnings(as.numeric(x))
  if (any(!is.finite(value))) stop("Non-finite or nonnumeric value in ", field, ".", call. = FALSE)
  value
}

pmr_main <- function(fun) {
  tryCatch(fun(), error = function(error) {
    message("ERROR: ", conditionMessage(error))
    quit(save = "no", status = 1L)
  })
}
