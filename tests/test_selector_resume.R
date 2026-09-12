#!/usr/bin/env Rscript
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])), ".."), winslash = "/")
Sys.setenv(SELECTOR_TRAIN_SOURCE_ONLY = "1")
on.exit(Sys.unsetenv("SELECTOR_TRAIN_SOURCE_ONLY"), add = TRUE)
source(file.path(root, "R", "selector", "train.R"))

assert_error <- function(expr, pattern) {
  error <- tryCatch({ force(expr); NULL }, error = identity)
  if (is.null(error) || !grepl(pattern, conditionMessage(error), ignore.case = TRUE))
    stop("Expected error matching ", pattern, call. = FALSE)
}

temporary <- tempfile("selector-resume-")
dir.create(temporary)
on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
signature <- list(schema_version = "test", data_hashes = list(rows = "abc"),
                  source_hashes = list(train = "def"), runtime = list(R = "4.6.1"),
                  config = list(folds = 3L))
calls <- 0L
trainer <- function() {
  calls <<- calls + 1L
  list(candidate = list(validation_macro_f1 = 0.5), tuning = list(scores = c(0.4, 0.5)))
}

first <- selector_checkpoint_run("context_glmnet", signature, temporary, trainer)
second <- selector_checkpoint_run("context_glmnet", signature, temporary, trainer)
stopifnot(calls == 1L, identical(first, second))
record <- readRDS(file.path(temporary, "context_glmnet.rds"))
stopifnot(record$status == "success", identical(record$cv_scores, c(0.4, 0.5)))

changed_data <- signature; changed_data$data_hashes$rows <- "changed"
assert_error(selector_checkpoint_run("context_glmnet", changed_data, temporary, trainer), "signature mismatch")
changed_runtime <- signature; changed_runtime$runtime$R <- "changed"
assert_error(selector_checkpoint_run("context_glmnet", changed_runtime, temporary, trainer), "signature mismatch")
changed_source <- signature; changed_source$source_hashes$train <- "changed"
assert_error(selector_checkpoint_run("context_glmnet", changed_source, temporary, trainer), "signature mismatch")
changed_config <- signature; changed_config$config$folds <- 5L
assert_error(selector_checkpoint_run("context_glmnet", changed_config, temporary, trainer), "signature mismatch")

failed_calls <- 0L
failing <- function() { failed_calls <<- failed_calls + 1L; stop("deliberate failure") }
assert_error(selector_checkpoint_run("response_rf", signature, temporary, failing), "deliberate failure")
failure <- readRDS(file.path(temporary, "response_rf.rds"))
stopifnot(failure$status == "failed", failure$error == "deliberate failure")
assert_error(selector_checkpoint_run("response_rf", signature, temporary, failing), "earlier failure")
stopifnot(failed_calls == 1L)

cat("PASS: candidate checkpoints reuse exact signatures, preserve CV scores and failures, and reject stale data/source/runtime/config.\n")
