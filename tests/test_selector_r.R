#!/usr/bin/env Rscript
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])), ".."), winslash = "/")
source(file.path(root, "R", "common.R")); source(file.path(root, "R", "selector", "core.R")); pmr_use_library(root)
pmr_require("Matrix")

assert_error <- function(expr, pattern) {
  error <- tryCatch({ force(expr); NULL }, error = identity)
  if (is.null(error) || !grepl(pattern, conditionMessage(error), ignore.case = TRUE))
    stop("Expected error matching ", pattern, call. = FALSE)
}

# The EDA must audit the selector's pinned BIG5-CHAT source, never the legacy benchmark inputs.
eda_source <- paste(readLines(file.path(root, "R", "selector", "eda.Rmd"), warn = FALSE), collapse = "\n")
stopifnot(grepl('data", "raw", "selector", "big5_chat_dataset.csv', eda_source, fixed = TRUE),
          grepl('reports", "selector", "source_manifest.json', eda_source, fixed = TRUE),
          !grepl("download_manifest.json", eda_source, fixed = TRUE),
          !grepl('data", "raw", "benchmark', eda_source, fixed = TRUE))

# Group folds are deterministic, cover three folds, and never split a group.
groups <- rep(sprintf("g%02d", 1:12), each = 10)
folds <- selector_group_folds(groups)
stopifnot(identical(sort(unique(folds)), 1:3))
for (g in unique(groups)) stopifnot(length(unique(folds[groups == g])) == 1L)

# Constant columns are excluded while center/scale retain the full portable input schema.
x <- cbind(e000 = 1:12, e001 = 7, e002 = rep(c(0, 1), 6))
prep <- selector_preprocessor(x)
stopifnot(identical(prep$selected, c(1L, 3L)), prep$scale[[2]] == 1,
          identical(colnames(selector_apply_preprocessor(x, prep)), c("e000", "e002")))

# The class vocabulary is stable regardless of input order.
stopifnot(identical(SELECTOR_CLASSES, sort(SELECTOR_CLASSES)), length(SELECTOR_CLASSES) == 10L)

# TF-IDF vocabulary and IDF are fit from supplied training text only.
tfidf <- selector_fit_tfidf(c("trainonly shared", "shared trainonly", "shared token"))
stopifnot("trainonly" %in% tfidf$vocabulary, !"unseen" %in% tfidf$vocabulary)
unseen <- selector_apply_tfidf("unseen unseen", tfidf)
stopifnot(sum(unseen) == 0)

# Multinomial probabilities preserve the fixed class order and normalize exactly.
if (requireNamespace("glmnet", quietly = TRUE)) {
  set.seed(310)
  model_x <- matrix(rnorm(120 * 8), 120, 8)
  colnames(model_x) <- sprintf("e%03d", 0:7)
  model_y <- rep(SELECTOR_CLASSES, 12)
  model_x[cbind(seq_len(120), rep(seq_len(8), length.out = 120))] <-
    model_x[cbind(seq_len(120), rep(seq_len(8), length.out = 120))] + 2
  model <- selector_fit_glmnet(model_x, model_y, alpha = 0, lambda = 0.1)
  probability <- selector_predict_glmnet(model, model_x[1:7, , drop = FALSE])
  link <- predict(model$fit, selector_apply_preprocessor(model_x[1:7, , drop = FALSE], model$prep),
                  s = model$lambda, type = "link")[, , 1L]
  stable_probability <- selector_softmax(link[, SELECTOR_CLASSES, drop = FALSE])
  stopifnot(identical(colnames(probability), SELECTOR_CLASSES),
            max(abs(rowSums(probability) - 1)) < 1e-10,
            max(abs(probability - stable_probability)) < 1e-12)
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    portable_path <- tempfile(fileext = ".json")
    portable <- selector_export_logistic(model, "context", colnames(model_x), 0, portable_path)
    z <- sweep(sweep(model_x[1:7, , drop = FALSE], 2, unlist(portable$preprocessing$center), "-"),
               2, unlist(portable$preprocessing$scale), "/")
    z <- z[, unlist(portable$preprocessing$selected_indices) + 1L, drop = FALSE]
    portable_probability <- selector_softmax(sweep(z %*% t(as.matrix(portable$model$coefficients)),
                                                    2, unlist(portable$model$intercept), "+"))
    stopifnot(max(abs(portable_probability - probability)) < 1e-8)
  }

  cv_groups <- rep(sprintf("cvgroup%d", 1:12), each = 10)
  cv_text <- paste("shared shared", cv_groups)
  cv_y <- rep(SELECTOR_CLASSES, 12)
  cv <- selector_cv_tune_tfidf(cv_text, cv_y, cv_groups,
                               data.frame(alpha = 0, lambda = 0.1), max_terms = 100L)
  for (f in 1:3) {
    heldout_terms <- unique(cv_groups[cv$folds == f])
    stopifnot(!length(intersect(heldout_terms, cv$fold_vocabularies[[f]])))
  }
}

# row_id joins reject extras, omissions, duplicates, and preserve the rows.csv order.
temporary <- tempfile("selector-test-"); dir.create(temporary); on.exit(unlink(temporary, recursive = TRUE))
rows <- data.frame(row_id = c("003", "001", "002"), stringsAsFactors = FALSE)
features <- data.frame(row_id = c("001", "002", "003"), matrix(seq_len(9), 3, 3), check.names = FALSE)
names(features)[-1] <- selector_embedding_names(3)
path <- file.path(temporary, "features.csv"); pmr_write_csv(features, path)
joined <- selector_read_embeddings(path, rows, selector_embedding_names(3))
stopifnot(identical(rownames(joined), rows$row_id))
bad <- features[-1, ]; pmr_write_csv(bad, path)
assert_error(selector_read_embeddings(path, rows, selector_embedding_names(3)), "mismatch")

# Portable softmax is numerically stable and rows sum to one.
p <- selector_softmax(matrix(c(1000, 999, -1000, -999), 2, byrow = TRUE))
stopifnot(all(is.finite(p)), max(abs(rowSums(p) - 1)) < 1e-12)

# Frozen test evaluation rejects an absent marker before touching test labels.
assert_error(selector_check_frozen(temporary, "missing.csv"), "locked")

# The standalone R gate checks source bytes and the actual R/package runtime too.
required_source <- c("R/common.R", "R/selector/core.R", "R/selector/evaluate_test.R")
for (rel in c(required_source, "data.csv")) {
  target <- file.path(temporary, rel); dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  writeLines("frozen input", target)
}
frozen_file <- file.path(temporary, "artifacts/selector/frozen.json")
marker <- list(artifact_hashes = list(data.csv = selector_sha256(file.path(temporary, "data.csv"))),
               hashes = setNames(lapply(required_source, function(p) selector_sha256(file.path(temporary, p))), required_source),
               r_runtime = selector_runtime())
pmr_write_json(marker, frozen_file)
stopifnot(selector_check_frozen(temporary, "data.csv"))
writeLines("modified prediction helper", file.path(temporary, "R/common.R"))
assert_error(selector_check_frozen(temporary, "data.csv"), "hash mismatch")
writeLines("frozen input", file.path(temporary, "R/common.R"))
marker$r_runtime$R <- "0.0.0"; pmr_write_json(marker, frozen_file)
assert_error(selector_check_frozen(temporary, "data.csv"), "runtime changed")

# Evaluation covers every candidate, preserves identities, and resolves exact ties lexically.
truth <- c(SELECTOR_CLASSES[[1]], SELECTOR_CLASSES[[2]])
tie <- matrix(0.1, 2, 10, dimnames = list(NULL, SELECTOR_CLASSES))
perfect <- matrix(0, 2, 10, dimnames = list(NULL, SELECTOR_CLASSES))
perfect[cbind(1:2, 1:2)] <- 1
evaluation <- selector_evaluation_tables(list(tied = tie, perfect = perfect), truth,
                                         selected_name = "perfect", deployed_name = "tied")
stopifnot(nrow(evaluation$overall) == 2L, nrow(evaluation$per_class) == 20L,
          nrow(evaluation$confusion) == 200L, nrow(evaluation$confidence) == 20L,
          evaluation$overall$selected[evaluation$overall$candidate == "perfect"],
          evaluation$overall$deployed[evaluation$overall$candidate == "tied"],
          evaluation$overall$macro_f1[evaluation$overall$candidate == "perfect"] == 0.2)
tied_confusion <- subset(evaluation$confusion, candidate == "tied" & count > 0)
stopifnot(all(tied_confusion$predicted == SELECTOR_CLASSES[[1]]))
bad_probability <- tie; bad_probability[1, 1] <- Inf
assert_error(selector_check_probabilities(bad_probability), "Non-finite")
assert_error(selector_check_glmnet_fit(list(jerr = -1L), 0.1), "converge")
assert_error(selector_check_glmnet_fit(list(jerr = 0L, lambda = c(1, 0.2)), 0.1), "requested lambda")
cat("PASS: grouped isolation, stable class order, constant features, train-only TF-IDF, exact row joins, stable probabilities, and test freeze gate.\n")
