#!/usr/bin/env Rscript
# Offline behavioral checks; fixtures are intentionally synthetic, not metrics.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/")
source(file.path(root, "R", "common.R"))
source(file.path(root, "R", "prepare.R"))
source(file.path(root, "R", "models.R"))
pmr_use_library(root)
pmr_require(c("jsonlite", "ranger"))
assert_error <- function(expression, pattern) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  if (is.null(error) || !grepl(pattern, conditionMessage(error), ignore.case = TRUE)) {
    stop("Expected error matching: ", pattern)
  }
}
fixture <- data.frame(query_id = c("001", "002", "003"), persona_id = c("p1", "p2", "p3"),
  split = c("train", "val", "benchmark"), query = c("Where do I work?", "What do I eat?", "Where do I live?"),
  history_path = c("a.json", "b.json", "c.json"),
  gold_json = '[{"role":"user","content":"I work at a library."}]',
  updated = c("true", "0", "1"), who = "user", topic_query = "work", extra = "retain",
  stringsAsFactors = FALSE)
dirty <- rbind(fixture, fixture[1, ], transform(fixture[1, ], query_id = "004", query = ""),
               transform(fixture[1, ], query_id = "005", history_path = "", query = "Other?"),
               transform(fixture[1, ], query_id = "006", gold_json = "[]", query = "Another?"))
cleaned <- pmr_clean_queries(dirty)
stopifnot(nrow(cleaned$data) == 3L, cleaned$report$excluded_rows == 4L,
          all(cleaned$data$extra == "retain"), identical(cleaned$data$query_id, fixture$query_id))
source_text <- fixture
source_text$query[1] <- "  Where do I work?\r\n"
source_text$gold_json[1] <- paste0(" \n", source_text$gold_json[1], "\n ")
source_text$original_split <- c("train", "train", "benchmark")
source_clean <- pmr_clean_queries(source_text)$data
stopifnot(identical(source_clean$query, source_text$query),
          identical(source_clean$gold_json, source_text$gold_json),
          identical(source_clean$original_split, source_text$original_split))
assert_error(pmr_clean_queries(transform(fixture, persona_id = "shared")), "Cross-split")
wrapped <- fixture
wrapped$query[1] <- "user: Where do I work?"
assert_error(pmr_clean_queries(wrapped), "role wrapper")
assert_error(pmr_clean_queries(rbind(fixture, transform(fixture[1, ], query_id = "999"))), "Duplicate question")
bad_bool <- fixture
bad_bool$updated[1] <- "maybe"
assert_error(pmr_clean_queries(bad_bool), "updated")

set.seed(8123)
make_features <- function(personas, prefix) {
  n <- length(personas) * 48L
  persona_id <- rep(personas, each = 48L)
  query_id <- paste0(prefix, "q", rep(seq_len(length(personas) * 4L), each = 12L))
  data <- data.frame(query_id = query_id, persona_id = persona_id,
    memory_id = paste0(persona_id, "m", rep(seq_len(48L), times = length(personas))),
    bm25 = runif(n, 0, 5), semantic = runif(n, -0.2, 0.9), overlap = runif(n),
    log_length = runif(n, 1, 5), position = runif(n), rrf = runif(n, 0.01, 0.04),
    label = rep(c(0, 1, 0, 0), length.out = n), token_count = sample(10:80, n, TRUE),
    stringsAsFactors = FALSE)
  data
}
train <- make_features(paste0("train", 1:6), "0")
val <- make_features(paste0("val", 1:2), "v")
temporary <- tempfile("pmr-r-test-")
dir.create(temporary)
on.exit_cleanup <- function() unlink(temporary, recursive = TRUE)
result <- pmr_train(train, val, temporary)
stopifnot(result$report$grouped_cv_folds == 5L,
          nrow(result$cv_predictions) == nrow(train),
          !anyDuplicated(result$cv_predictions[c("query_id", "memory_id")]),
          nrow(result$assignments) == 6L)
stopifnot(all(is.finite(result$models$logistic$coefficients)),
          max(abs(result$models$logistic$means - colMeans(train[PMR_FEATURES]))) < 1e-12,
          !"position" %in% result$models$logistic_no_position$feature_names)
for (fold in result$report$folds) {
  train_personas <- unlist(fold$train_personas)
  heldout_personas <- unlist(fold$heldout_personas)
  stopifnot(!length(intersect(train_personas, heldout_personas)))
  expected <- colMeans(train[train$persona_id %in% train_personas, PMR_FEATURES])
  stopifnot(max(abs(expected - unlist(fold$normalization$means))) < 1e-12)
}
json_model <- jsonlite::fromJSON(file.path(temporary, "logistic.json"))
normalized <- sweep(sweep(as.matrix(val[json_model$feature_names]), 2, json_model$means, "-"),
                    2, json_model$scales, "/")
json_prediction <- plogis(drop(normalized %*% json_model$coefficients) + json_model$intercept)
stopifnot(max(abs(json_prediction - result$val_predictions$logistic)) < 1e-12)
frozen <- pmr_predict_models(readRDS(file.path(temporary, "models.rds")), val)
stopifnot(identical(frozen, result$val_predictions))
glm_values <- data.frame(sweep(sweep(as.matrix(val[PMR_FEATURES]), 2, result$models$logistic$means, "-"),
                              2, result$models$logistic$scales, "/"))
stopifnot(max(abs(as.numeric(predict(result$models$logistic$fit, glm_values, type = "response")) -
                  json_prediction)) < 1e-12)
bad <- val
bad$persona_id[1] <- train$persona_id[1]
assert_error(pmr_check_partition(train, bad), "persona leakage")
bad <- train
bad$semantic[1] <- Inf
assert_error(pmr_validate_candidates(bad), "Non-finite")
assert_error(pmr_validate_candidates(transform(train, gold_text = "leak")), "Unexpected columns")
bad <- train
bad$label[1] <- 2
assert_error(pmr_validate_candidates(bad), "binary")
assert_error(pmr_group_folds(rep("only", 4)), "at least two")
stopifnot(max(pmr_group_folds(c("a", "b"))$fold) == 2L)
constant <- train
constant$position <- 0
constant$overlap <- constant$bm25
constant_model <- pmr_fit_logistic(constant)
stopifnot(all(is.finite(constant_model$coefficients)),
          "position" %in% constant_model$inactive_features,
          "overlap" %in% constant_model$inactive_features)
id_path <- file.path(temporary, "ids.csv")
pmr_write_csv(fixture, id_path)
stopifnot(identical(pmr_read_csv(id_path)$query_id, fixture$query_id))
unicode_path <- file.path(temporary, "unicode-multiline.csv")
unicode_query <- paste0("Caf\u00e9, \u4f60\u597d \U0001f600\n", 'He said "hello"; keep this line.')
unicode_gold <- '[{"role":"user","content":"Caf\u00e9 \u4f60\u597d \U0001f600\\nexact"}]'
csv_quote <- function(x) paste0('"', gsub('"', '""', x, fixed = TRUE), '"')
unicode_csv <- paste0("query_id,query,gold_json\r\n", csv_quote("001"), ",",
                      csv_quote(unicode_query), ",", csv_quote(unicode_gold), "\r\n")
# This is Python-style UTF-8 CSV: write bytes without invoking R's CSV writer.
writeBin(charToRaw(enc2utf8(unicode_csv)), unicode_path)
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "C")))
unicode_read <- pmr_read_csv(unicode_path)
stopifnot(identical(unicode_read$query, unicode_query), identical(unicode_read$gold_json, unicode_gold),
          identical(unicode_read$query_id, "001"))
unicode_output <- file.path(temporary, "unicode-output.csv")
pmr_write_csv(unicode_read, unicode_output)
stopifnot(identical(pmr_read_csv(unicode_output), unicode_read))
pmr_use_pandoc(root)
if (requireNamespace("rmarkdown", quietly = TRUE) && rmarkdown::pandoc_available()) {
  pmr_write_csv(cleaned$data, file.path(temporary, "data/processed/queries.csv"))
  pmr_write_csv(dirty, file.path(temporary, "data/intermediate/queries.csv"))
  pmr_write_json(cleaned$report, file.path(temporary, "reports/cleaning.json"))
  pmr_write_csv(data.frame(query_id = fixture$query_id, persona_id = fixture$persona_id,
    split = fixture$split, status = "exact", gold_count = 1L, memory_count = 48L,
    updated = fixture$updated, topic_query = fixture$topic_query,
    gold_relative_position = c(0.3, 0.6, 0.8)), file.path(temporary, "reports/alignment.csv"))
  pmr_write_csv(train, file.path(temporary, "data/processed/features_train.csv"))
  pmr_write_csv(val, file.path(temporary, "data/processed/features_val.csv"))
  html <- rmarkdown::render(file.path(root, "R/eda.Rmd"), output_file = "fixture.html",
    output_dir = temporary, params = list(project_root = temporary),
    envir = new.env(parent = globalenv()), quiet = TRUE)
  stopifnot(file.exists(html), file.info(html)$size > 10000L)
  rendered_html <- paste(readLines(html, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  caption_tags <- regmatches(rendered_html,
    gregexpr("(?s)<caption[^>]*>.*?</caption>", rendered_html, perl = TRUE))[[1]]
  rendered_captions <- trimws(gsub("\\s+", " ", gsub("<[^>]+>", "", caption_tags), perl = TRUE))
  for (caption in c("Data-cleaning counts",
                    "Alignment status by partition; zero-count combinations are shown",
                    "Observed candidate labels: zero means unannotated, not verified irrelevant")) {
    if (!caption %in% rendered_captions) stop("Missing rendered table caption: ", caption)
  }
  cat("PASS: reproducible EDA notebook renders to HTML from synthetic fixtures.\n")
  pmr_write_csv(data.frame(method = "logistic", budget_recall = 0.9),
                file.path(temporary, "reports/benchmark_summary.csv"))
  assert_error(rmarkdown::render(file.path(root, "R/eda.Rmd"), output_file = "orphan.html",
    output_dir = temporary, params = list(project_root = temporary),
    envir = new.env(parent = globalenv()), quiet = TRUE),
    "Incomplete evaluation artifacts for benchmark")
  cat("PASS: notebook rejects an orphan benchmark summary before plotting.\n")
}
on.exit_cleanup()
cat("PASS: cleaning, isolation, schema, 5-fold grouped OOF, train-only scaling, finite models, JSON/glm/RDS parity, and identifier preservation.\n")
