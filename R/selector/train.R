#!/usr/bin/env Rscript
root_default <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
root <- if (length(commandArgs(TRUE))) normalizePath(commandArgs(TRUE)[[1]], winslash = "/", mustWork = TRUE) else root_default
source(file.path(root, "R", "common.R")); source(file.path(root, "R", "selector", "core.R"))
pmr_use_library(root)

selector_training_signature <- function(root, data_paths, config) {
  source_paths <- c("R/common.R", "R/selector/core.R", "R/selector/train.R")
  missing <- c(data_paths, source_paths)[!file.exists(file.path(root, c(data_paths, source_paths)))]
  if (length(missing)) stop("Cannot sign missing training input: ", missing[[1]], call. = FALSE)
  list(
    schema_version = "selector-candidate-signature-v1",
    data_hashes = setNames(lapply(data_paths, function(path) selector_sha256(file.path(root, path))), data_paths),
    source_hashes = setNames(lapply(source_paths, function(path) selector_sha256(file.path(root, path))), source_paths),
    runtime = selector_runtime(),
    config = config
  )
}

selector_checkpoint_write <- function(record, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  saveRDS(record, temporary, version = 3L)
  if (file.exists(path) || !file.rename(temporary, path))
    stop("Refusing to overwrite candidate checkpoint: ", basename(path), call. = FALSE)
  invisible(record)
}

selector_checkpoint_read <- function(path, candidate_name, signature) {
  record <- tryCatch(readRDS(path), error = function(error)
    stop("Unreadable candidate checkpoint ", basename(path), ": ", conditionMessage(error), call. = FALSE))
  required <- c("schema_version", "candidate", "status", "signature", "signature_sha256")
  if (!is.list(record) || !all(required %in% names(record)) ||
      !identical(record$schema_version, "selector-candidate-checkpoint-v1") ||
      !identical(record$candidate, candidate_name))
    stop("Invalid candidate checkpoint: ", basename(path), call. = FALSE)
  expected_hash <- digest::digest(signature, algo = "sha256", serialize = TRUE)
  if (!identical(record$signature, signature) || !identical(record$signature_sha256, expected_hash))
    stop("Candidate checkpoint signature mismatch for ", candidate_name,
         "; refusing stale reuse.", call. = FALSE)
  if (identical(record$status, "failed"))
    stop("Candidate checkpoint records an earlier failure for ", candidate_name, ": ",
         record$error, call. = FALSE)
  if (!identical(record$status, "success") || is.null(record$payload))
    stop("Invalid candidate checkpoint status for ", candidate_name, call. = FALSE)
  record$payload
}

selector_checkpoint_run <- function(candidate_name, signature, checkpoint_dir, trainer) {
  path <- file.path(checkpoint_dir, paste0(candidate_name, ".rds"))
  if (file.exists(path)) {
    message("Resuming exact candidate checkpoint: ", candidate_name)
    return(selector_checkpoint_read(path, candidate_name, signature))
  }
  signature_hash <- digest::digest(signature, algo = "sha256", serialize = TRUE)
  result <- tryCatch(trainer(), error = function(error) {
    selector_checkpoint_write(list(
      schema_version = "selector-candidate-checkpoint-v1",
      candidate = candidate_name,
      status = "failed",
      signature = signature,
      signature_sha256 = signature_hash,
      cv_scores = NULL,
      error = conditionMessage(error)
    ), path)
    stop(conditionMessage(error), call. = FALSE)
  })
  if (!is.list(result) || is.null(result$candidate) || is.null(result$tuning) ||
      is.null(result$tuning$scores))
    stop("Candidate trainer returned an incomplete result for ", candidate_name, call. = FALSE)
  selector_checkpoint_write(list(
    schema_version = "selector-candidate-checkpoint-v1",
    candidate = candidate_name,
    status = "success",
    signature = signature,
    signature_sha256 = signature_hash,
    cv_scores = unname(as.numeric(result$tuning$scores)),
    payload = result
  ), path)
  message("Saved candidate checkpoint: ", candidate_name)
  result
}

main <- function() {
  if (file.exists(file.path(root, "artifacts", "selector", "frozen.json")))
    stop("Training is locked: remove or supersede the frozen protocol through the parent workflow before retraining.", call. = FALSE)
  pmr_require(c("glmnet", "ranger", "jsonlite", "digest", "Matrix"))
  rows_path <- file.path(root, "data", "processed", "selector", "rows.csv")
  context_path <- file.path(root, "artifacts", "selector", "features_context.csv")
  response_path <- file.path(root, "artifacts", "selector", "features_response.csv")
  situation_path <- file.path(root, "artifacts", "selector", "features_situation.csv")
  rows <- pmr_read_csv(rows_path); selector_validate_rows(rows)
  context <- selector_read_embeddings(context_path, rows)
  response <- selector_read_embeddings(response_path, rows)
  situation <- selector_read_embeddings(situation_path, rows)
  train <- rows$split == "train"; val <- rows$split == "val"
  if (!any(train) || !any(val)) stop("Both train and validation rows are required.", call. = FALSE)
  y <- rows$label; groups <- rows$group_id
  glm_grid <- expand.grid(alpha = c(0, 1), lambda = c(0.01, 0.1))
  rf_grid <- data.frame(mtry = c(24L, 48L), min_node_size = c(10L, 50L))
  training_config <- list(grouped_cv_folds = 3L, random_seed = 310L,
    random_forest_num_trees = 100L, random_forest_num_threads = 4L,
    glmnet_warm_start_points = 30L, glmnet_grid = glm_grid,
    random_forest_grid = rf_grid, tfidf_max_terms = 2000L,
    class_order = SELECTOR_CLASSES, embedding_features = selector_embedding_names(),
    fold_local_preprocessing = c("normalization", "constant_feature_removal",
                                 "tfidf_vocabulary", "tfidf_idf"))
  data_paths <- c("data/processed/selector/rows.csv", "artifacts/selector/features_context.csv",
                  "artifacts/selector/features_response.csv", "artifacts/selector/features_situation.csv")
  signature <- selector_training_signature(root, data_paths, training_config)
  checkpoint_dir <- file.path(root, "artifacts", "selector", "train_checkpoints")
  sources <- list(context = context, response = response)
  candidates <- list(); tuning <- list()
  for (source_name in names(sources)) {
    x <- sources[[source_name]]
    name <- paste0(source_name, "_glmnet")
    result <- selector_checkpoint_run(name, signature, checkpoint_dir, function() {
      tuned <- selector_cv_tune(x[train, , drop = FALSE], y[train], groups[train], "glmnet", glm_grid)
      p <- tuned$params
      fit <- selector_fit_glmnet(x[train, , drop = FALSE], y[train], p$alpha, p$lambda)
      prob <- selector_predict_glmnet(fit, x[val, , drop = FALSE])
      list(candidate = list(model = fit, source = source_name, probabilities = prob,
        validation_macro_f1 = selector_macro_f1(y[val], selector_classify(prob))), tuning = tuned)
    })
    candidates[[name]] <- result$candidate; tuning[[name]] <- result$tuning
    name_rf <- paste0(source_name, "_rf")
    result_rf <- selector_checkpoint_run(name_rf, signature, checkpoint_dir, function() {
      tuned_rf <- selector_cv_tune(x[train, , drop = FALSE], y[train], groups[train], "rf", rf_grid)
      pr <- tuned_rf$params
      fit_rf <- selector_fit_rf(x[train, , drop = FALSE], y[train], pr$mtry, pr$min_node_size)
      prob_rf <- selector_predict_rf(fit_rf, x[val, , drop = FALSE])
      list(candidate = list(model = fit_rf, source = source_name, probabilities = prob_rf,
        validation_macro_f1 = selector_macro_f1(y[val], selector_classify(prob_rf))), tuning = tuned_rf)
    })
    candidates[[name_rf]] <- result_rf$candidate; tuning[[name_rf]] <- result_rf$tuning
  }
  result_tfidf <- selector_checkpoint_run("tfidf_glmnet", signature, checkpoint_dir, function() {
    tfidf <- selector_fit_tfidf(rows$model_text[train])
    if (!length(tfidf$vocabulary)) stop("TF-IDF training data has no eligible vocabulary.", call. = FALSE)
    x_train <- selector_apply_tfidf(rows$model_text[train], tfidf)
    x_val <- selector_apply_tfidf(rows$model_text[val], tfidf)
    tuned <- selector_cv_tune_tfidf(rows$model_text[train], y[train], groups[train], glm_grid)
    p <- tuned$params; fit <- selector_fit_glmnet(x_train, y[train], p$alpha, p$lambda)
    prob <- selector_predict_glmnet(fit, x_val)
    list(candidate = list(model = fit, source = "model_text_tfidf", probabilities = prob,
      validation_macro_f1 = selector_macro_f1(y[val], selector_classify(prob)), tfidf = tfidf),
      tuning = tuned)
  })
  candidates$tfidf_glmnet <- result_tfidf$candidate; tuning$tfidf_glmnet <- result_tfidf$tuning
  f1 <- vapply(candidates, `[[`, numeric(1), "validation_macro_f1")
  selected_name <- names(which.max(f1)); selected <- candidates[[selected_name]]
  logistic_names <- c("context_glmnet", "response_glmnet")
  logistic_name <- logistic_names[which.max(f1[logistic_names])]
  logistic <- candidates[[logistic_name]]
  out <- file.path(root, "artifacts", "selector"); model_dir <- file.path(root, "models", "selector")
  dir.create(out, recursive = TRUE, showWarnings = FALSE); dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
  val_prob <- data.frame(row_id = rows$row_id[val], selected$probabilities, check.names = FALSE)
  pmr_write_csv(val_prob, file.path(out, "val_probabilities.csv"))
  metrics <- data.frame(candidate = names(f1), validation_macro_f1 = unname(f1), selected = names(f1) == selected_name)
  pmr_write_csv(metrics, file.path(out, "validation_metrics.csv"))
  pmr_write_json(list(selection_metric = "validation_macro_f1", chosen_model = selected_name,
    deployed_model = logistic_name, candidates = as.list(f1)), file.path(out, "selection.json"))
  pmr_write_json(training_config, file.path(out, "training_config.json"))
  cv_table <- do.call(rbind, lapply(names(tuning), function(name) {
    grid <- if (grepl("_rf$", name)) rf_grid else glm_grid
    for (column in setdiff(c("alpha", "lambda", "mtry", "min_node_size"), names(grid))) grid[[column]] <- NA_real_
    data.frame(candidate = name, grid[, c("alpha", "lambda", "mtry", "min_node_size")],
               cv_macro_f1 = tuning[[name]]$scores)
  }))
  pmr_write_csv(cv_table, file.path(out, "cv_metrics.csv"))
  saveRDS(list(candidates = candidates, selected_name = selected_name, deployed_name = logistic_name,
               class_order = SELECTOR_CLASSES,
               feature_names = selector_embedding_names()), file.path(out, "model_bundle.rds"))
  selector_export_logistic(logistic$model, logistic$source, selector_embedding_names(),
                           logistic$validation_macro_f1, file.path(model_dir, "logistic.json"))
  fixture_index <- head(which(val), 20L)
  fixture_x <- sources[[logistic$source]][fixture_index, , drop = FALSE]
  fixture_prob <- selector_predict_glmnet(logistic$model, fixture_x)
  pmr_write_csv(data.frame(row_id = rows$row_id[fixture_index], fixture_x, check.names = FALSE), file.path(out, "parity_fixture.csv"))
  pmr_write_csv(data.frame(row_id = rows$row_id[fixture_index], fixture_prob, check.names = FALSE), file.path(out, "parity_expected_probabilities.csv"))
  cluster_rows <- which(train & !duplicated(ifelse(train, rows$model_context, NA_character_)))
  cluster_x <- situation[cluster_rows, , drop = FALSE]
  if (nrow(cluster_x) < 30L) stop("At least 30 training rows are required for k-means.", call. = FALSE)
  norms <- sqrt(rowSums(cluster_x^2)); if (any(norms == 0)) stop("Zero-norm context embedding cannot be clustered.", call. = FALSE)
  set.seed(310); km <- stats::kmeans(cluster_x / norms, centers = 30L, iter.max = 100L, nstart = 5L)
  centers <- km$centers / sqrt(rowSums(km$centers^2))
  pmr_write_json(list(schema_version = "selector-clusters-v1", feature_source = "situation",
    input_feature_order = selector_embedding_names(), preprocessing = list(l2_normalize = TRUE),
    assignment = "maximum_cosine_similarity", training_rows = nrow(cluster_x),
    deduplication_key = "model_context", centers = unname(centers)), file.path(out, "clusters.json"))
  inputs <- c("data/processed/selector/rows.csv", "artifacts/selector/features_context.csv",
              "artifacts/selector/features_response.csv", "artifacts/selector/features_situation.csv",
              "artifacts/selector/model_bundle.rds")
  pmr_write_json(list(schema_version = "selector-training-manifest-v1",
    artifact_hashes = setNames(lapply(inputs, function(p) selector_sha256(file.path(root, p))), inputs)),
    file.path(out, "training_manifest.json"))
  message("Selected ", selected_name, "; validation Macro-F1 = ", signif(max(f1), 5),
          ". Test split was not scored.")
}
if (!identical(Sys.getenv("SELECTOR_TRAIN_SOURCE_ONLY"), "1")) pmr_main(main)
