# Model fitting functions. Gold text and identifiers never enter a model matrix.
PMR_FEATURES <- c("bm25", "semantic", "overlap", "log_length", "position")
PMR_CANDIDATE_COLUMNS <- c("query_id", "persona_id", "memory_id", PMR_FEATURES,
                           "rrf", "label", "token_count")

pmr_validate_candidates <- function(data, labels_required = TRUE) {
  expected <- if (labels_required) PMR_CANDIDATE_COLUMNS else setdiff(PMR_CANDIDATE_COLUMNS, "label")
  # Frozen prediction can consume a labeled evaluation file without using label.
  if (!labels_required && "label" %in% names(data)) expected <- PMR_CANDIDATE_COLUMNS
  pmr_require_columns(data, expected, exact = TRUE)
  if (!nrow(data)) stop("Candidate data must contain rows.", call. = FALSE)
  for (name in c("query_id", "persona_id", "memory_id")) {
    data[[name]] <- as.character(data[[name]])
    if (any(!pmr_nonempty(data[[name]])) || any(data[[name]] != trimws(data[[name]]))) {
      stop("Invalid identifier in ", name, ".", call. = FALSE)
    }
  }
  if (anyDuplicated(data[c("query_id", "memory_id")])) stop("Duplicate query_id/memory_id pair.", call. = FALSE)
  query_groups <- split(data$persona_id, data$query_id)
  if (any(vapply(query_groups, function(x) length(unique(x)) != 1L, logical(1)))) {
    stop("A query_id maps to multiple personas.", call. = FALSE)
  }
  memory_groups <- split(data$persona_id, data$memory_id)
  if (any(vapply(memory_groups, function(x) length(unique(x)) != 1L, logical(1)))) {
    stop("A memory_id maps to multiple personas.", call. = FALSE)
  }
  for (name in intersect(c(PMR_FEATURES, "rrf", "token_count", "label"), names(data))) {
    data[[name]] <- pmr_finite_number(data[[name]], name)
  }
  if ("label" %in% names(data) && any(!data$label %in% c(0, 1))) stop("label must be binary 0/1.", call. = FALSE)
  if (any(data$token_count < 0 | data$token_count != floor(data$token_count))) {
    stop("token_count must be a nonnegative integer.", call. = FALSE)
  }
  data
}

pmr_check_partition <- function(train, val) {
  if (length(intersect(train$persona_id, val$persona_id))) stop("Train/validation persona leakage.", call. = FALSE)
  if (length(intersect(train$query_id, val$query_id))) stop("Train/validation query leakage.", call. = FALSE)
  invisible(TRUE)
}

pmr_fit_logistic <- function(data, features = PMR_FEATURES) {
  if (length(unique(data$label)) != 2L) stop("Logistic training needs both label classes.", call. = FALSE)
  values <- as.matrix(data[features])
  means <- colMeans(values)
  scales <- apply(values, 2L, sd)
  constant <- !is.finite(scales) | scales == 0
  scales[constant] <- 1
  normalized <- sweep(sweep(values, 2L, means, "-"), 2L, scales, "/")
  # Keep the complete public feature schema while assigning zero to redundant
  # columns. This yields an identifiable glm with finite exported coefficients.
  design <- cbind("(Intercept)" = 1, normalized)
  independent <- qr(design, tol = 1e-10)
  retained <- sort(independent$pivot[seq_len(independent$rank)])
  active <- features[(seq_along(features) + 1L) %in% retained]
  model_data <- data.frame(label = data$label, normalized[, active, drop = FALSE], check.names = FALSE)
  formula <- if (length(active)) reformulate(active, response = "label") else label ~ 1
  warnings <- character()
  fit <- withCallingHandlers(
    glm(formula, data = model_data, family = binomial(), control = glm.control(maxit = 100L)),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  if (!isTRUE(fit$converged) || any(!is.finite(coef(fit)))) {
    stop("Logistic fit did not converge with finite coefficients.", call. = FALSE)
  }
  coefficients <- setNames(rep(0, length(features)), features)
  coefficients[active] <- coef(fit)[active]
  list(fit = fit, feature_names = features, means = means, scales = scales,
       coefficients = coefficients, intercept = unname(coef(fit)["(Intercept)"]),
       inactive_features = setdiff(features, active), warnings = unique(warnings))
}

pmr_predict_logistic <- function(model, data) {
  values <- as.matrix(data[model$feature_names])
  normalized <- sweep(sweep(values, 2L, model$means, "-"), 2L, model$scales, "/")
  unname(plogis(drop(normalized %*% model$coefficients) + model$intercept))
}

pmr_export_logistic <- function(model) {
  list(schema_version = 1L, model_type = "logistic", feature_version = "pmr-v1",
       feature_names = unname(as.list(model$feature_names)), means = unname(as.list(model$means)),
       scales = unname(as.list(model$scales)), coefficients = unname(as.list(model$coefficients)),
       intercept = model$intercept)
}

pmr_fit_models <- function(data) {
  pmr_require("ranger")
  logistic <- pmr_fit_logistic(data)
  no_position <- pmr_fit_logistic(data, setdiff(PMR_FEATURES, "position"))
  forest_data <- data.frame(label = factor(data$label, levels = c(0, 1)), data[PMR_FEATURES])
  random_forest <- ranger::ranger(label ~ ., data = forest_data, probability = TRUE,
    num.trees = 300L, seed = 310L, mtry = 2L, min.node.size = 10L, num.threads = 2L,
    write.forest = TRUE)
  list(schema_version = 1L, feature_version = "pmr-v1", logistic = logistic,
       logistic_no_position = no_position, random_forest = random_forest)
}

pmr_predict_models <- function(models, data) {
  if (!identical(models$feature_version, "pmr-v1") || models$schema_version != 1L) {
    stop("Unsupported frozen model version.", call. = FALSE)
  }
  predictions <- data.frame(query_id = data$query_id, memory_id = data$memory_id,
    logistic = pmr_predict_logistic(models$logistic, data),
    logistic_no_position = pmr_predict_logistic(models$logistic_no_position, data),
    random_forest = as.numeric(predict(models$random_forest, data = data[PMR_FEATURES],
                                      num.threads = 2L)$predictions[, "1"]), stringsAsFactors = FALSE)
  probabilities <- as.matrix(predictions[c("logistic", "logistic_no_position", "random_forest")])
  if (any(!is.finite(probabilities) | probabilities < 0 | probabilities > 1)) {
    stop("Models produced invalid probabilities.", call. = FALSE)
  }
  predictions
}

pmr_group_folds <- function(persona_id) {
  personas <- sort(unique(persona_id))
  if (length(personas) < 2L) stop("Grouped CV requires at least two training personas.", call. = FALSE)
  folds <- min(5L, length(personas))
  set.seed(310L)
  shuffled <- sample(personas)
  data.frame(persona_id = shuffled, fold = rep(seq_len(folds), length.out = length(shuffled)),
             stringsAsFactors = FALSE)
}

pmr_train <- function(train, val, out) {
  train <- pmr_validate_candidates(train)
  val <- pmr_validate_candidates(val)
  pmr_check_partition(train, val)
  assignments <- pmr_group_folds(train$persona_id)
  row_folds <- assignments$fold[match(train$persona_id, assignments$persona_id)]
  cv <- vector("list", max(assignments$fold))
  fold_reports <- vector("list", length(cv))
  for (fold in seq_along(cv)) {
    held_out <- row_folds == fold
    fold_models <- pmr_fit_models(train[!held_out, , drop = FALSE])
    predictions <- pmr_predict_models(fold_models, train[held_out, , drop = FALSE])
    predictions$persona_id <- train$persona_id[held_out]
    predictions$fold <- fold
    cv[[fold]] <- predictions
    fold_reports[[fold]] <- list(fold = fold,
      train_personas = unname(as.list(sort(unique(train$persona_id[!held_out])))),
      heldout_personas = unname(as.list(sort(unique(train$persona_id[held_out])))),
      train_rows = sum(!held_out), heldout_rows = sum(held_out),
      normalization = pmr_export_logistic(fold_models$logistic),
      inactive_features = as.list(fold_models$logistic$inactive_features),
      warnings = as.list(unique(c(fold_models$logistic$warnings, fold_models$logistic_no_position$warnings))))
    cat("Completed grouped fold", fold, "of", length(cv), "\n")
  }
  models <- pmr_fit_models(train)
  val_predictions <- pmr_predict_models(models, val)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  saveRDS(models, file.path(out, "models.rds"), version = 3L)
  saveRDS(models$logistic$fit, file.path(out, "logistic.rds"), version = 3L)
  saveRDS(models$logistic_no_position$fit, file.path(out, "logistic_no_position.rds"), version = 3L)
  saveRDS(models$random_forest, file.path(out, "random_forest.rds"), version = 3L)
  pmr_write_json(pmr_export_logistic(models$logistic), file.path(out, "logistic.json"))
  pmr_write_json(pmr_export_logistic(models$logistic_no_position), file.path(out, "logistic_no_position.json"))
  pmr_write_csv(val_predictions, file.path(out, "val_predictions.csv"))
  cv_predictions <- do.call(rbind, cv)
  pmr_write_csv(cv_predictions, file.path(out, "cv_predictions.csv"))
  pmr_write_csv(assignments, file.path(out, "cv_assignments.csv"))
  report <- list(schema_version = 1L, seed = 310L, feature_version = "pmr-v1",
    feature_names = as.list(PMR_FEATURES), train_rows = nrow(train), val_rows = nrow(val),
    train_personas = length(unique(train$persona_id)), val_personas = length(unique(val$persona_id)),
    grouped_cv_folds = length(cv), folds = fold_reports,
    training_files_scope = "Caller-supplied train and validation only; benchmark is not read.",
    logistic = list(inactive_features = as.list(models$logistic$inactive_features),
                    warnings = as.list(models$logistic$warnings)),
    logistic_no_position = list(inactive_features = as.list(models$logistic_no_position$inactive_features),
                                warnings = as.list(models$logistic_no_position$warnings)),
    random_forest = list(num_trees = 300L, mtry = 2L, min_node_size = 10L, num_threads = 2L),
    software = list(R = R.version.string, ranger = as.character(utils::packageVersion("ranger")),
                    jsonlite = as.character(utils::packageVersion("jsonlite"))))
  pmr_write_json(report, file.path(out, "training_report.json"))
  invisible(list(models = models, val_predictions = val_predictions,
                 cv_predictions = cv_predictions, assignments = assignments, report = report))
}
