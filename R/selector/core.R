SELECTOR_CLASSES <- sort(c(
  "agreeableness_high", "agreeableness_low",
  "conscientiousness_high", "conscientiousness_low",
  "extraversion_high", "extraversion_low",
  "neuroticism_high", "neuroticism_low",
  "openness_high", "openness_low"
))

selector_embedding_names <- function(n = 384L) sprintf("e%03d", seq_len(n) - 1L)

selector_validate_rows <- function(rows) {
  pmr_require_columns(rows, c("row_id", "group_id", "split", "role", "label", "model_text"))
  if (anyDuplicated(rows$row_id) || any(!pmr_nonempty(rows$row_id)))
    stop("row_id must be non-empty and unique.", call. = FALSE)
  if (any(!pmr_nonempty(rows$group_id))) stop("group_id must be non-empty.", call. = FALSE)
  if (any(!rows$split %in% c("train", "val", "test"))) stop("Invalid split.", call. = FALSE)
  if (any(!rows$role %in% c("train", "memory", "query"))) stop("Invalid role.", call. = FALSE)
  if (!setequal(unique(rows$label), SELECTOR_CLASSES))
    stop("Rows must contain exactly the ten fixed BIG5 labels.", call. = FALSE)
  split_by_group <- split(rows$split, rows$group_id)
  if (any(vapply(split_by_group, function(x) length(unique(x)) > 1L, logical(1))))
    stop("group_id leakage across splits.", call. = FALSE)
  invisible(rows)
}

selector_read_embeddings <- function(path, rows, expected = selector_embedding_names()) {
  if (!file.exists(path)) stop("Input does not exist: ", path, call. = FALSE)
  pmr_utf8_locale()
  emb <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = NULL,
                  encoding = "UTF-8", colClasses = c("character", rep("numeric", length(expected))))
  pmr_require_columns(emb, c("row_id", expected), exact = TRUE)
  if (anyDuplicated(emb$row_id) || !setequal(emb$row_id, rows$row_id))
    stop("Embedding row_id mismatch: IDs must match rows.csv exactly.", call. = FALSE)
  emb <- emb[match(rows$row_id, emb$row_id), , drop = FALSE]
  if (!identical(emb$row_id, rows$row_id)) stop("Embedding row order alignment failed.", call. = FALSE)
  x <- as.matrix(emb[expected])
  if (any(!is.finite(x))) stop("Non-finite or nonnumeric value in embedding.", call. = FALSE)
  dimnames(x) <- list(rows$row_id, expected)
  x
}

selector_group_folds <- function(group_id, k = 3L) {
  groups <- sort(unique(group_id))
  if (length(groups) < k) stop("Grouped CV requires at least three groups.", call. = FALSE)
  # Stable greedy assignment balances row counts without splitting a group.
  counts <- sort(table(group_id), decreasing = TRUE)
  loads <- integer(k); assignment <- integer(length(counts)); names(assignment) <- names(counts)
  for (g in names(counts)) {
    fold <- which.min(loads)
    assignment[[g]] <- fold
    loads[[fold]] <- loads[[fold]] + counts[[g]]
  }
  unname(assignment[group_id])
}

selector_preprocessor <- function(x) {
  if (inherits(x, "Matrix")) {
    center <- rep(0, ncol(x)); names(center) <- colnames(x)
    scale <- sqrt(Matrix::colMeans(x ^ 2)); names(scale) <- colnames(x)
    active <- is.finite(scale) & scale > 1e-12
    scale[!active] <- 1
    return(list(center = center, scale = scale, selected = unname(which(active)), sparse = TRUE))
  }
  center <- colMeans(x)
  scale <- apply(x, 2L, stats::sd)
  active <- is.finite(scale) & scale > 1e-12
  scale[!active] <- 1
  list(center = center, scale = scale, selected = unname(which(active)))
}

selector_apply_preprocessor <- function(x, prep) {
  if (isTRUE(prep$sparse)) {
    return(x[, prep$selected, drop = FALSE] %*%
             Matrix::Diagonal(x = 1 / prep$scale[prep$selected]))
  }
  z <- sweep(sweep(x, 2L, prep$center, "-"), 2L, prep$scale, "/")
  z[, prep$selected, drop = FALSE]
}

selector_macro_f1 <- function(truth, predicted, classes = SELECTOR_CLASSES) {
  scores <- vapply(classes, function(label) {
    tp <- sum(truth == label & predicted == label)
    fp <- sum(truth != label & predicted == label)
    fn <- sum(truth == label & predicted != label)
    if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  }, numeric(1))
  mean(scores)
}

selector_softmax <- function(logits) {
  if (any(!is.finite(logits))) stop("Non-finite model logits.", call. = FALSE)
  shifted <- logits - apply(logits, 1L, max)
  p <- exp(shifted)
  p / rowSums(p)
}

selector_check_probabilities <- function(probabilities) {
  if (!is.matrix(probabilities) || !identical(colnames(probabilities), SELECTOR_CLASSES))
    stop("Probability matrix must use the fixed BIG5 class order.", call. = FALSE)
  if (any(!is.finite(probabilities)) || any(probabilities < 0))
    stop("Non-finite or negative model probability.", call. = FALSE)
  totals <- rowSums(probabilities)
  if (any(!is.finite(totals)) || any(totals <= 0)) stop("Invalid probability row sum.", call. = FALSE)
  probabilities / totals
}

selector_classify <- function(probabilities) {
  p <- selector_check_probabilities(probabilities)
  # Fixed lexical class order plus ties.method='first' makes ties reproducible.
  SELECTOR_CLASSES[max.col(p, ties.method = "first")]
}

selector_evaluation_tables <- function(candidate_probabilities, truth, selected_name, deployed_name) {
  if (!length(candidate_probabilities) || anyDuplicated(names(candidate_probabilities)))
    stop("Candidate probabilities must be a uniquely named list.", call. = FALSE)
  if (!selected_name %in% names(candidate_probabilities) || !deployed_name %in% names(candidate_probabilities))
    stop("Selected and deployed model identities must name evaluated candidates.", call. = FALSE)
  if (any(!truth %in% SELECTOR_CLASSES)) stop("Unknown evaluation label.", call. = FALSE)
  overall <- list(); per_class <- list(); confusion <- list(); confidence <- list()
  bin_levels <- sprintf("%.1f-%.1f", seq(0, 0.9, 0.1), seq(0.1, 1, 0.1))
  for (name in names(candidate_probabilities)) {
    p <- selector_check_probabilities(candidate_probabilities[[name]])
    if (nrow(p) != length(truth)) stop("Probability/truth row mismatch.", call. = FALSE)
    predicted <- selector_classify(p)
    identity <- data.frame(candidate = name, selected = name == selected_name,
                           deployed = name == deployed_name, stringsAsFactors = FALSE)
    overall[[name]] <- cbind(identity, n = length(truth), macro_f1 = selector_macro_f1(truth, predicted))
    per_class[[name]] <- do.call(rbind, lapply(SELECTOR_CLASSES, function(label) {
      tp <- sum(truth == label & predicted == label); fp <- sum(truth != label & predicted == label)
      fn <- sum(truth == label & predicted != label); support <- sum(truth == label)
      precision <- if (tp + fp) tp / (tp + fp) else 0
      recall <- if (tp + fn) tp / (tp + fn) else 0
      f1 <- if (precision + recall) 2 * precision * recall / (precision + recall) else 0
      cbind(identity, class = label, support = support, true_positive = tp, false_positive = fp,
            false_negative = fn, precision = precision, recall = recall, f1 = f1)
    }))
    grid <- expand.grid(truth = SELECTOR_CLASSES, predicted = SELECTOR_CLASSES,
                        stringsAsFactors = FALSE)
    observed <- as.data.frame(table(factor(truth, levels = SELECTOR_CLASSES),
                                    factor(predicted, levels = SELECTOR_CLASSES)), stringsAsFactors = FALSE)
    grid$count <- observed$Freq
    confusion[[name]] <- cbind(identity[rep(1L, nrow(grid)), ], grid)
    certainty <- apply(p, 1L, max)
    index <- pmin(floor(certainty * 10), 9L) + 1L
    confidence[[name]] <- do.call(rbind, lapply(seq_along(bin_levels), function(bin) {
      keep <- index == bin
      cbind(identity, confidence_bin = bin_levels[[bin]], count = sum(keep),
            mean_confidence = if (any(keep)) mean(certainty[keep]) else NA_real_,
            accuracy = if (any(keep)) mean(predicted[keep] == truth[keep]) else NA_real_)
    }))
  }
  list(overall = do.call(rbind, overall), per_class = do.call(rbind, per_class),
       confusion = do.call(rbind, confusion), confidence = do.call(rbind, confidence))
}

selector_check_glmnet_fit <- function(fit, lambda) {
  if (is.null(fit$jerr) || length(fit$jerr) != 1L || !is.finite(fit$jerr) || fit$jerr != 0L)
    stop("glmnet failed to converge (jerr=", paste(fit$jerr, collapse = ","), ").", call. = FALSE)
  if (!length(fit$lambda) || !any(abs(fit$lambda - lambda) <= max(1, abs(lambda)) * 1e-12))
    stop("glmnet did not fit the requested lambda; interpolation is not accepted.", call. = FALSE)
  values <- unlist(lapply(coef(fit, s = lambda), function(x) as.numeric(x)), use.names = FALSE)
  if (!length(values) || any(!is.finite(values))) stop("glmnet produced non-finite coefficients.", call. = FALSE)
  invisible(TRUE)
}

selector_fit_glmnet <- function(x, y, alpha, lambda) {
  prep <- selector_preprocessor(x)
  if (!length(prep$selected)) stop("No non-constant features.", call. = FALSE)
  # glmnet requires a decreasing path for stable warm starts. Only the requested
  # endpoint is evaluated; these intermediate fits add no tuning candidates.
  if (length(lambda) != 1L || !is.finite(lambda) || lambda <= 0)
    stop("lambda must be a finite positive scalar.", call. = FALSE)
  path <- exp(seq(log(max(1, lambda * 100)), log(lambda), length.out = 30L))
  path[length(path)] <- lambda
  fit <- withCallingHandlers(
    glmnet::glmnet(selector_apply_preprocessor(x, prep), factor(y, levels = SELECTOR_CLASSES),
                  family = "multinomial", alpha = alpha, lambda = path,
                  standardize = FALSE, type.multinomial = "ungrouped"),
    warning = function(w) {
      if (grepl("Convergence for|error code", conditionMessage(w), ignore.case = TRUE))
        stop("glmnet convergence failure: ", conditionMessage(w), call. = FALSE)
    })
  selector_check_glmnet_fit(fit, lambda)
  list(kind = "glmnet", fit = fit, prep = prep, alpha = alpha, lambda = lambda)
}

selector_predict_glmnet <- function(model, x) {
  raw <- predict(model$fit, selector_apply_preprocessor(x, model$prep),
                 s = model$lambda, type = "link")
  logits <- if (length(dim(raw)) == 3L) raw[, , 1L] else raw
  logits <- logits[, SELECTOR_CLASSES, drop = FALSE]
  p <- selector_softmax(logits)
  colnames(p) <- SELECTOR_CLASSES
  selector_check_probabilities(p)
}

selector_fit_rf <- function(x, y, mtry, min_node_size, seed = 310L) {
  prep <- selector_preprocessor(x)
  z <- data.frame(selector_apply_preprocessor(x, prep), check.names = FALSE)
  z$.label <- factor(y, levels = SELECTOR_CLASSES)
  fit <- ranger::ranger(.label ~ ., data = z, probability = TRUE, num.trees = 100L,
                        mtry = min(mtry, ncol(z) - 1L), min.node.size = min_node_size,
                        seed = seed, num.threads = 4L)
  list(kind = "rf", fit = fit, prep = prep, mtry = mtry, min_node_size = min_node_size)
}

selector_predict_rf <- function(model, x) {
  p <- predict(model$fit, data.frame(selector_apply_preprocessor(x, model$prep),
                                     check.names = FALSE))$predictions
  missing <- setdiff(SELECTOR_CLASSES, colnames(p))
  if (length(missing)) p <- cbind(p, matrix(0, nrow(p), length(missing), dimnames = list(NULL, missing)))
  selector_check_probabilities(p[, SELECTOR_CLASSES, drop = FALSE])
}

selector_tokenize <- function(text) {
  text <- tolower(enc2utf8(text))
  lapply(strsplit(gsub("[^[:alnum:]_']+", " ", text), " +"), function(x) x[nzchar(x)])
}

selector_fit_tfidf <- function(text, max_terms = 2000L) {
  tokens <- selector_tokenize(text)
  df <- table(unlist(lapply(tokens, unique), use.names = FALSE))
  df <- sort(df[df >= 2L], decreasing = TRUE)
  vocabulary <- names(head(df, max_terms))
  idf <- log((length(tokens) + 1) / (as.numeric(df[vocabulary]) + 1)) + 1
  names(idf) <- vocabulary
  list(vocabulary = vocabulary, idf = idf)
}

selector_apply_tfidf <- function(text, spec) {
  tokens <- selector_tokenize(text)
  entries <- lapply(seq_along(tokens), function(i) {
    counts <- table(tokens[[i]])
    terms <- intersect(names(counts), spec$vocabulary)
    if (length(terms)) {
      return(list(i = rep.int(i, length(terms)), j = match(terms, spec$vocabulary),
                  x = as.numeric(counts[terms]) / sum(counts) * spec$idf[terms]))
    }
    list(i = integer(), j = integer(), x = numeric())
  })
  ii <- unlist(lapply(entries, `[[`, "i"), use.names = FALSE)
  jj <- unlist(lapply(entries, `[[`, "j"), use.names = FALSE)
  xx <- unlist(lapply(entries, `[[`, "x"), use.names = FALSE)
  Matrix::sparseMatrix(i = ii, j = jj, x = xx, dims = c(length(text), length(spec$vocabulary)),
                       dimnames = list(NULL, spec$vocabulary))
}

selector_cv_tune <- function(x, y, groups, kind, grid) {
  fold <- selector_group_folds(groups, 3L)
  scores <- numeric(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    prediction <- rep(NA_character_, length(y))
    for (f in 1:3) {
      tr <- fold != f; va <- !tr
      model <- if (kind == "glmnet") selector_fit_glmnet(x[tr, , drop = FALSE], y[tr], grid$alpha[g], grid$lambda[g]) else
        selector_fit_rf(x[tr, , drop = FALSE], y[tr], grid$mtry[g], grid$min_node_size[g], 310L + f)
      p <- if (kind == "glmnet") selector_predict_glmnet(model, x[va, , drop = FALSE]) else selector_predict_rf(model, x[va, , drop = FALSE])
      prediction[va] <- SELECTOR_CLASSES[max.col(p, ties.method = "first")]
    }
    scores[g] <- selector_macro_f1(y, prediction)
  }
  best <- which.max(scores)
  list(params = grid[best, , drop = FALSE], cv_macro_f1 = scores[best], scores = scores,
       folds = fold)
}

selector_cv_tune_tfidf <- function(text, y, groups, grid, max_terms = 2000L) {
  fold <- selector_group_folds(groups, 3L)
  cached <- lapply(1:3, function(f) {
    tr <- fold != f; va <- !tr
    spec <- selector_fit_tfidf(text[tr], max_terms = max_terms)
    if (!length(spec$vocabulary)) stop("A TF-IDF training fold has no eligible vocabulary.", call. = FALSE)
    list(train = tr, validation = va, x_train = selector_apply_tfidf(text[tr], spec),
         x_validation = selector_apply_tfidf(text[va], spec))
  })
  scores <- numeric(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    prediction <- rep(NA_character_, length(y))
    for (f in 1:3) {
      item <- cached[[f]]
      model <- selector_fit_glmnet(item$x_train, y[item$train], grid$alpha[g], grid$lambda[g])
      p <- selector_predict_glmnet(model, item$x_validation)
      prediction[item$validation] <- SELECTOR_CLASSES[max.col(p, ties.method = "first")]
    }
    scores[g] <- selector_macro_f1(y, prediction)
  }
  best <- which.max(scores)
  list(params = grid[best, , drop = FALSE], cv_macro_f1 = scores[best], scores = scores,
       folds = fold, fold_vocabularies = lapply(cached, function(x) colnames(x$x_train)))
}

selector_export_logistic <- function(model, feature_source, feature_names, validation_f1, path) {
  coefs <- coef(model$fit, s = model$lambda)
  intercept <- vapply(SELECTOR_CLASSES, function(cl) as.numeric(coefs[[cl]][1, 1]), numeric(1))
  beta <- t(vapply(SELECTOR_CLASSES, function(cl) as.numeric(coefs[[cl]][-1, 1]),
                   numeric(length(model$prep$selected))))
  object <- list(schema_version = "selector-logistic-v1", model_type = "multinomial_softmax",
    feature_source = feature_source, class_order = SELECTOR_CLASSES,
    input_feature_order = feature_names,
    preprocessing = list(center = unname(model$prep$center), scale = unname(model$prep$scale),
      selected_indices = as.integer(model$prep$selected - 1L),
      selected_features = feature_names[model$prep$selected]),
    model = list(lambda = model$lambda, intercept = unname(intercept), coefficients = unname(beta)),
    training = list(selection_metric = "validation_macro_f1", validation_macro_f1 = validation_f1))
  pmr_write_json(object, path)
  invisible(object)
}

selector_sha256 <- function(path) {
  pmr_require("digest")
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

selector_runtime <- function() {
  packages <- c("glmnet", "ranger", "Matrix", "jsonlite", "digest")
  pmr_require(packages)
  list(R = as.character(getRversion()), platform = R.version$platform,
       packages = setNames(lapply(packages, function(p) as.character(utils::packageVersion(p))), packages))
}

selector_check_frozen <- function(root, required_paths) {
  frozen_path <- file.path(root, "artifacts", "selector", "frozen.json")
  if (!file.exists(frozen_path)) stop("Test evaluation is locked: artifacts/selector/frozen.json is absent.", call. = FALSE)
  frozen <- jsonlite::fromJSON(frozen_path, simplifyVector = FALSE)
  hashes <- frozen$artifact_hashes
  if (is.null(hashes) || !all(required_paths %in% names(hashes)))
    stop("Frozen marker lacks required artifact hashes.", call. = FALSE)
  if (is.null(frozen$hashes) || !all(c("R/common.R", "R/selector/core.R", "R/selector/evaluate_test.R") %in% names(frozen$hashes)))
    stop("Frozen marker lacks required source hashes.", call. = FALSE)
  hashes <- c(hashes, frozen$hashes)
  for (i in seq_along(hashes)) {
    rel <- names(hashes)[[i]]
    actual <- selector_sha256(file.path(root, rel))
    if (!identical(tolower(actual), tolower(hashes[[i]]))) stop("Frozen artifact hash mismatch: ", rel, call. = FALSE)
  }
  if (!identical(frozen$r_runtime, selector_runtime())) stop("Frozen R runtime changed.", call. = FALSE)
  invisible(TRUE)
}
