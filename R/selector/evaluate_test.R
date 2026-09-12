#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
root_default <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
root <- if (length(commandArgs(TRUE))) normalizePath(commandArgs(TRUE)[[1]], winslash = "/", mustWork = TRUE) else root_default
source(file.path(root, "R", "common.R")); source(file.path(root, "R", "selector", "core.R")); pmr_use_library(root)
main <- function() {
  pmr_require(c("glmnet", "ranger", "jsonlite", "digest"))
  required <- c("data/processed/selector/rows.csv", "artifacts/selector/features_context.csv",
                "artifacts/selector/features_response.csv", "artifacts/selector/features_situation.csv",
                "artifacts/selector/model_bundle.rds")
  selector_check_frozen(root, required)
  rows <- pmr_read_csv(file.path(root, required[[1]])); selector_validate_rows(rows)
  context <- selector_read_embeddings(file.path(root, required[[2]]), rows)
  response <- selector_read_embeddings(file.path(root, required[[3]]), rows)
  bundle <- readRDS(file.path(root, required[[5]]))
  if (is.null(bundle$selected_name) || is.null(bundle$deployed_name))
    stop("Model bundle lacks explicit selected/deployed identities.", call. = FALSE)
  test <- rows$split == "test"; if (!any(test)) stop("No test rows.", call. = FALSE)
  sources <- list(context = context, response = response)
  probabilities <- lapply(bundle$candidates, function(candidate) {
    x <- if (candidate$source == "model_text_tfidf")
      selector_apply_tfidf(rows$model_text[test], candidate$tfidf) else
      sources[[candidate$source]][test, , drop = FALSE]
    if (candidate$model$kind == "glmnet") selector_predict_glmnet(candidate$model, x) else selector_predict_rf(candidate$model, x)
  })
  tables <- selector_evaluation_tables(probabilities, rows$label[test], bundle$selected_name, bundle$deployed_name)
  probability_rows <- do.call(rbind, lapply(names(probabilities), function(name)
    data.frame(candidate = name, selected = name == bundle$selected_name, deployed = name == bundle$deployed_name,
               row_id = rows$row_id[test], probabilities[[name]], check.names = FALSE)))
  out <- file.path(root, "artifacts", "selector")
  pmr_write_csv(probability_rows, file.path(out, "test_probabilities.csv"))
  pmr_write_csv(tables$overall, file.path(out, "test_metrics.csv"))
  pmr_write_csv(tables$per_class, file.path(out, "test_per_class.csv"))
  pmr_write_csv(tables$confusion, file.path(out, "test_confusion_matrices.csv"))
  pmr_write_csv(tables$confidence, file.path(out, "test_confidence_bins.csv"))
  pmr_write_json(list(selected_model = bundle$selected_name, deployed_model = bundle$deployed_name,
    n_test = sum(test), candidates = names(probabilities)), file.path(out, "test_evaluation.json"))
}
pmr_main(main)
