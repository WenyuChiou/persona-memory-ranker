#!/usr/bin/env Rscript
script <- if (sys.nframe() > 0L) sys.frame(1)$ofile else
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(script), "common.R"))
pmr_use_library(pmr_root())

pmr_clean_queries <- function(data) {
  pmr_require("jsonlite")
  required <- c("query_id", "persona_id", "split", "query", "history_path", "gold_json",
                "updated", "who", "topic_query")
  pmr_require_columns(data, required)
  original_n <- nrow(data)
  row_number <- seq_len(original_n)
  # Content is source evidence: preserve query/gold bytes, including whitespace.
  for (name in setdiff(required, c("query", "gold_json"))) {
    data[[name]] <- trimws(gsub("\r\n?", "\n", as.character(data[[name]])))
  }
  for (name in c("query_id", "persona_id", "split")) {
    if (any(!pmr_nonempty(data[[name]]))) stop("Empty identifier or split in ", name, ".", call. = FALSE)
  }
  data$split <- tolower(data$split)
  if (any(!data$split %in% c("train", "val", "benchmark"))) stop("Unknown split; expected train, val, benchmark.", call. = FALSE)
  bool <- tolower(data$updated)
  if (any(!bool %in% c("true", "false", "1", "0"))) stop("updated must be true/false or 1/0.", call. = FALSE)
  data$updated <- bool %in% c("true", "1")
  exclusion <- rep("", original_n)
  add_reason <- function(mask, reason) {
    exclusion[mask] <<- ifelse(nzchar(exclusion[mask]), paste0(exclusion[mask], ";", reason), reason)
  }
  add_reason(!pmr_nonempty(data$query), "empty_query")
  add_reason(!pmr_nonempty(data$history_path), "empty_history_path")
  gold_valid <- vapply(data$gold_json, function(value) {
    if (!pmr_nonempty(value) || !startsWith(trimws(value), "[")) return(FALSE)
    messages <- tryCatch(jsonlite::fromJSON(value, simplifyVector = FALSE), error = function(e) NULL)
    is.list(messages) && length(messages) > 0L && all(vapply(messages, function(message) {
      is.list(message) && is.character(message$content) && length(message$content) == 1L &&
        pmr_nonempty(message$content) && is.character(message$role) && length(message$role) == 1L &&
        pmr_nonempty(message$role)
    }, logical(1)))
  }, logical(1))
  add_reason(!gold_valid, "empty_or_invalid_gold_messages")
  # Chat wrappers would let role metadata become part of the actual search query.
  role_wrapped <- grepl("(?i)^\\s*(system|assistant|user|human)\\s*:", data$query, perl = TRUE) |
    grepl("(?i)(<\\|(?:system|user|assistant|im_start)\\|>|\\[/?INST\\])", data$query, perl = TRUE) |
    grepl('^\\s*[\\[{].*"role"\\s*:', data$query, perl = TRUE)
  if (any(role_wrapped & nzchar(data$query))) stop("Query contains a chat role wrapper at row(s): ",
    paste(row_number[role_wrapped], collapse = ", "), call. = FALSE)
  # Enforce split isolation even for rows excluded from later modeling.
  personas <- split(data$split, data$persona_id)
  if (any(vapply(personas, function(s) length(unique(s)) > 1L, logical(1)))) {
    stop("Cross-split persona leakage detected.", call. = FALSE)
  }
  duplicate <- duplicated(data)
  add_reason(duplicate, "exact_duplicate_within_split")
  keep <- !nzchar(exclusion)
  clean <- data[keep, , drop = FALSE]
  if (anyDuplicated(clean[c("persona_id", "query")])) {
    stop("Duplicate question for the same persona remains after exact deduplication.", call. = FALSE)
  }
  if (anyDuplicated(clean$query_id)) stop("query_id must be unique after exact deduplication.", call. = FALSE)
  if (!nrow(clean)) stop("No usable queries remain after cleaning.", call. = FALSE)
  rejected <- data.frame(input_row = row_number[!keep], query_id = data$query_id[!keep],
                         reason = exclusion[!keep], stringsAsFactors = FALSE)
  reasons <- unlist(strsplit(exclusion[nzchar(exclusion)], ";", fixed = TRUE))
  reason_counts <- if (length(reasons)) as.list(table(reasons)) else list()
  list(data = clean, report = list(schema_version = 1L, input_rows = original_n,
    output_rows = nrow(clean), excluded_rows = sum(!keep), reason_counts = reason_counts,
    split_counts = as.list(table(clean$split)), exclusions = rejected,
    checks = list(persona_split_isolation = TRUE, unique_persona_questions = TRUE,
                  unique_query_ids = TRUE, role_free_queries = TRUE),
    note = "Gold messages are incomplete synthetic annotations; cleaning does not turn unlabeled memories into verified negatives."))
}

if (sys.nframe() == 0L) pmr_main(function() {
  args <- pmr_args(c("input", "output", "report"))
  result <- pmr_clean_queries(pmr_read_csv(args$input))
  pmr_write_csv(result$data, args$output)
  pmr_write_json(result$report, args$report)
  cat("Cleaned", result$report$input_rows, "rows to", result$report$output_rows,
      "; excluded", result$report$excluded_rows, "\n")
})
