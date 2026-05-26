base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/_busqueda_candidatos"

sex_ensembl <- c(
  XIST = "ENSG00000229807",
  DDX3Y = "ENSG00000067048",
  EIF1AY = "ENSG00000198692",
  KDM5D = "ENSG00000012817",
  RPS4Y1 = "ENSG00000129824",
  RPS4Y2 = "ENSG00000157828",
  TMSB4Y = "ENSG00000154620",
  USP9Y = "ENSG00000114374",
  UTY = "ENSG00000183878",
  ZFY = "ENSG00000067646",
  PRKY = "ENSG00000099725",
  NLGN4Y = "ENSG00000165246",
  TXLNGY = "ENSG00000131002"
)

read_geo_meta <- function(path) {
  lines <- readLines(gzfile(path, "rt"), warn = FALSE)
  meta_lines <- lines[seq_len(grep("!series_matrix_table_begin", lines, fixed = TRUE)[1] - 1)]
  get_rows <- function(key) {
    rows <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    lapply(rows, function(row) {
      vals <- strsplit(row, "\t", fixed = TRUE)[[1]][-1]
      gsub('^"|"$', "", vals)
    })
  }
  list(
    title = get_rows("!Sample_title")[[1]],
    geo = get_rows("!Sample_geo_accession")[[1]],
    characteristics = get_rows("!Sample_characteristics_ch1"),
    description = get_rows("!Sample_description")
  )
}

infer_sex <- function(expr, dataset) {
  names(expr)[1] <- "ensembl_id"
  expr$ensembl_id <- sub("\\..*$", "", expr$ensembl_id)
  marker <- expr[expr$ensembl_id %in% unname(sex_ensembl), , drop = FALSE]
  marker$gene_symbol <- names(sex_ensembl)[match(marker$ensembl_id, sex_ensembl)]
  marker <- marker[, c("ensembl_id", "gene_symbol", setdiff(names(marker), c("ensembl_id", "gene_symbol")))]

  sample_cols <- setdiff(names(marker), c("ensembl_id", "gene_symbol"))
  y_genes <- setdiff(unique(marker$gene_symbol), "XIST")
  y_expr <- marker[marker$gene_symbol %in% y_genes, sample_cols, drop = FALSE]
  y_dynamic <- apply(y_expr, 1, function(x) diff(range(as.numeric(x), na.rm = TRUE)))
  y_keep <- marker$gene_symbol[marker$gene_symbol %in% y_genes][order(y_dynamic, decreasing = TRUE)]
  y_keep <- unique(y_keep)[seq_len(min(5, length(unique(y_keep))))]

  calls <- do.call(rbind, lapply(sample_cols, function(s) {
    xist <- marker[marker$gene_symbol == "XIST", s, drop = TRUE]
    yvals <- unlist(marker[marker$gene_symbol %in% y_keep, s, drop = TRUE])
    data.frame(
      dataset = dataset,
      sample_col = s,
      xist_score = if (length(xist)) as.numeric(xist[1]) else NA_real_,
      y_marker_score = median(as.numeric(yvals), na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
  y_cut <- mean(range(calls$y_marker_score, na.rm = TRUE))
  calls$sex_inferred <- ifelse(calls$y_marker_score > y_cut, "male", "female")
  attr(calls, "marker_expr") <- marker
  attr(calls, "y_markers_used") <- y_keep
  calls
}

process_gse228087 <- function() {
  meta <- read_geo_meta(file.path(base_dir, "GSE228087_series_matrix.txt.gz"))
  expr <- read.csv(gzfile(file.path(base_dir, "GSE228087_MoMF_tpm.csv.gz")), check.names = FALSE)
  calls <- infer_sex(expr, "GSE228087")
  desc <- meta$description[[1]]
  sample_meta <- data.frame(
    sample_col = desc,
    geo_accession = meta$geo,
    sample_title = meta$title,
    donor_from_title = sub("^([0-9]+)_.*$", "\\1", meta$title),
    donor_id_geo_field = sub("^donor id: ", "", meta$characteristics[[4]]),
    treatment = sub("^treatment: ", "", meta$characteristics[[3]]),
    stringsAsFactors = FALSE
  )
  calls <- merge(sample_meta, calls, by = "sample_col", all.x = TRUE)
  write.csv(calls, file.path(base_dir, "GSE228087_sample_metadata_sex_inferred.csv"), row.names = FALSE)
  write.csv(attr(infer_sex(expr, "GSE228087"), "marker_expr"), file.path(base_dir, "GSE228087_sex_marker_tpm.csv"), row.names = FALSE)
}

process_gse174689 <- function() {
  meta <- read_geo_meta(file.path(base_dir, "GSE174689_series_matrix.txt.gz"))
  expr <- read.delim(gzfile(file.path(base_dir, "GSE174689_All_logTPMs_exprTable.txt.gz")), check.names = FALSE)
  calls <- infer_sex(expr, "GSE174689")
  desc <- meta$description[[2]]
  sample_meta <- data.frame(
    sample_col = desc,
    geo_accession = meta$geo,
    sample_title = meta$title,
    time_point = sub("^time point: ", "", meta$characteristics[[1]]),
    treatment = sub("^treatment: ", "", meta$characteristics[[2]]),
    infection = sub("^infection: ", "", meta$characteristics[[3]]),
    stringsAsFactors = FALSE
  )
  calls <- merge(sample_meta, calls, by = "sample_col", all.x = TRUE)
  write.csv(calls, file.path(base_dir, "GSE174689_sample_metadata_sex_inferred.csv"), row.names = FALSE)
  write.csv(attr(infer_sex(expr, "GSE174689"), "marker_expr"), file.path(base_dir, "GSE174689_sex_marker_logTPM.csv"), row.names = FALSE)
}

process_gse228087()
process_gse174689()
