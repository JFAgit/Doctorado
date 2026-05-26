base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos"

sex_genes <- c(
  "XIST", "TSIX",
  "DDX3Y", "EIF1AY", "KDM5D", "RPS4Y1", "RPS4Y2", "TMSB4Y",
  "USP9Y", "UTY", "ZFY", "PRKY", "NLGN4Y", "TXLNGY"
)

sex_entrez <- c(
  XIST = "7503",
  DDX3Y = "8653",
  EIF1AY = "9086",
  KDM5D = "8284",
  RPS4Y1 = "6192",
  RPS4Y2 = "140032",
  TMSB4Y = "9087",
  USP9Y = "8287",
  UTY = "7404",
  ZFY = "7544",
  PRKY = "5616",
  NLGN4Y = "22829",
  TXLNGY = "246126"
)

make_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

read_series_matrix <- function(path) {
  lines <- readLines(gzfile(path, "rt"), warn = FALSE)
  start <- grep("!series_matrix_table_begin", lines, fixed = TRUE)
  end <- grep("!series_matrix_table_end", lines, fixed = TRUE)
  if (length(start) != 1 || length(end) != 1) {
    stop("Could not find series matrix table boundaries")
  }
  expr <- read.delim(
    text = paste(lines[(start + 1):(end - 1)], collapse = "\n"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  rownames(expr) <- expr[[1]]
  expr[[1]] <- NULL

  meta_lines <- lines[seq_len(start - 1)]
  get_meta <- function(key) {
    row <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    if (length(row) == 0) return(NULL)
    vals <- strsplit(row[1], "\t", fixed = TRUE)[[1]][-1]
    gsub('^"|"$', "", vals)
  }
  meta <- data.frame(
    geo_accession = get_meta("!Sample_geo_accession"),
    sample_title = get_meta("!Sample_title"),
    cell_type_raw = get_meta("!Sample_characteristics_ch1"),
    stringsAsFactors = FALSE
  )
  meta$cell_type <- sub("^cell type: ", "", meta$cell_type_raw)
  meta$donor_id <- sub(".*_([0-9]+)$", "\\1", meta$sample_title)
  list(expr = expr, meta = meta)
}

read_bgx_probes <- function(path) {
  lines <- readLines(gzfile(path, "rt"), warn = FALSE)
  start <- grep("^\\[Probes\\]$", lines)
  if (length(start) != 1) stop("Could not find [Probes] section")
  probe_lines <- lines[(start + 1):length(lines)]
  read.delim(
    text = paste(probe_lines, collapse = "\n"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

infer_from_marker_matrix <- function(marker_expr, marker_gene_col = "gene_symbol") {
  y_markers <- setdiff(unique(marker_expr[[marker_gene_col]]), c("XIST", "TSIX"))
  sample_cols <- setdiff(colnames(marker_expr), c("probe_id", "gene_symbol", "entrez_id", "chromosome"))
  gene_expr <- do.call(rbind, lapply(unique(marker_expr[[marker_gene_col]]), function(g) {
    rows <- marker_expr[[marker_gene_col]] == g
    vals <- apply(marker_expr[rows, sample_cols, drop = FALSE], 2, function(x) median(as.numeric(x), na.rm = TRUE))
    data.frame(gene_symbol = g, t(vals), check.names = FALSE)
  }))
  rownames(gene_expr) <- gene_expr$gene_symbol
  y_gene_expr <- gene_expr[gene_expr$gene_symbol %in% y_markers, sample_cols, drop = FALSE]
  y_dynamic <- apply(y_gene_expr, 1, function(x) diff(range(as.numeric(x), na.rm = TRUE)))
  y_markers_for_score <- names(sort(y_dynamic, decreasing = TRUE))[seq_len(min(5, length(y_dynamic)))]

  out <- lapply(sample_cols, function(s) {
    xist_vals <- gene_expr[gene_expr$gene_symbol == "XIST", s]
    y_vals <- gene_expr[gene_expr$gene_symbol %in% y_markers_for_score, s]
    xist_score <- if (length(xist_vals)) median(as.numeric(xist_vals), na.rm = TRUE) else NA_real_
    y_score <- if (length(y_vals)) median(as.numeric(y_vals), na.rm = TRUE) else NA_real_
    data.frame(sample = s, xist_score = xist_score, y_marker_score = y_score)
  })
  out <- do.call(rbind, out)
  if (all(is.na(out$y_marker_score))) {
    out$sex_inferred <- NA_character_
  } else {
    y_cut <- mean(range(out$y_marker_score, na.rm = TRUE))
    out$sex_inferred <- ifelse(out$y_marker_score > y_cut, "male", "female")
  }
  out
}

process_gse35449 <- function() {
  ds_dir <- file.path(base_dir, "GSE35449")
  sm <- read_series_matrix(file.path(ds_dir, "metadata", "GSE35449_series_matrix.txt.gz"))
  bgx <- read_bgx_probes(file.path(ds_dir, "metadata", "GPL6947_HumanHT-12_V3_0_R1_11283641_A.bgx.gz"))

  ann <- bgx[, c("Probe_Id", "Symbol", "Entrez_Gene_ID", "Chromosome")]
  colnames(ann) <- c("probe_id", "gene_symbol", "entrez_id", "chromosome")
  ann$gene_symbol <- toupper(ann$gene_symbol)
  marker_ann <- ann[ann$gene_symbol %in% sex_genes | ann$chromosome == "Y", ]
  marker_ann <- marker_ann[marker_ann$probe_id %in% rownames(sm$expr), ]
  marker_expr <- cbind(marker_ann, sm$expr[marker_ann$probe_id, , drop = FALSE])

  sex_by_sample <- infer_from_marker_matrix(marker_expr, "gene_symbol")
  names(sex_by_sample)[names(sex_by_sample) == "sample"] <- "geo_accession"
  meta <- merge(sm$meta, sex_by_sample, by = "geo_accession", all.x = TRUE)
  donor_sex <- aggregate(
    cbind(y_marker_score, xist_score) ~ donor_id,
    data = meta,
    FUN = median,
    na.rm = TRUE
  )
  y_cut <- mean(range(donor_sex$y_marker_score, na.rm = TRUE))
  donor_sex$sex_inferred_donor <- ifelse(donor_sex$y_marker_score > y_cut, "male", "female")
  meta <- merge(meta, donor_sex[, c("donor_id", "sex_inferred_donor")], by = "donor_id", all.x = TRUE)

  write.csv(marker_expr, file.path(ds_dir, "metadata", "GSE35449_sex_marker_probe_expression.csv"), row.names = FALSE)
  write.csv(meta, file.path(ds_dir, "metadata", "GSE35449_sample_metadata_sex_inferred.csv"), row.names = FALSE)
  write.csv(donor_sex, file.path(ds_dir, "metadata", "GSE35449_donor_sex_inferred.csv"), row.names = FALSE)
}

process_emtab7572 <- function() {
  ds_dir <- file.path(base_dir, "E-MTAB-7572")
  counts <- read.csv(file.path(ds_dir, "processed", "entrez_counts.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  names(counts)[1] <- "entrez_id"
  counts$entrez_id <- as.character(counts$entrez_id)

  sdrf <- read.delim(file.path(ds_dir, "metadata", "E-MTAB-7572.sdrf.txt"), check.names = FALSE, stringsAsFactors = FALSE)
  meta <- data.frame(
    source_name = sdrf[["Source Name"]],
    sex_metadata = sdrf[["Characteristics[sex]"]],
    disease = sdrf[["Characteristics[disease]"]],
    individual = sdrf[["Characteristics[individual]"]],
    cell_type = sdrf[["Characteristics[cell type]"]],
    stimulus = sdrf[["Characteristics[stimulus]"]],
    assay_name = sdrf[["Assay Name"]],
    stringsAsFactors = FALSE
  )
  meta$count_col <- gsub("-", ".", meta$assay_name)

  marker_counts <- counts[counts$entrez_id %in% sex_entrez, , drop = FALSE]
  marker_counts$gene_symbol <- names(sex_entrez)[match(marker_counts$entrez_id, sex_entrez)]
  sample_cols <- intersect(meta$count_col, colnames(marker_counts))
  lib_size <- colSums(counts[, sample_cols, drop = FALSE])
  logcpm <- marker_counts
  logcpm[, sample_cols] <- sweep(log2(marker_counts[, sample_cols, drop = FALSE] + 0.5), 2, log2(lib_size / 1e6), "-")
  marker_expr <- logcpm[, c("entrez_id", "gene_symbol", sample_cols)]

  tmp <- marker_expr
  tmp$probe_id <- tmp$entrez_id
  tmp$chromosome <- ifelse(tmp$gene_symbol == "XIST", "X", "Y")
  sex_by_sample <- infer_from_marker_matrix(tmp[, c("probe_id", "gene_symbol", "entrez_id", "chromosome", sample_cols)], "gene_symbol")
  names(sex_by_sample)[names(sex_by_sample) == "sample"] <- "count_col"
  meta <- merge(meta, sex_by_sample, by = "count_col", all.x = TRUE)
  meta$sex_match_metadata <- meta$sex_metadata == meta$sex_inferred

  write.csv(marker_expr, file.path(ds_dir, "metadata", "E-MTAB-7572_sex_marker_logCPM.csv"), row.names = FALSE)
  write.csv(meta, file.path(ds_dir, "metadata", "E-MTAB-7572_sample_metadata_sex_checked.csv"), row.names = FALSE)
}

process_gse18686 <- function() {
  ds_dir <- file.path(base_dir, "GSE18686")
  sm_path <- file.path(ds_dir, "metadata", "GSE18686_series_matrix.txt.gz")
  lines <- readLines(gzfile(sm_path, "rt"), warn = FALSE)
  start <- grep("!series_matrix_table_begin", lines, fixed = TRUE)
  end <- grep("!series_matrix_table_end", lines, fixed = TRUE)
  expr <- read.delim(
    text = paste(lines[(start + 1):(end - 1)], collapse = "\n"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  rownames(expr) <- expr[[1]]
  expr[[1]] <- NULL

  meta_lines <- lines[seq_len(start - 1)]
  get_meta <- function(key) {
    row <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    if (length(row) == 0) return(NULL)
    vals <- strsplit(row[1], "\t", fixed = TRUE)[[1]][-1]
    gsub('^"|"$', "", vals)
  }
  get_all_meta <- function(key) {
    rows <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    lapply(rows, function(row) {
      vals <- strsplit(row, "\t", fixed = TRUE)[[1]][-1]
      gsub('^"|"$', "", vals)
    })
  }
  titles <- get_meta("!Sample_title")
  geo <- get_meta("!Sample_geo_accession")
  source <- get_meta("!Sample_source_name_ch1")
  chars <- get_all_meta("!Sample_characteristics_ch1")
  char_df <- data.frame(
    id_raw = chars[[1]],
    treatment_raw = chars[[2]],
    cell_type_raw = chars[[3]],
    batch_raw = chars[[4]],
    line_raw = chars[[5]],
    stringsAsFactors = FALSE
  )
  meta <- data.frame(
    geo_accession = geo,
    sample_title = titles,
    source_name = source,
    char_df,
    stringsAsFactors = FALSE
  )
  meta$donor_id <- sub("^ID: ", "", meta$id_raw)
  meta$treatment <- sub("^treatment group: ", "", meta$treatment_raw)
  meta$cell_type <- sub("^cell type: ", "", meta$cell_type_raw)
  meta$is_macrophage_culture <- meta$source_name == "Macrophages culture" & meta$cell_type == "cultured macrophages"

  bgx <- read_bgx_probes(file.path(ds_dir, "metadata", "GPL6947_HumanHT-12_V3_0_R1_11283641_A.bgx.gz"))
  ann <- bgx[, c("Probe_Id", "Symbol", "Entrez_Gene_ID", "Chromosome")]
  colnames(ann) <- c("probe_id", "gene_symbol", "entrez_id", "chromosome")
  ann$gene_symbol <- toupper(ann$gene_symbol)
  marker_ann <- ann[ann$gene_symbol %in% sex_genes | ann$chromosome == "Y", ]
  marker_ann <- marker_ann[marker_ann$probe_id %in% rownames(expr), ]
  macrophage_cols <- meta$geo_accession[meta$is_macrophage_culture]
  marker_expr <- cbind(marker_ann, expr[marker_ann$probe_id, macrophage_cols, drop = FALSE])

  sex_by_sample <- infer_from_marker_matrix(marker_expr, "gene_symbol")
  names(sex_by_sample)[names(sex_by_sample) == "sample"] <- "geo_accession"
  meta <- merge(meta, sex_by_sample, by = "geo_accession", all.x = TRUE)
  donor_sex <- aggregate(
    cbind(y_marker_score, xist_score) ~ donor_id,
    data = meta[meta$is_macrophage_culture, ],
    FUN = median,
    na.rm = TRUE
  )
  y_cut <- mean(range(donor_sex$y_marker_score, na.rm = TRUE))
  donor_sex$sex_inferred_donor <- ifelse(donor_sex$y_marker_score > y_cut, "male", "female")
  meta <- merge(meta, donor_sex[, c("donor_id", "sex_inferred_donor")], by = "donor_id", all.x = TRUE)

  write.csv(marker_expr, file.path(ds_dir, "metadata", "GSE18686_sex_marker_probe_expression.csv"), row.names = FALSE)
  write.csv(meta, file.path(ds_dir, "metadata", "GSE18686_sample_metadata_sex_inferred.csv"), row.names = FALSE)
  write.csv(donor_sex, file.path(ds_dir, "metadata", "GSE18686_donor_sex_inferred.csv"), row.names = FALSE)
}

process_gse35449()
process_emtab7572()
if (dir.exists(file.path(base_dir, "GSE18686"))) process_gse18686()
