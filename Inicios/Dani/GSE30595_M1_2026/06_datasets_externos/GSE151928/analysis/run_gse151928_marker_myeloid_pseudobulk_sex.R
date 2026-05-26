project_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026"
dataset_dir <- file.path(project_dir, "06_datasets_externos", "GSE151928")
processed_dir <- file.path(dataset_dir, "processed")
out_dir <- file.path(dataset_dir, "analysis")
local_lib <- file.path(out_dir, "R_libs")
if (dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))
if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("Package data.table is required. Install it in analysis/R_libs or the active R library.")
}

sample_meta <- read.delim(
  file.path(dataset_dir, "metadata", "GSE151928_sample_summary.tsv"),
  stringsAsFactors = FALSE
)
sample_meta$subject_number <- as.integer(sub("Subject ", "", sample_meta$subject))
sample_meta$sex <- tolower(sample_meta$sex)

count_files <- list.files(processed_dir, pattern = "UMI_counts\\.csv\\.gz$", full.names = TRUE)
file_subject <- ifelse(
  grepl("_sample_[0-9]+_", basename(count_files)),
  sub(".*_sample_([0-9]+)_.*", "\\1", basename(count_files)),
  sub(".*_subject_([0-9]+)_.*", "\\1", basename(count_files))
)
file_map <- data.frame(
  file = count_files,
  subject_number = as.integer(file_subject),
  stringsAsFactors = FALSE
)
file_map <- merge(file_map, sample_meta, by = "subject_number", all.x = TRUE)
file_map <- file_map[order(file_map$subject_number), ]

marker_sets <- list(
  alveolar_macrophage = c("MARCO", "FABP4", "PPARG", "MRC1", "MSR1", "CD68", "CD163", "C1QA", "C1QB", "C1QC", "APOC1", "APOE", "LIPA", "MERTK"),
  monocyte_like = c("LYZ", "LST1", "FCN1", "VCAN", "S100A8", "S100A9", "CTSS", "TYROBP", "FCER1G", "AIF1", "LGALS3"),
  other_lineage = c("CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "MS4A1", "CD79A", "FCER1A", "CLEC10A", "EPCAM", "KRT8", "KRT18", "PECAM1", "VWF", "COL1A1")
)

score_cells <- function(x) {
  genes <- toupper(as.character(x[[1]]))
  count_only <- x[, -1, drop = FALSE]
  cell_umi <- colSums(count_only)
  score_set <- function(gset) {
    rows <- genes %in% toupper(gset)
    if (!any(rows)) return(rep(0, ncol(count_only)))
    raw <- colSums(count_only[rows, , drop = FALSE])
    log1p((raw / pmax(cell_umi, 1)) * 10000)
  }
  am <- score_set(marker_sets$alveolar_macrophage)
  mono <- score_set(marker_sets$monocyte_like)
  other <- score_set(marker_sets$other_lineage)
  myeloid <- am + mono
  myeloid_keep <- myeloid >= 1 & myeloid > other
  subtype <- rep("other_or_uncertain", length(myeloid_keep))
  subtype[myeloid_keep & am >= mono] <- "alveolar_macrophage_like"
  subtype[myeloid_keep & mono > am] <- "monocyte_like"
  data.frame(
    cell_barcode = colnames(count_only),
    cell_umi = cell_umi,
    alveolar_macrophage_score = am,
    monocyte_like_score = mono,
    other_lineage_score = other,
    marker_myeloid_keep = myeloid_keep,
    marker_subtype = subtype,
    stringsAsFactors = FALSE
  )
}

make_pseudobulk <- function(x, keep) {
  genes <- make.unique(as.character(x[[1]]))
  count_only <- x[, -1, drop = FALSE]
  if (sum(keep) == 0) {
    counts <- rep(0, length(genes))
  } else {
    counts <- rowSums(count_only[, keep, drop = FALSE])
  }
  names(counts) <- genes
  counts
}

pb_myeloid <- list()
pb_am <- list()
pb_mono <- list()
cell_qc <- list()

for (i in seq_len(nrow(file_map))) {
  message("Reading and scoring ", basename(file_map$file[i]))
  x <- data.table::fread(file_map$file[i], data.table = FALSE, check.names = FALSE, showProgress = FALSE)
  scores <- score_cells(x)
  scores$subject <- file_map$subject[i]
  scores$sex <- file_map$sex[i]
  scores$age <- file_map$age[i]
  cell_qc[[i]] <- scores[, c("subject", "sex", "age", "cell_barcode", "cell_umi",
                             "alveolar_macrophage_score", "monocyte_like_score",
                             "other_lineage_score", "marker_myeloid_keep", "marker_subtype")]
  pb_myeloid[[file_map$subject[i]]] <- make_pseudobulk(x, scores$marker_myeloid_keep)
  pb_am[[file_map$subject[i]]] <- make_pseudobulk(x, scores$marker_subtype == "alveolar_macrophage_like")
  pb_mono[[file_map$subject[i]]] <- make_pseudobulk(x, scores$marker_subtype == "monocyte_like")
  rm(x)
  gc()
}

cell_qc <- do.call(rbind, cell_qc)
data.table::fwrite(cell_qc, file.path(out_dir, "GSE151928_marker_based_cell_scores.tsv"), sep = "\t")

make_matrix <- function(pb_list) {
  all_genes <- Reduce(union, lapply(pb_list, names))
  mat <- matrix(0, nrow = length(all_genes), ncol = length(pb_list))
  rownames(mat) <- all_genes
  colnames(mat) <- names(pb_list)
  for (nm in names(pb_list)) mat[names(pb_list[[nm]]), nm] <- pb_list[[nm]]
  mat
}

analyze_pb <- function(count_mat, label) {
  lib_size <- colSums(count_mat)
  cpm <- sweep(count_mat, 2, pmax(lib_size, 1) / 1e6, "/")
  logcpm <- log2(cpm + 0.5)
  keep <- rowSums(cpm >= 1) >= 3
  sex <- sample_meta$sex[match(colnames(count_mat), sample_meta$subject)]
  male_cols <- which(sex == "male")
  female_cols <- which(sex == "female")
  test_one <- function(x) {
    if (sd(x[male_cols]) == 0 && sd(x[female_cols]) == 0) return(NA_real_)
    tryCatch(t.test(x[male_cols], x[female_cols])$p.value, error = function(e) NA_real_)
  }
  stats <- data.frame(
    gene_symbol = rownames(logcpm),
    mean_logCPM_male = rowMeans(logcpm[, male_cols, drop = FALSE]),
    mean_logCPM_female = rowMeans(logcpm[, female_cols, drop = FALSE]),
    logFC_male_vs_female = rowMeans(logcpm[, male_cols, drop = FALSE]) -
      rowMeans(logcpm[, female_cols, drop = FALSE]),
    mean_CPM = rowMeans(cpm),
    keep_for_test = keep,
    stringsAsFactors = FALSE
  )
  stats$pvalue <- NA_real_
  stats$pvalue[keep] <- apply(logcpm[keep, , drop = FALSE], 1, test_one)
  stats$padj <- NA_real_
  stats$padj[keep] <- p.adjust(stats$pvalue[keep], method = "BH")
  stats <- stats[order(stats$pvalue, -abs(stats$logFC_male_vs_female), na.last = TRUE), ]
  write.csv(count_mat, file.path(out_dir, paste0("GSE151928_", label, "_pseudobulk_counts_by_subject.csv")))
  write.csv(logcpm, file.path(out_dir, paste0("GSE151928_", label, "_pseudobulk_logCPM_by_subject.csv")))
  write.csv(stats, file.path(out_dir, paste0("GSE151928_", label, "_sex_DE_welch_logCPM.csv")), row.names = FALSE)
  stats
}

stats_by_label <- list(
  marker_myeloid = analyze_pb(make_matrix(pb_myeloid), "marker_myeloid"),
  marker_alveolar_macrophage = analyze_pb(make_matrix(pb_am), "marker_alveolar_macrophage"),
  marker_monocyte_like = analyze_pb(make_matrix(pb_mono), "marker_monocyte_like")
)

read_gene_set <- function(path, col = "gene_symbol") {
  x <- read.csv(path, stringsAsFactors = FALSE)
  unique(toupper(x[[col]]))
}
gene_sets <- list(
  inmunometabolismo_expandida = readLines(file.path(project_dir, "02_tablas_y_listas", "genes_inmunometabolismo_expandida_symbols.txt"), warn = FALSE),
  antiinflammatory_general = read_gene_set(file.path(project_dir, "02_tablas_y_listas", "antiinflammatory_general_curated_gene_list.csv")),
  proinflammatory_general = read_gene_set(file.path(project_dir, "02_tablas_y_listas", "proinflammatory_general_curated_gene_list.csv"))
)

summary_rows <- list()
for (label in names(stats_by_label)) {
  stats <- stats_by_label[[label]]
  for (set_name in names(gene_sets)) {
    overlap <- stats[toupper(stats$gene_symbol) %in% toupper(gene_sets[[set_name]]), ]
    write.csv(overlap, file.path(out_dir, paste0("GSE151928_", label, "_", set_name, "_sex_DE_overlap.csv")), row.names = FALSE)
    detected <- overlap[overlap$keep_for_test %in% TRUE, ]
    nominal <- detected[!is.na(detected$pvalue) & detected$pvalue < 0.05, ]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      population = label,
      gene_set = set_name,
      n_genes_in_set = length(unique(toupper(gene_sets[[set_name]]))),
      n_detected_in_dataset = nrow(detected),
      n_nominal_p_lt_0_05 = nrow(nominal),
      nominal_pct_of_detected = ifelse(nrow(detected) > 0, 100 * nrow(nominal) / nrow(detected), NA_real_),
      n_padj_lt_0_10 = sum(detected$padj < 0.10, na.rm = TRUE),
      n_up_male_nominal = sum(nominal$logFC_male_vs_female > 0, na.rm = TRUE),
      n_up_female_nominal = sum(nominal$logFC_male_vs_female < 0, na.rm = TRUE),
      n_up_male_any_logFC = sum(detected$logFC_male_vs_female > 0, na.rm = TRUE),
      n_up_female_any_logFC = sum(detected$logFC_male_vs_female < 0, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
}
summary_rows <- do.call(rbind, summary_rows)
write.csv(summary_rows, file.path(out_dir, "GSE151928_marker_based_gene_set_overlap_summary.csv"), row.names = FALSE)

cell_summary <- aggregate(
  marker_myeloid_keep ~ subject + sex + age,
  data = cell_qc,
  FUN = function(x) c(total_cells = length(x), marker_myeloid_cells = sum(x))
)
cell_summary <- do.call(data.frame, cell_summary)
names(cell_summary)[names(cell_summary) == "marker_myeloid_keep.total_cells"] <- "total_cells"
names(cell_summary)[names(cell_summary) == "marker_myeloid_keep.marker_myeloid_cells"] <- "marker_myeloid_cells"
subtype_counts <- as.data.frame.matrix(table(cell_qc$subject, cell_qc$marker_subtype))
subtype_counts$subject <- rownames(subtype_counts)
cell_summary <- merge(cell_summary, subtype_counts, by = "subject", all.x = TRUE)
cell_summary$marker_myeloid_pct <- 100 * cell_summary$marker_myeloid_cells / cell_summary$total_cells
write.csv(cell_summary, file.path(out_dir, "GSE151928_marker_based_cell_population_summary_by_subject.csv"), row.names = FALSE)

message("Done marker-based pseudobulk.")
