project_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026"
dataset_dir <- file.path(project_dir, "06_datasets_externos", "GSE151928")
processed_dir <- file.path(dataset_dir, "processed")
out_dir <- file.path(dataset_dir, "analysis")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
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

read_pseudobulk <- function(path) {
  message("Reading ", basename(path))
  x <- data.table::fread(path, data.table = FALSE, check.names = FALSE, showProgress = FALSE)
  genes <- make.unique(as.character(x[[1]]))
  counts <- rowSums(x[, -1, drop = FALSE])
  names(counts) <- genes
  data.frame(
    gene_symbol = genes,
    pseudobulk_count = as.numeric(counts),
    n_cells = ncol(x) - 1,
    stringsAsFactors = FALSE
  )
}

pseudobulk_n_cells <- integer(nrow(file_map))
pseudobulk_list <- lapply(seq_len(nrow(file_map)), function(i) {
  pb <- read_pseudobulk(file_map$file[i])
  pseudobulk_n_cells[i] <<- pb$n_cells[1]
  colnames(pb)[2] <- file_map$subject[i]
  pb[, c("gene_symbol", file_map$subject[i])]
})

all_genes <- Reduce(union, lapply(pseudobulk_list, `[[`, "gene_symbol"))
count_mat <- matrix(0, nrow = length(all_genes), ncol = length(pseudobulk_list))
rownames(count_mat) <- all_genes
colnames(count_mat) <- file_map$subject
for (i in seq_along(pseudobulk_list)) {
  v <- pseudobulk_list[[i]][[2]]
  names(v) <- pseudobulk_list[[i]]$gene_symbol
  count_mat[names(v), i] <- v
}

lib_size <- colSums(count_mat)
cpm <- sweep(count_mat, 2, lib_size / 1e6, "/")
logcpm <- log2(cpm + 0.5)
keep <- rowSums(cpm >= 1) >= 3

sex <- sample_meta$sex[match(colnames(count_mat), sample_meta$subject)]
male_cols <- which(sex == "male")
female_cols <- which(sex == "female")

test_one <- function(x) {
  if (sd(x[male_cols]) == 0 && sd(x[female_cols]) == 0) return(c(pvalue = NA_real_))
  c(pvalue = tryCatch(t.test(x[male_cols], x[female_cols])$p.value, error = function(e) NA_real_))
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

write.csv(count_mat, file.path(out_dir, "GSE151928_BAL_total_pseudobulk_counts_by_subject.csv"))
write.csv(logcpm, file.path(out_dir, "GSE151928_BAL_total_pseudobulk_logCPM_by_subject.csv"))
write.csv(stats, file.path(out_dir, "GSE151928_BAL_total_sex_DE_welch_logCPM.csv"), row.names = FALSE)

sample_qc <- data.frame(
  subject = colnames(count_mat),
  sex = sex,
  age = sample_meta$age[match(colnames(count_mat), sample_meta$subject)],
  n_cells = pseudobulk_n_cells,
  library_size_UMI = lib_size,
  detected_genes_CPM_ge_1 = colSums(cpm >= 1),
  stringsAsFactors = FALSE
)
write.csv(sample_qc, file.path(out_dir, "GSE151928_BAL_total_sample_qc.csv"), row.names = FALSE)

read_gene_set <- function(path, col = "gene_symbol") {
  x <- read.csv(path, stringsAsFactors = FALSE)
  unique(toupper(x[[col]]))
}
gene_sets <- list(
  inmunometabolismo_expandida = readLines(file.path(project_dir, "02_tablas_y_listas", "genes_inmunometabolismo_expandida_symbols.txt"), warn = FALSE),
  antiinflammatory_general = read_gene_set(file.path(project_dir, "02_tablas_y_listas", "antiinflammatory_general_curated_gene_list.csv")),
  proinflammatory_general = read_gene_set(file.path(project_dir, "02_tablas_y_listas", "proinflammatory_general_curated_gene_list.csv"))
)

for (set_name in names(gene_sets)) {
  overlap <- stats[toupper(stats$gene_symbol) %in% toupper(gene_sets[[set_name]]), ]
  write.csv(overlap, file.path(out_dir, paste0("GSE151928_BAL_total_", set_name, "_sex_DE_overlap.csv")), row.names = FALSE)
}

summary_rows <- do.call(rbind, lapply(names(gene_sets), function(set_name) {
  overlap <- stats[toupper(stats$gene_symbol) %in% toupper(gene_sets[[set_name]]), ]
  data.frame(
    gene_set = set_name,
    n_genes_in_set = length(unique(toupper(gene_sets[[set_name]]))),
    n_detected_in_dataset = sum(overlap$keep_for_test, na.rm = TRUE),
    n_nominal_p_lt_0_05 = sum(overlap$pvalue < 0.05, na.rm = TRUE),
    n_padj_lt_0_10 = sum(overlap$padj < 0.10, na.rm = TRUE),
    n_up_male_nominal = sum(overlap$pvalue < 0.05 & overlap$logFC_male_vs_female > 0, na.rm = TRUE),
    n_up_female_nominal = sum(overlap$pvalue < 0.05 & overlap$logFC_male_vs_female < 0, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
write.csv(summary_rows, file.path(out_dir, "GSE151928_BAL_total_gene_set_overlap_summary.csv"), row.names = FALSE)

message("Done. Outputs in: ", out_dir)
