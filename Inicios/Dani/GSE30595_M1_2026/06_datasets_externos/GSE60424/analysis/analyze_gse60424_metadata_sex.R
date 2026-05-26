dataset_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GSE60424"

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

read_series_meta <- function(path) {
  lines <- readLines(gzfile(path, "rt"), warn = FALSE)
  start <- grep("!series_matrix_table_begin", lines, fixed = TRUE)[1]
  meta_lines <- lines[seq_len(start - 1)]
  get_one <- function(key) {
    row <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    if (!length(row)) return(NULL)
    vals <- strsplit(row[1], "\t", fixed = TRUE)[[1]][-1]
    gsub('^"|"$', "", vals)
  }
  get_all <- function(key) {
    rows <- meta_lines[grepl(paste0("^", key, "\t"), meta_lines)]
    lapply(rows, function(row) {
      vals <- strsplit(row, "\t", fixed = TRUE)[[1]][-1]
      gsub('^"|"$', "", vals)
    })
  }
  chars <- get_all("!Sample_characteristics_ch1")
  char_mat <- do.call(cbind, lapply(chars, function(x) {
    key <- sub(":.*$", "", x[1])
    val <- sub("^[^:]+: ?", "", x)
    data.frame(setNames(list(val), key), check.names = FALSE)
  }))
  data.frame(
    sample_title = get_one("!Sample_title"),
    geo_accession = get_one("!Sample_geo_accession"),
    source_name = get_one("!Sample_source_name_ch1"),
    char_mat,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

meta <- read_series_meta(file.path(dataset_dir, "metadata", "GSE60424_series_matrix.txt.gz"))
meta$gender <- trimws(meta$gender)
meta$sex_metadata <- ifelse(meta$gender == "M", "male", ifelse(meta$gender == "F", "female", NA))
meta$subject_id <- sub("_.*$", "", meta$samplename)
meta$sample_kind <- sub("^[^_]+_", "", meta$samplename)
meta$age_num <- suppressWarnings(as.numeric(meta$age))

counts <- read.delim(
  gzfile(file.path(dataset_dir, "processed", "GSE60424_GEOSubmit_FC1to11_normalized_counts.txt.gz")),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
names(counts)[1] <- "ensembl_id"
counts$ensembl_id <- sub("\\..*$", "", counts$ensembl_id)
marker <- counts[counts$ensembl_id %in% sex_ensembl, , drop = FALSE]
marker$gene_symbol <- names(sex_ensembl)[match(marker$ensembl_id, sex_ensembl)]
marker <- marker[, c("ensembl_id", "gene_symbol", setdiff(names(marker), c("ensembl_id", "gene_symbol")))]

sample_cols <- intersect(meta$sample_title, names(marker))
y_genes <- setdiff(unique(marker$gene_symbol), "XIST")
y_marker_rows <- marker$gene_symbol %in% y_genes
y_dynamic <- apply(marker[y_marker_rows, sample_cols, drop = FALSE], 1, function(x) diff(range(as.numeric(x), na.rm = TRUE)))
y_keep <- unique(marker$gene_symbol[y_marker_rows][order(y_dynamic, decreasing = TRUE)])
y_keep <- y_keep[seq_len(min(5, length(y_keep)))]

sex_scores <- do.call(rbind, lapply(sample_cols, function(s) {
  xist <- marker[marker$gene_symbol == "XIST", s, drop = TRUE]
  yvals <- unlist(marker[marker$gene_symbol %in% y_keep, s, drop = TRUE])
  data.frame(
    sample_title = s,
    xist_score = if (length(xist)) as.numeric(xist[1]) else NA_real_,
    y_marker_score = median(as.numeric(yvals), na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
y_cut <- mean(range(sex_scores$y_marker_score, na.rm = TRUE))
sex_scores$sex_inferred <- ifelse(sex_scores$y_marker_score > y_cut, "male", "female")

meta <- merge(meta, sex_scores, by = "sample_title", all.x = TRUE)
meta$sex_match_metadata <- ifelse(is.na(meta$sex_metadata), NA, meta$sex_metadata == meta$sex_inferred)

write.csv(meta, file.path(dataset_dir, "metadata", "GSE60424_sample_metadata_sex_checked.csv"), row.names = FALSE)
write.csv(marker, file.path(dataset_dir, "metadata", "GSE60424_sex_marker_normalized_counts.csv"), row.names = FALSE)

summary_disease_celltype <- as.data.frame(table(meta$diseasestatus, meta$celltype, useNA = "ifany"))
names(summary_disease_celltype) <- c("disease_status", "celltype", "n_samples")
write.csv(summary_disease_celltype, file.path(dataset_dir, "analysis", "GSE60424_sample_counts_by_disease_celltype.csv"), row.names = FALSE)

sample_summary <- aggregate(
  sample_title ~ diseasestatus + celltype + sex_metadata,
  data = meta,
  FUN = length
)
names(sample_summary)[4] <- "n_samples"
write.csv(sample_summary, file.path(dataset_dir, "analysis", "GSE60424_sample_counts_by_celltype_sex_disease.csv"), row.names = FALSE)

subject_summary <- unique(meta[, c("subject_id", "diseasestatus", "sex_metadata", "age_num")])
subject_counts <- aggregate(
  subject_id ~ diseasestatus + sex_metadata,
  data = subject_summary,
  FUN = length
)
names(subject_counts)[3] <- "n_subjects"
write.csv(subject_counts, file.path(dataset_dir, "analysis", "GSE60424_subject_counts_by_sex_disease.csv"), row.names = FALSE)

monocyte_hc <- meta[meta$celltype == "Monocytes" & meta$diseasestatus == "Healthy Control", ]
write.csv(monocyte_hc, file.path(dataset_dir, "analysis", "GSE60424_healthy_control_monocytes_metadata.csv"), row.names = FALSE)

cat("Samples:", nrow(meta), "\n")
cat("Unique subjects:", length(unique(meta$subject_id)), "\n")
cat("Healthy control monocytes:", nrow(monocyte_hc), "\n")
cat("Healthy control monocytes sex:\n")
print(table(monocyte_hc$sex_metadata, useNA = "ifany"))
cat("Sex metadata match among non-missing:", sum(meta$sex_match_metadata %in% TRUE, na.rm = TRUE), "/", sum(!is.na(meta$sex_match_metadata)), "\n")
