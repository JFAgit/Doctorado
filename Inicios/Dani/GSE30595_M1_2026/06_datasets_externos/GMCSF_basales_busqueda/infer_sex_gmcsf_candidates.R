base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda/raw_matrices"
out_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda"

y_ens <- c(
  DDX3Y = "ENSG00000067048",
  KDM5D = "ENSG00000012817",
  EIF1AY = "ENSG00000198692",
  RPS4Y1 = "ENSG00000129824",
  UTY = "ENSG00000183878",
  ZFY = "ENSG00000067646"
)
xist_ens <- "ENSG00000229807"
y_sym <- names(y_ens)

infer_from_values <- function(mat, ids, xist_id, y_ids, dataset, samples_keep = NULL) {
  rownames(mat) <- sub("\\..*", "", ids)
  if (!is.null(samples_keep)) mat <- mat[, intersect(samples_keep, colnames(mat)), drop = FALSE]
  y_present <- intersect(y_ids, rownames(mat))
  x_present <- intersect(xist_id, rownames(mat))
  y_signal <- if (length(y_present)) colMeans(log2(mat[y_present, , drop = FALSE] + 1), na.rm = TRUE) else rep(NA_real_, ncol(mat))
  xist <- if (length(x_present)) as.numeric(log2(mat[x_present[1], ] + 1)) else rep(NA_real_, ncol(mat))
  call <- ifelse(!is.na(y_signal) & y_signal >= 1, "M_inferido", "F_inferido")
  data.frame(dataset = dataset, sample = colnames(mat), y_marker_log = round(y_signal, 3),
             xist_log = round(xist, 3), sex_inferido = call, stringsAsFactors = FALSE)
}

results <- list()

x <- read.csv(gzfile(file.path(base_dir, "GSE160862_tablecounts_raw.csv.gz")), check.names = FALSE)
ids <- x[[1]]
mat <- as.matrix(x[, -1])
mode(mat) <- "numeric"
results[["GSE160862"]] <- infer_from_values(
  mat, ids, xist_ens, unname(y_ens), "GSE160862",
  c("D367T01", "D367T07", "D367T13", "D367T19", "D367T25")
)

x <- read.csv(gzfile(file.path(base_dir, "GSE160863_tablecounts_raw_batch2.csv.gz")), check.names = FALSE)
ids <- x[[1]]
mat <- as.matrix(x[, -1])
mode(mat) <- "numeric"
results[["GSE160863"]] <- infer_from_values(
  mat, ids, xist_ens, unname(y_ens), "GSE160863",
  c("D440T01", "D440T05", "D440T09", "D440T13", "D440T17")
)

infer_symbol_matrix <- function(file, dataset) {
  x <- read.delim(gzfile(file.path(base_dir, file)), check.names = FALSE)
  gm_cols <- grep("M1", colnames(x), value = TRUE)
  mat <- as.matrix(x[, gm_cols])
  mode(mat) <- "numeric"
  symbol <- x[["Gene Symbol"]]
  y_present <- intersect(y_sym, symbol)
  x_present <- intersect("XIST", symbol)
  y_signal <- colMeans(log2(mat[match(y_present, symbol), , drop = FALSE] + 1), na.rm = TRUE)
  xist <- if (length(x_present)) as.numeric(log2(mat[match("XIST", symbol), ] + 1)) else rep(NA_real_, ncol(mat))
  data.frame(dataset = dataset, sample = colnames(mat),
             y_marker_log = round(y_signal, 3), xist_log = round(xist, 3),
             sex_inferido = ifelse(y_signal >= 1, "M_inferido", "F_inferido"),
             stringsAsFactors = FALSE)
}

results[["GSE102492"]] <- rbind(
  infer_symbol_matrix("GSE102492_Bazzi_et_al_RPKM_04_25_2017.txt.gz", "GSE102492"),
  infer_symbol_matrix("GSE102492_Bazzi.07.04.2021.txt.gz", "GSE102492")
)

x <- read.delim(gzfile(file.path(base_dir, "GSE304218_counts.tsv.gz")), check.names = FALSE)
control_cols <- grep("_control$", colnames(x), value = TRUE)
mat <- as.matrix(x[, control_cols])
mode(mat) <- "numeric"
results[["GSE304218"]] <- infer_from_values(
  mat, sub("\\..*", "", x[["Geneid"]]), xist_ens, unname(y_ens), "GSE304218"
)

res <- do.call(rbind, results)
write.table(res, file.path(out_dir, "sex_inference_GMCSF_candidates.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(res)
cat("\nResumen por dataset:\n")
print(with(res, table(dataset, sex_inferido)))
