base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda/raw_matrices"

files <- c(
  "GSE160862_tablecounts_raw.csv.gz",
  "GSE160863_tablecounts_raw_batch2.csv.gz",
  "GSE102492_Bazzi_et_al_RPKM_04_25_2017.txt.gz",
  "GSE304218_counts.tsv.gz"
)

for (f in files) {
  cat("\n##", f, "\n")
  p <- file.path(base_dir, f)
  sep <- if (grepl("csv", f)) "," else "\t"
  x <- read.table(gzfile(p), sep = sep, header = TRUE, nrows = 4,
                  check.names = FALSE, quote = "\"", comment.char = "")
  print(dim(x))
  print(names(x)[seq_len(min(12, ncol(x)))])
  print(x[seq_len(min(3, nrow(x))), seq_len(min(6, ncol(x)))])
}
