base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda/raw_matrices"
out_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/GSE30595_M1_2026/06_datasets_externos/GMCSF_basales_busqueda"

y_entrez <- c(DDX3Y = "8653", KDM5D = "8284", EIF1AY = "9086", RPS4Y1 = "6192", UTY = "7404", ZFY = "7544")
xist_entrez <- "7503"

read_one <- function(file) {
  x <- read.delim(gzfile(file), check.names = FALSE, stringsAsFactors = FALSE)
  val <- gsub(",", ".", x[[2]], fixed = TRUE)
  data.frame(id = as.character(x[[1]]), value = as.numeric(val))
}

infer_files <- function(dataset, pattern, keep_pattern) {
  d <- file.path(base_dir, paste0(dataset, "_RAW"))
  files <- list.files(d, pattern = pattern, full.names = TRUE)
  files <- files[grepl(keep_pattern, basename(files))]
  out <- lapply(files, function(f) {
    x <- read_one(f)
    ids_for_y <- if (any(y_entrez %in% x$id)) y_entrez else names(y_entrez)
    id_for_x <- if (xist_entrez %in% x$id) xist_entrez else "XIST"
    y <- x$value[match(ids_for_y, x$id)]
    xist <- x$value[match(id_for_x, x$id)]
    y_signal <- mean(log2(y + 1), na.rm = TRUE)
    data.frame(dataset = dataset, sample = sub("_FPKM.*|\\.FPKM.*|_fpkm.*|\\.fpkm.*", "", sub("^GSM[0-9]+_", "", basename(f))),
               y_marker_log = round(y_signal, 3), xist_log = round(log2(xist + 1), 3),
               sex_inferido = ifelse(y_signal >= 1, "M_inferido", "F_inferido"),
               stringsAsFactors = FALSE)
  })
  do.call(rbind, out)
}

res <- rbind(
  infer_files("GSE224845", "FPKM.*gz$", "GM-UNT-4h|GM-UNT-12h|GM-UNT-36h"),
  infer_files("GSE232044", "fpkm.*gz$", "M1_[A-D]_"),
  infer_files("GSE256208", "FPKM.*gz$", "M1_[ABC]\\.FPKM"),
  infer_files("GSE266236", "fpkm.*gz$", "M1_siCNT_[ABC]"),
  infer_files("GSE156696", "fpkm.*gz$", "AGA13|AGA14|AGA15")
)

write.table(res, file.path(out_dir, "sex_inference_GMCSF_raw_fpkm_candidates.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(res)
cat("\nResumen por dataset:\n")
print(with(res, table(dataset, sex_inferido)))
