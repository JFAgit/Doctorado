.libPaths(c("C:/Users/fran_/Documents/Doctorado/Inicios/Dani/.Rlib", .libPaths()))
library(pheatmap)

base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Analisis hormonas"
out_prefix <- file.path(base_dir, "pheatmap_genes_XY_muestras_hormonales")

sex_markers <- data.frame(
  gene = c(
    "DDX3X", "EIF1AX", "KDM5C", "KDM6A", "RPS4X", "TMSB4X", "XIST", "ZFX",
    "DDX3Y", "EIF1AY", "KDM5D", "RPS4Y1", "RPS4Y2", "SRY", "TBL1Y",
    "TMSB4Y", "USP9Y", "UTY", "ZFY"
  ),
  chr = c(rep("X", 8), rep("Y", 11)),
  accession = c(
    "NM_001356", "NM_001412", "NM_004187", "NM_021140", "NM_001007",
    "NM_021109", "NR_001564", "NM_003410",
    "NM_004660", "NM_004681", "NM_004653", "NM_001008", "NM_001039567",
    "NM_003140", "NM_033284", "NM_004202", "NM_004654", "NM_007125",
    "NM_003411"
  ),
  stringsAsFactors = FALSE
)

donante <- function(id) {
  if (grepl("GSM7589(02|65|66|67)$", id)) return("M20")
  if (grepl("GSM7589(69|70|71|74)$", id)) return("M21")
  if (grepl("GSM7589(11|12|22)$", id)) return("M23_excluido")
  if (grepl("GSM7589(42|25|26)$", id)) return("M24")
  if (grepl("GSM7589(04|05|06|09)$", id)) return("M25")
  if (grepl("GSM7589(57|58|59|62)$", id)) return("M27")
  "Otro"
}

tratamiento <- function(id) {
  if (grepl("GSM758(905|911|925|958|966|970)$", id)) return("Estrogeno")
  if (grepl("GSM758(906|912|926|959|967|971)$", id)) return("Progesterona")
  if (grepl("GSM758(902|909|922|962|974)$", id)) return("Combo")
  "Otro"
}

leer_muestra <- function(file) {
  id <- sub("\\.txt$", "", basename(file))
  x <- read.delim(
    file, header = FALSE, sep = "\t", quote = "", comment.char = "",
    stringsAsFactors = FALSE, fill = TRUE
  )
  dat <- x[x$V1 == "DATA" & x$V6 == "0", c("V7", "V8", "V11")]
  colnames(dat) <- c("probe", "accession", "signal")
  dat$signal <- suppressWarnings(as.numeric(dat$signal))
  dat <- dat[dat$accession %in% sex_markers$accession & is.finite(dat$signal), ]
  if (nrow(dat) == 0) return(NULL)

  merged <- merge(dat, sex_markers, by = "accession")
  merged$expr_log2 <- log2(merged$signal + 1)
  values <- aggregate(expr_log2 ~ gene + chr, merged, mean)
  values$sample <- id
  values
}

files <- list.files(file.path(base_dir, "RawHormonas"), pattern = "^GSM.*\\.txt$", full.names = TRUE)
long <- do.call(rbind, lapply(files, leer_muestra))

genes_present <- unique(long[, c("gene", "chr")])
gene_order <- c(
  sex_markers$gene[sex_markers$chr == "Y"],
  sex_markers$gene[sex_markers$chr == "X"]
)
genes_present <- genes_present[match(intersect(gene_order, genes_present$gene), genes_present$gene), ]
samples <- sub("\\.txt$", "", basename(files))

mat <- matrix(NA_real_, nrow = nrow(genes_present), ncol = length(samples))
rownames(mat) <- genes_present$gene
colnames(mat) <- samples

for (i in seq_len(nrow(long))) {
  mat[long$gene[i], long$sample[i]] <- long$expr_log2[i]
}

sample_annot <- data.frame(
  Donante = vapply(colnames(mat), donante, character(1)),
  Tratamiento = vapply(colnames(mat), tratamiento, character(1)),
  row.names = colnames(mat)
)

donor_order <- c("M20", "M21", "M23_excluido", "M24", "M25", "M27", "Otro")
treat_order <- c("Estrogeno", "Progesterona", "Combo", "Otro")
ord <- order(
  match(sample_annot$Donante, donor_order),
  match(sample_annot$Tratamiento, treat_order),
  rownames(sample_annot)
)
mat <- mat[, ord, drop = FALSE]
sample_annot <- sample_annot[ord, , drop = FALSE]
gaps_row <- sum(genes_present$chr == "Y")

annotation_colors <- list(
  Tratamiento = c(Estrogeno = "#E78AC3", Progesterona = "#A6D854", Combo = "#FFD92F")
)

write.csv(mat, paste0(out_prefix, "_matriz_log2.csv"))
write.csv(genes_present, paste0(out_prefix, "_genes_presentes.csv"), row.names = FALSE)

breaks <- seq(-2.5, 2.5, length.out = 101)
colors <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)

pheatmap(
  mat,
  scale = "row",
  color = colors,
  breaks = breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (log2, z-score por gen)",
  filename = paste0(out_prefix, "_zscore.png"),
  width = 12,
  height = 6.8
)

pheatmap(
  mat,
  scale = "row",
  color = colors,
  breaks = breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (log2, z-score por gen)",
  filename = paste0(out_prefix, "_zscore.pdf"),
  width = 12,
  height = 6.8
)

message("Genes presentes: ", paste(rownames(mat), collapse = ", "))
message("PNG: ", paste0(out_prefix, "_zscore.png"))
message("PDF: ", paste0(out_prefix, "_zscore.pdf"))

raw_breaks <- seq(floor(min(mat, na.rm = TRUE)), ceiling(max(mat, na.rm = TRUE)), length.out = 101)
raw_colors <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)

pheatmap(
  mat,
  scale = "none",
  color = raw_colors,
  breaks = raw_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (log2 expresion)",
  filename = paste0(out_prefix, "_log2_expr.png"),
  width = 12,
  height = 6.8
)

pheatmap(
  mat,
  scale = "none",
  color = raw_colors,
  breaks = raw_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (log2 expresion)",
  filename = paste0(out_prefix, "_log2_expr.pdf"),
  width = 12,
  height = 6.8
)

message("PNG log2: ", paste0(out_prefix, "_log2_expr.png"))
message("PDF log2: ", paste0(out_prefix, "_log2_expr.pdf"))

minmax_row <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) {
    return(rep(0.5, length(x)))
  }
  (x - rng[1]) / (rng[2] - rng[1])
}

mat_minmax <- t(apply(mat, 1, minmax_row))
colnames(mat_minmax) <- colnames(mat)

write.csv(mat_minmax, paste0(out_prefix, "_matriz_minmax_por_gen.csv"))

minmax_breaks <- seq(0, 1, length.out = 101)
minmax_colors <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)

pheatmap(
  mat_minmax,
  scale = "none",
  color = minmax_colors,
  breaks = minmax_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (normalizado 0-1 por gen)",
  filename = paste0(out_prefix, "_minmax_por_gen.png"),
  width = 12,
  height = 6.8
)

pheatmap(
  mat_minmax,
  scale = "none",
  color = minmax_colors,
  breaks = minmax_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (normalizado 0-1 por gen)",
  filename = paste0(out_prefix, "_minmax_por_gen.pdf"),
  width = 12,
  height = 6.8
)

message("PNG min-max: ", paste0(out_prefix, "_minmax_por_gen.png"))
message("PDF min-max: ", paste0(out_prefix, "_minmax_por_gen.pdf"))

center_minus1_1_row <- function(x) {
  center <- mean(x, na.rm = TRUE)
  denom <- max(abs(x - center), na.rm = TRUE)
  if (!is.finite(center) || !is.finite(denom) || denom == 0) {
    return(rep(0, length(x)))
  }
  (x - center) / denom
}

mat_centered <- t(apply(mat, 1, center_minus1_1_row))
colnames(mat_centered) <- colnames(mat)

write.csv(mat_centered, paste0(out_prefix, "_matriz_centrada_menos1_1_por_gen.csv"))

centered_breaks <- seq(-1, 1, length.out = 101)
centered_colors <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)

pheatmap(
  mat_centered,
  scale = "none",
  color = centered_colors,
  breaks = centered_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (centrado -1 a 1 por gen)",
  filename = paste0(out_prefix, "_centrado_menos1_1_por_gen.png"),
  width = 12,
  height = 6.8
)

pheatmap(
  mat_centered,
  scale = "none",
  color = centered_colors,
  breaks = centered_breaks,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  gaps_row = gaps_row,
  annotation_col = sample_annot,
  annotation_row = NULL,
  annotation_colors = annotation_colors,
  border_color = "grey85",
  fontsize = 9,
  fontsize_row = 10,
  fontsize_col = 8,
  angle_col = 45,
  main = "Genes ligados a X/Y en muestras hormonales (centrado -1 a 1 por gen)",
  filename = paste0(out_prefix, "_centrado_menos1_1_por_gen.pdf"),
  width = 12,
  height = 6.8
)

message("PNG centrado -1 a 1: ", paste0(out_prefix, "_centrado_menos1_1_por_gen.png"))
message("PDF centrado -1 a 1: ", paste0(out_prefix, "_centrado_menos1_1_por_gen.pdf"))
