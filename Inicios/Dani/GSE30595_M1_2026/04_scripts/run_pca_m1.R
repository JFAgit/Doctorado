args <- commandArgs(trailingOnly = TRUE)

expr_path <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Expresion/Criterio clasico/ExpresionM1Cruda.csv"
gene_list_path <- NA_character_
prefix <- "PCA_M1"

if (length(args) == 1) {
  if (file.exists(args[1])) {
    gene_list_path <- args[1]
  } else if (nzchar(args[1])) {
    prefix <- args[1]
  }
} else if (length(args) >= 2) {
  if (nzchar(args[1])) gene_list_path <- args[1]
  if (nzchar(args[2])) prefix <- args[2]
}

out_dir <- "C:/Users/fran_/Documents/Codex/2026-05-13/holaaaaaaa-tenes-los-datos-que-trabaje"
out_png <- file.path(out_dir, paste0(prefix, ".png"))
out_coords <- file.path(out_dir, paste0(prefix, "_coordinates.csv"))
out_sex <- file.path(out_dir, paste0(prefix, "_inferred_sex_from_Y_markers.csv"))
out_used_genes <- file.path(out_dir, paste0(prefix, "_genes_used.csv"))

expr <- read.csv(expr_path, check.names = FALSE, stringsAsFactors = FALSE)
colnames(expr)[1] <- "Gene"
expr <- expr[!is.na(expr$Gene) & expr$Gene != "", ]
expr <- expr[!duplicated(expr$Gene), ]
rownames(expr) <- expr$Gene

if (!is.na(gene_list_path)) {
  genes <- readLines(gene_list_path, warn = FALSE)
  genes <- unique(trimws(genes))
  genes <- genes[genes != ""]
  expr <- expr[expr$Gene %in% genes, , drop = FALSE]
}

mat <- as.matrix(expr[, -1, drop = FALSE])
storage.mode(mat) <- "numeric"
keep <- apply(mat, 1, function(x) all(is.finite(x)) && sd(x) > 0)
mat <- mat[keep, , drop = FALSE]

if (nrow(mat) < 2) {
  stop("Need at least two matched genes with non-zero variance for PCA.")
}

write.csv(data.frame(Gene = rownames(mat)), out_used_genes, row.names = FALSE)

markers_y <- intersect(c("UTY", "TMSB4Y", "EIF1AY", "RPS4Y1", "RPS4Y2", "USP9Y", "DDX3Y", "KDM5D", "ZFY"), rownames(mat))
if (length(markers_y) == 0) {
  full_mat <- as.matrix(read.csv(expr_path, check.names = FALSE, stringsAsFactors = FALSE)[, -1, drop = FALSE])
  markers_y <- character(0)
}

expr_all <- read.csv(expr_path, check.names = FALSE, stringsAsFactors = FALSE)
colnames(expr_all)[1] <- "Gene"
expr_all <- expr_all[!duplicated(expr_all$Gene), ]
rownames(expr_all) <- expr_all$Gene
markers_y_all <- intersect(c("UTY", "TMSB4Y", "EIF1AY", "RPS4Y1", "RPS4Y2", "USP9Y", "DDX3Y", "KDM5D", "ZFY"), rownames(expr_all))
y_mat <- as.matrix(expr_all[markers_y_all, -1, drop = FALSE])
storage.mode(y_mat) <- "numeric"
y_score <- colMeans(y_mat, na.rm = TRUE)
sex <- ifelse(y_score > 0.5, "Male", "Female")
meta <- data.frame(Sample = names(y_score), Sex_inferred = sex, Y_marker_mean = as.numeric(y_score), stringsAsFactors = FALSE)
write.csv(meta, out_sex, row.names = FALSE)

pca <- prcomp(t(mat), center = TRUE, scale. = TRUE)
var_exp <- round((pca$sdev^2 / sum(pca$sdev^2)) * 100, 1)
coords <- data.frame(Sample = rownames(pca$x), PC1 = pca$x[, 1], PC2 = pca$x[, 2], stringsAsFactors = FALSE)
coords <- merge(coords, meta, by = "Sample", sort = FALSE)
write.csv(coords, out_coords, row.names = FALSE)

cols <- c(Female = "#D9487D", Male = "#2563EB")
png(out_png, width = 1900, height = 1450, res = 180)
par(mar = c(6, 5.2, 4.2, 2.2), xpd = FALSE, las = 1)

range_with_pad <- function(x, pad = 0.16) {
  r <- range(x, finite = TRUE)
  d <- diff(r)
  if (d == 0) d <- max(abs(r), 1)
  c(r[1] - d * pad, r[2] + d * pad)
}

xlim <- range_with_pad(coords$PC1, 0.18)
ylim <- range_with_pad(coords$PC2, 0.20)
yr <- diff(ylim)
label_y <- coords$PC2 + yr * 0.035
label_y <- pmin(label_y, ylim[2] - yr * 0.035)
label_y <- pmax(label_y, ylim[1] + yr * 0.035)

plot(
  coords$PC1, coords$PC2,
  pch = 21, bg = cols[coords$Sex_inferred], col = "#111827", cex = 1.9, lwd = 1.2,
  xlab = paste0("PC1 (", var_exp[1], "%)"),
  ylab = paste0("PC2 (", var_exp[2], "%)"),
  main = paste0("PCA M1 - ", nrow(mat), " genes"),
  xlim = xlim,
  ylim = ylim,
  axes = FALSE
)
axis(1, las = 1)
axis(2, las = 1)
box()
grid(col = "#e5e7eb", lty = 1)
points(coords$PC1, coords$PC2, pch = 21, bg = cols[coords$Sex_inferred], col = "#111827", cex = 1.9, lwd = 1.2)
text(coords$PC1, label_y, labels = coords$Sample, cex = 0.72, col = "#111827", srt = 0)
legend("topright", inset = 0.02, legend = names(cols), pt.bg = cols, pch = 21, pt.cex = 1.35, bty = "n", title = "Sex inferred")
mtext(paste0("Sex inferred from Y markers: ", paste(markers_y_all, collapse = ", ")), side = 1, line = 4, cex = 0.65, col = "#4b5563")
dev.off()

cat("genes_used=", nrow(mat), "\n", sep = "")
cat("pc1_var=", var_exp[1], "\n", sep = "")
cat("pc2_var=", var_exp[2], "\n", sep = "")
cat("png=", out_png, "\n", sep = "")
cat("coords=", out_coords, "\n", sep = "")
cat("sex_table=", out_sex, "\n", sep = "")
