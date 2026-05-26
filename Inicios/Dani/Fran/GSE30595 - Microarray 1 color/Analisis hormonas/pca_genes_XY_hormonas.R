.libPaths(c("C:/Users/fran_/Documents/Doctorado/Inicios/Dani/.Rlib", .libPaths()))

base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Analisis hormonas"
infile <- file.path(base_dir, "pheatmap_genes_XY_muestras_hormonales_matriz_log2.csv")
out_prefix <- file.path(base_dir, "pca_genes_XY_muestras_hormonales")

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

mat <- read.csv(infile, row.names = 1, check.names = FALSE)
mat <- as.matrix(mat)

keep <- apply(mat, 1, function(x) all(is.finite(x)) && sd(x) > 0)
mat <- mat[keep, , drop = FALSE]

pca <- prcomp(t(mat), center = TRUE, scale. = TRUE)
var_exp <- (pca$sdev^2) / sum(pca$sdev^2)

scores <- as.data.frame(pca$x)
scores$sample <- rownames(scores)
scores$Donante <- vapply(scores$sample, donante, character(1))
scores$Tratamiento <- vapply(scores$sample, tratamiento, character(1))
scores <- scores[, c("sample", "Donante", "Tratamiento", colnames(pca$x))]

loadings <- as.data.frame(pca$rotation)
loadings$gene <- rownames(loadings)
loadings <- loadings[, c("gene", colnames(pca$rotation))]

write.csv(scores, paste0(out_prefix, "_scores.csv"), row.names = FALSE)
write.csv(loadings, paste0(out_prefix, "_loadings.csv"), row.names = FALSE)

donor_levels <- c("M20", "M21", "M23_excluido", "M24", "M25", "M27", "Otro")
donor_cols <- c(
  M20 = "#1B9E77",
  M21 = "#D95F02",
  M23_excluido = "#7570B3",
  M24 = "#E7298A",
  M25 = "#66A61E",
  M27 = "#E6AB02",
  Otro = "#666666"
)

treat_pch <- c(
  Estrogeno = 16,
  Progesterona = 17,
  Combo = 15,
  Otro = 1
)

plot_pca <- function(filename, device = c("png", "pdf")) {
  device <- match.arg(device)
  if (device == "png") {
    png(filename, width = 1800, height = 1400, res = 180)
  } else {
    pdf(filename, width = 10, height = 7.8)
  }
  on.exit(dev.off())

  par(mar = c(5, 5, 4, 12), xpd = NA)
  plot(
    scores$PC1, scores$PC2,
    col = donor_cols[scores$Donante],
    pch = treat_pch[scores$Tratamiento],
    cex = 1.25,
    xlab = sprintf("PC1 (%.1f%%)", 100 * var_exp[1]),
    ylab = sprintf("PC2 (%.1f%%)", 100 * var_exp[2]),
    main = "PCA de genes ligados a X/Y en muestras hormonales",
    sub = "Matriz log2, genes centrados y escalados para PCA"
  )
  grid(col = "grey88")
  abline(h = 0, v = 0, col = "grey70", lty = 2)
  points(
    scores$PC1, scores$PC2,
    col = donor_cols[scores$Donante],
    pch = treat_pch[scores$Tratamiento],
    cex = 1.25
  )
  text(scores$PC1, scores$PC2, labels = scores$sample, pos = 3, cex = 0.6)

  legend(
    "topright",
    inset = c(-0.34, 0),
    legend = donor_levels[donor_levels %in% scores$Donante],
    col = donor_cols[donor_levels[donor_levels %in% scores$Donante]],
    pch = 19,
    title = "Donante",
    bty = "n",
    cex = 0.85
  )
  legend(
    "right",
    inset = c(-0.34, 0),
    legend = names(treat_pch)[names(treat_pch) %in% scores$Tratamiento],
    pch = treat_pch[names(treat_pch) %in% scores$Tratamiento],
    col = "black",
    title = "Tratamiento",
    bty = "n",
    cex = 0.85
  )
}

plot_pca(paste0(out_prefix, "_PC1_PC2.png"), "png")
plot_pca(paste0(out_prefix, "_PC1_PC2.pdf"), "pdf")

message(sprintf("PC1: %.1f%%", 100 * var_exp[1]))
message(sprintf("PC2: %.1f%%", 100 * var_exp[2]))
message("PNG: ", paste0(out_prefix, "_PC1_PC2.png"))
message("PDF: ", paste0(out_prefix, "_PC1_PC2.pdf"))
message("Scores: ", paste0(out_prefix, "_scores.csv"))
message("Loadings: ", paste0(out_prefix, "_loadings.csv"))

y_genes <- intersect(
  c("EIF1AY", "KDM5D", "RPS4Y2", "TBL1Y", "TMSB4Y", "UTY"),
  rownames(mat)
)
mat_y <- mat[y_genes, , drop = FALSE]

pca_y <- prcomp(t(mat_y), center = TRUE, scale. = TRUE)
var_exp_y <- (pca_y$sdev^2) / sum(pca_y$sdev^2)

scores_y <- as.data.frame(pca_y$x)
scores_y$sample <- rownames(scores_y)
scores_y$Donante <- vapply(scores_y$sample, donante, character(1))
scores_y$Tratamiento <- vapply(scores_y$sample, tratamiento, character(1))
scores_y <- scores_y[, c("sample", "Donante", "Tratamiento", colnames(pca_y$x))]

loadings_y <- as.data.frame(pca_y$rotation)
loadings_y$gene <- rownames(loadings_y)
loadings_y <- loadings_y[, c("gene", colnames(pca_y$rotation))]

write.csv(scores_y, paste0(out_prefix, "_soloY_scores.csv"), row.names = FALSE)
write.csv(loadings_y, paste0(out_prefix, "_soloY_loadings.csv"), row.names = FALSE)

plot_pca_y <- function(filename, device = c("png", "pdf")) {
  device <- match.arg(device)
  if (device == "png") {
    png(filename, width = 1800, height = 1400, res = 180)
  } else {
    pdf(filename, width = 10, height = 7.8)
  }
  on.exit(dev.off())

  par(mar = c(5, 5, 4, 12), xpd = NA)
  plot(
    scores_y$PC1, scores_y$PC2,
    col = donor_cols[scores_y$Donante],
    pch = treat_pch[scores_y$Tratamiento],
    cex = 1.25,
    xlab = sprintf("PC1 (%.1f%%)", 100 * var_exp_y[1]),
    ylab = sprintf("PC2 (%.1f%%)", 100 * var_exp_y[2]),
    main = "PCA de genes ligados al Y en muestras hormonales",
    sub = "Matriz log2, genes centrados y escalados para PCA"
  )
  grid(col = "grey88")
  abline(h = 0, v = 0, col = "grey70", lty = 2)
  points(
    scores_y$PC1, scores_y$PC2,
    col = donor_cols[scores_y$Donante],
    pch = treat_pch[scores_y$Tratamiento],
    cex = 1.25
  )
  text(scores_y$PC1, scores_y$PC2, labels = scores_y$sample, pos = 3, cex = 0.6)

  legend(
    "topright",
    inset = c(-0.34, 0),
    legend = donor_levels[donor_levels %in% scores_y$Donante],
    col = donor_cols[donor_levels[donor_levels %in% scores_y$Donante]],
    pch = 19,
    title = "Donante",
    bty = "n",
    cex = 0.85
  )
  legend(
    "right",
    inset = c(-0.34, 0),
    legend = names(treat_pch)[names(treat_pch) %in% scores_y$Tratamiento],
    pch = treat_pch[names(treat_pch) %in% scores_y$Tratamiento],
    col = "black",
    title = "Tratamiento",
    bty = "n",
    cex = 0.85
  )
}

plot_pca_y(paste0(out_prefix, "_soloY_PC1_PC2.png"), "png")
plot_pca_y(paste0(out_prefix, "_soloY_PC1_PC2.pdf"), "pdf")

message(sprintf("Solo Y PC1: %.1f%%", 100 * var_exp_y[1]))
message(sprintf("Solo Y PC2: %.1f%%", 100 * var_exp_y[2]))
message("PNG solo Y: ", paste0(out_prefix, "_soloY_PC1_PC2.png"))
message("PDF solo Y: ", paste0(out_prefix, "_soloY_PC1_PC2.pdf"))
message("Scores solo Y: ", paste0(out_prefix, "_soloY_scores.csv"))
message("Loadings solo Y: ", paste0(out_prefix, "_soloY_loadings.csv"))
