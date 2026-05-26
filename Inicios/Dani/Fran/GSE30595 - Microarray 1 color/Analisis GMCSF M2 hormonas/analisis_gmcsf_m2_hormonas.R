.libPaths(c("C:/Users/fran_/Documents/Doctorado/Inicios/Dani/.Rlib", .libPaths()))

library(pheatmap)

base_dir <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani"
analysis_dir <- file.path(base_dir, "Fran/GSE30595 - Microarray 1 color/Analisis GMCSF M2 hormonas")
raw_dir <- file.path(analysis_dir, "RawSeleccion")
soft_file <- file.path(base_dir, "GSE30595_family.soft.gz")
out_prefix <- file.path(analysis_dir, "GMCSF_M2_hormonas")

dir.create(analysis_dir, recursive = TRUE, showWarnings = FALSE)

sample_meta <- data.frame(
  sample = c(
    "GSM758942", "GSM758928", "GSM758925", "GSM758926",
    "GSM758957", "GSM758961", "GSM758958", "GSM758959", "GSM758962"
  ),
  donor = c(
    "M24", "M24", "M24", "M24",
    "M27", "M27", "M27", "M27", "M27"
  ),
  condition = c(
    "GM-CSF", "GM-CSF+IL4/13", "GM-CSF+E", "GM-CSF+P",
    "GM-CSF", "GM-CSF+IL4/13", "GM-CSF+E", "GM-CSF+P", "Combo_separado"
  ),
  factor_geo = c(
    "GM-CSF", "GM-CSF + IL4/13", "GM-CSF + E", "GM-CSF + P",
    "GM-CSF", "GM-CSF + IL4/13", "GM-CSF + E", "GM-CSF + P",
    "GM-CSF + E/P/IL-10/4/13"
  ),
  stringsAsFactors = FALSE
)

condition_order <- c("GM-CSF", "GM-CSF+IL4/13", "GM-CSF+E", "GM-CSF+P", "Combo_separado")
sample_meta$condition <- factor(sample_meta$condition, levels = condition_order)
sample_meta <- sample_meta[order(sample_meta$donor, sample_meta$condition), ]

m2_canonical <- c(
  "MRC1", "CD163", "MSR1", "MERTK", "STAB1", "STAB2", "CD36", "CD200R1",
  "CCL18", "CCL22", "IL10", "IL10RA", "IL4R", "IL13RA1", "TGFB1", "TGFB2",
  "MAF", "MAFB", "KLF4", "STAT6", "PPARG", "TREM2", "VSIG4", "MARCO",
  "CLEC7A", "ARG1", "ARG2"
)

m2_metabolic <- c(
  "ARG1", "ARG2", "PPARG", "PPARGC1A", "PPARGC1B", "PPARA", "RXRA", "NR1H3",
  "ABCA1", "ABCG1", "APOE", "LPL", "LIPA", "CD36", "FABP4", "FABP5",
  "ALOX15", "ALOX15B", "HPGD", "HPGDS", "SLC40A1", "HMOX1", "NFE2L2",
  "CAT", "SOD2", "GPX4", "UCP2", "CPT1A", "CPT2", "ACADM", "ACADVL",
  "HADHA", "HADHB", "TFAM", "MTOR", "TSC1", "TSC2", "STK11", "PRKAA1",
  "PRKAA2", "SQSTM1", "ATG5", "BNIP3L", "LAMP2"
)

gene_panel <- unique(c(m2_canonical, m2_metabolic))
panel_info <- data.frame(
  gene = gene_panel,
  panel = ifelse(
    gene_panel %in% m2_canonical & gene_panel %in% m2_metabolic,
    "M2_canonico+inmunometabolico",
    ifelse(gene_panel %in% m2_canonical, "M2_canonico", "Inmunometabolico_M2")
  ),
  stringsAsFactors = FALSE
)

read_platform <- function(soft_file) {
  con <- gzfile(soft_file, open = "rt")
  on.exit(close(con), add = TRUE)

  header <- NULL
  rows <- character()
  in_table <- FALSE

  repeat {
    line <- readLines(con, n = 1)
    if (length(line) == 0) break

    if (line == "!platform_table_begin") {
      in_table <- TRUE
      header <- readLines(con, n = 1)
      next
    }

    if (in_table) {
      if (line == "!platform_table_end") break
      rows <- c(rows, line)
    }
  }

  if (is.null(header) || length(rows) == 0) {
    stop("No se pudo leer la tabla de plataforma desde el SOFT.")
  }

  tab <- read.delim(
    text = paste(c(header, rows), collapse = "\n"),
    sep = "\t", quote = "", comment.char = "", stringsAsFactors = FALSE,
    check.names = FALSE
  )

  colnames(tab)[1:7] <- c(
    "probe", "control_type", "gb_acc_1", "gb_acc_2", "entrez", "gene", "gene_name"
  )

  tab$gene <- trimws(tab$gene)
  tab <- tab[tab$probe != "" & tab$gene != "", c("probe", "gene", "gene_name")]
  tab[!duplicated(tab$probe), ]
}

read_sample <- function(file, platform) {
  id <- sub("\\.txt$", "", basename(file))
  x <- read.delim(
    file, header = FALSE, sep = "\t", quote = "", comment.char = "",
    stringsAsFactors = FALSE, fill = TRUE
  )

  dat <- x[x$V1 == "DATA" & x$V6 == "0", c("V7", "V11")]
  colnames(dat) <- c("probe", "signal")
  dat$signal <- suppressWarnings(as.numeric(dat$signal))
  dat <- dat[is.finite(dat$signal), ]

  dat <- merge(dat, platform, by = "probe", all.x = FALSE, all.y = FALSE)
  dat <- dat[dat$gene %in% gene_panel, ]
  if (nrow(dat) == 0) {
    return(NULL)
  }

  dat$expr_log2 <- log2(dat$signal + 1)
  agg <- aggregate(expr_log2 ~ gene, dat, mean)
  agg$sample <- id
  agg
}

clip <- function(x, min_value, max_value) {
  x[x < min_value] <- min_value
  x[x > max_value] <- max_value
  x
}

row_scale <- function(mat) {
  z <- t(scale(t(mat)))
  z[!is.finite(z)] <- 0
  z
}

make_score_table <- function(zmat, sample_meta, panel_genes, score_name) {
  genes <- intersect(panel_genes, rownames(zmat))
  values <- colMeans(zmat[genes, , drop = FALSE], na.rm = TRUE)
  data.frame(
    sample = names(values),
    score = as.numeric(values),
    score_name = score_name,
    stringsAsFactors = FALSE
  )
}

plot_score_lines <- function(score_wide, outfile_png, outfile_pdf) {
  draw <- function() {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)

    par(mfrow = c(1, 2), mar = c(8.5, 5.0, 3.5, 1.2))
    panels <- c("M2 canonico", "Inmunometabolico M2")
    donor_cols <- c(M24 = "#0072B2", M27 = "#D55E00")
    short_labels <- c("GM", "IL4/13", "E", "P", "Combo")

    for (panel in panels) {
      sub <- score_wide[score_wide$score_name == panel, ]
      yy <- range(sub$score, na.rm = TRUE)
      yy <- yy + c(-0.15, 0.15)

      plot(
        seq_along(condition_order), rep(NA_real_, length(condition_order)),
        ylim = yy, xaxt = "n", xlab = "", ylab = "Score medio (z-score)",
        main = panel, pch = 16
      )
      axis(1, at = seq_along(condition_order), labels = short_labels, las = 2)
      abline(h = 0, col = "grey80", lty = 2)

      for (donor in unique(sub$donor)) {
        d <- sub[sub$donor == donor, ]
        x <- match(as.character(d$condition), condition_order)
        ord <- order(x)
        lines(x[ord], d$score[ord], type = "b", pch = 16, lwd = 2, col = donor_cols[donor])
      }

      if (panel == panels[1]) {
        legend(
          "topright", legend = names(donor_cols), col = donor_cols,
          pch = 16, lwd = 2, bty = "n"
        )
      }
    }
  }

  png(outfile_png, width = 1800, height = 900, res = 160)
  draw()
  dev.off()

  pdf(outfile_pdf, width = 11, height = 5.5)
  draw()
  dev.off()
}

platform <- read_platform(soft_file)
files <- file.path(raw_dir, paste0(sample_meta$sample, ".txt"))
missing_files <- files[!file.exists(files)]
if (length(missing_files) > 0) {
  stop("Faltan archivos raw: ", paste(missing_files, collapse = ", "))
}

long <- do.call(rbind, lapply(files, read_sample, platform = platform))
if (is.null(long) || nrow(long) == 0) {
  stop("No se encontraron genes del panel en las muestras seleccionadas.")
}

genes_present <- intersect(gene_panel, unique(long$gene))
panel_info$present_in_array <- panel_info$gene %in% genes_present

mat <- matrix(NA_real_, nrow = length(genes_present), ncol = nrow(sample_meta))
rownames(mat) <- genes_present
colnames(mat) <- sample_meta$sample

for (i in seq_len(nrow(long))) {
  mat[long$gene[i], long$sample[i]] <- long$expr_log2[i]
}

keep_rows <- rowSums(is.finite(mat)) > 0
mat <- mat[keep_rows, , drop = FALSE]

sample_annot <- sample_meta
rownames(sample_annot) <- sample_annot$sample
sample_annot <- sample_annot[colnames(mat), c("donor", "condition"), drop = FALSE]
colnames(sample_annot) <- c("Donante", "Condicion")

row_annot <- panel_info[match(rownames(mat), panel_info$gene), "panel", drop = FALSE]
rownames(row_annot) <- rownames(mat)
colnames(row_annot) <- "Panel"

z <- row_scale(mat)
z_clip <- clip(z, -2, 2)

main_samples <- sample_meta$sample[sample_meta$condition != "Combo_separado"]
mat_main <- mat[, main_samples, drop = FALSE]
z_main <- row_scale(mat_main)
z_main <- clip(z_main, -2, 2)
sample_annot_main <- sample_annot[main_samples, , drop = FALSE]
labels_col_main <- paste(sample_annot_main$Donante, as.character(sample_annot_main$Condicion), sep = " ")
labels_col_all <- paste(sample_annot$Donante, as.character(sample_annot$Condicion), sep = " ")

write.csv(sample_meta, paste0(out_prefix, "_metadata_muestras.csv"), row.names = FALSE)
write.csv(panel_info, paste0(out_prefix, "_panel_genes_estado.csv"), row.names = FALSE)
write.csv(mat, paste0(out_prefix, "_matriz_log2_genes.csv"))
write.csv(z, paste0(out_prefix, "_matriz_zscore_genes.csv"))

annotation_colors <- list(
  Donante = c(M24 = "#0072B2", M27 = "#D55E00"),
  Condicion = c(
    "GM-CSF" = "#999999",
    "GM-CSF+IL4/13" = "#009E73",
    "GM-CSF+E" = "#CC79A7",
    "GM-CSF+P" = "#E69F00",
    "Combo_separado" = "#56B4E9"
  ),
  Panel = c(
    "M2_canonico" = "#009E73",
    "Inmunometabolico_M2" = "#E69F00",
    "M2_canonico+inmunometabolico" = "#0072B2"
  )
)

heat_colors <- colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100)

pheatmap(
  z_main,
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot_main,
  annotation_row = row_annot[rownames(z_main), , drop = FALSE],
  annotation_colors = annotation_colors,
  labels_col = labels_col_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 7,
  main = "GM-CSF basal vs IL4/13 y hormonas puras (z-score por gen)",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_sin_combo.png"),
  width = 8.5,
  height = 10
)

pheatmap(
  z_main,
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot_main,
  annotation_row = row_annot[rownames(z_main), , drop = FALSE],
  annotation_colors = annotation_colors,
  labels_col = labels_col_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 7,
  main = "GM-CSF basal vs IL4/13 y hormonas puras (z-score por gen)",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_sin_combo.pdf"),
  width = 8.5,
  height = 10
)

pheatmap(
  z_clip,
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot,
  annotation_row = row_annot[rownames(z_clip), , drop = FALSE],
  annotation_colors = annotation_colors,
  labels_col = labels_col_all,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 7,
  main = "GM-CSF basal, IL4/13, hormonas puras y combo separado",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_con_combo.png"),
  width = 9.3,
  height = 10
)

canonical_rows_main <- intersect(m2_canonical, rownames(z_main))
canonical_rows_all <- intersect(m2_canonical, rownames(z_clip))
write.csv(
  data.frame(gene = canonical_rows_main, stringsAsFactors = FALSE),
  paste0(out_prefix, "_genes_M2_canonicos_presentes.csv"),
  row.names = FALSE
)

pheatmap(
  z_main[canonical_rows_main, , drop = FALSE],
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot_main,
  annotation_colors = annotation_colors,
  labels_col = labels_col_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 8,
  main = "Genes M2 canonicos: GM-CSF basal vs IL4/13 y hormonas puras",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_solo_M2_canonico_sin_combo.png"),
  width = 8.5,
  height = 7.5
)

pheatmap(
  z_main[canonical_rows_main, , drop = FALSE],
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot_main,
  annotation_colors = annotation_colors,
  labels_col = labels_col_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 8,
  main = "Genes M2 canonicos: GM-CSF basal vs IL4/13 y hormonas puras",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_solo_M2_canonico_sin_combo.pdf"),
  width = 8.5,
  height = 7.5
)

pheatmap(
  z_clip[canonical_rows_all, , drop = FALSE],
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_col = sample_annot,
  annotation_colors = annotation_colors,
  labels_col = labels_col_all,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 8,
  main = "Genes M2 canonicos: GM-CSF basal, tratamientos y combo separado",
  filename = paste0(out_prefix, "_heatmap_expr_zscore_solo_M2_canonico_con_combo.png"),
  width = 9.3,
  height = 7.5
)

delta_rows <- list()
idx <- 1
for (donor in unique(sample_meta$donor)) {
  base_sample <- sample_meta$sample[sample_meta$donor == donor & sample_meta$condition == "GM-CSF"]
  if (length(base_sample) != 1) next
  for (cond in condition_order[condition_order != "GM-CSF"]) {
    test_sample <- sample_meta$sample[sample_meta$donor == donor & sample_meta$condition == cond]
    if (length(test_sample) != 1) next
    delta <- mat[, test_sample] - mat[, base_sample]
    delta_rows[[idx]] <- data.frame(
      gene = names(delta),
      donor = donor,
      condition = cond,
      sample = test_sample,
      baseline_sample = base_sample,
      delta_log2_vs_GMCSF = as.numeric(delta),
      stringsAsFactors = FALSE
    )
    idx <- idx + 1
  }
}

delta_long <- do.call(rbind, delta_rows)
delta_long <- merge(delta_long, panel_info[, c("gene", "panel")], by = "gene", all.x = TRUE)
write.csv(delta_long, paste0(out_prefix, "_deltas_log2_vs_GMCSF_por_donante.csv"), row.names = FALSE)

delta_summary <- aggregate(
  delta_log2_vs_GMCSF ~ gene + condition + panel,
  delta_long,
  function(x) c(n = length(x), mean = mean(x), sd = ifelse(length(x) > 1, sd(x), NA_real_))
)
delta_summary <- do.call(data.frame, delta_summary)
colnames(delta_summary) <- c("gene", "condition", "panel", "n", "mean_delta_log2", "sd_delta_log2")
delta_summary <- delta_summary[order(delta_summary$condition, -abs(delta_summary$mean_delta_log2)), ]
write.csv(delta_summary, paste0(out_prefix, "_resumen_delta_log2_vs_GMCSF_por_gen.csv"), row.names = FALSE)

delta_cols <- paste(delta_long$donor, delta_long$condition, sep = "_")
delta_mat <- matrix(NA_real_, nrow = nrow(mat), ncol = length(unique(delta_cols)))
rownames(delta_mat) <- rownames(mat)
colnames(delta_mat) <- unique(delta_cols)

for (i in seq_len(nrow(delta_long))) {
  cname <- paste(delta_long$donor[i], delta_long$condition[i], sep = "_")
  delta_mat[delta_long$gene[i], cname] <- delta_long$delta_log2_vs_GMCSF[i]
}

main_delta_cols <- colnames(delta_mat)[!grepl("Combo", colnames(delta_mat))]
desired_delta_cols <- unlist(lapply(unique(sample_meta$donor), function(donor) {
  paste(donor, condition_order[condition_order != "GM-CSF"], sep = "_")
}))
main_delta_cols <- intersect(desired_delta_cols[!grepl("Combo", desired_delta_cols)], main_delta_cols)
delta_mat_main <- delta_mat[, main_delta_cols, drop = FALSE]
delta_mat_main <- delta_mat_main[rowSums(is.finite(delta_mat_main)) > 0, , drop = FALSE]
write.csv(delta_mat, paste0(out_prefix, "_matriz_delta_log2_vs_GMCSF.csv"))

labels_delta_main <- gsub("GM-CSF\\+", "", colnames(delta_mat_main))
labels_delta_main <- gsub("_", " ", labels_delta_main)

pheatmap(
  clip(delta_mat_main, -2, 2),
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_row = row_annot[rownames(delta_mat_main), , drop = FALSE],
  annotation_colors = annotation_colors,
  labels_col = labels_delta_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 7,
  main = "Cambio log2 contra GM-CSF basal (sin combo)",
  filename = paste0(out_prefix, "_heatmap_delta_log2_vs_GMCSF_sin_combo.png"),
  width = 8,
  height = 10
)

pheatmap(
  clip(delta_mat_main, -2, 2),
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  annotation_row = row_annot[rownames(delta_mat_main), , drop = FALSE],
  annotation_colors = annotation_colors,
  labels_col = labels_delta_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 7,
  main = "Cambio log2 contra GM-CSF basal (sin combo)",
  filename = paste0(out_prefix, "_heatmap_delta_log2_vs_GMCSF_sin_combo.pdf"),
  width = 8,
  height = 10
)

canonical_delta_rows <- intersect(m2_canonical, rownames(delta_mat_main))

pheatmap(
  clip(delta_mat_main[canonical_delta_rows, , drop = FALSE], -2, 2),
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  labels_col = labels_delta_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 8,
  main = "Genes M2 canonicos: cambio log2 contra GM-CSF basal",
  filename = paste0(out_prefix, "_heatmap_delta_log2_vs_GMCSF_solo_M2_canonico_sin_combo.png"),
  width = 8,
  height = 7.5
)

pheatmap(
  clip(delta_mat_main[canonical_delta_rows, , drop = FALSE], -2, 2),
  color = heat_colors,
  breaks = seq(-2, 2, length.out = 101),
  cluster_cols = FALSE,
  cluster_rows = TRUE,
  labels_col = labels_delta_main,
  angle_col = "45",
  fontsize = 8,
  fontsize_col = 8,
  fontsize_row = 8,
  main = "Genes M2 canonicos: cambio log2 contra GM-CSF basal",
  filename = paste0(out_prefix, "_heatmap_delta_log2_vs_GMCSF_solo_M2_canonico_sin_combo.pdf"),
  width = 8,
  height = 7.5
)

score_all <- rbind(
  make_score_table(z, sample_meta, m2_canonical, "M2 canonico"),
  make_score_table(z, sample_meta, m2_metabolic, "Inmunometabolico M2")
)
score_all <- merge(score_all, sample_meta, by = "sample", all.x = TRUE)
score_all <- score_all[order(score_all$score_name, score_all$donor, score_all$condition), ]
write.csv(score_all, paste0(out_prefix, "_scores_M2_por_muestra.csv"), row.names = FALSE)

plot_score_lines(
  score_all,
  paste0(out_prefix, "_scores_M2_pareado.png"),
  paste0(out_prefix, "_scores_M2_pareado.pdf")
)

score_delta <- list()
idx <- 1
for (score_name in unique(score_all$score_name)) {
  sub <- score_all[score_all$score_name == score_name, ]
  for (donor in unique(sub$donor)) {
    base_score <- sub$score[sub$donor == donor & sub$condition == "GM-CSF"]
    if (length(base_score) != 1) next
    test <- sub[sub$donor == donor & sub$condition != "GM-CSF", ]
    if (nrow(test) == 0) next
    test$delta_score_vs_GMCSF <- test$score - base_score
    score_delta[[idx]] <- test
    idx <- idx + 1
  }
}
score_delta <- do.call(rbind, score_delta)
write.csv(score_delta, paste0(out_prefix, "_scores_M2_delta_vs_GMCSF.csv"), row.names = FALSE)

score_delta_summary <- aggregate(
  delta_score_vs_GMCSF ~ score_name + condition,
  score_delta,
  function(x) c(n = length(x), mean = mean(x), sd = ifelse(length(x) > 1, sd(x), NA_real_))
)
score_delta_summary <- do.call(data.frame, score_delta_summary)
colnames(score_delta_summary) <- c("score_name", "condition", "n", "mean_delta_score", "sd_delta_score")
write.csv(score_delta_summary, paste0(out_prefix, "_scores_M2_delta_vs_GMCSF_resumen.csv"), row.names = FALSE)

top_by_condition <- do.call(rbind, lapply(unique(delta_summary$condition), function(cond) {
  sub <- delta_summary[delta_summary$condition == cond, ]
  sub <- sub[order(-abs(sub$mean_delta_log2)), ]
  head(sub, 12)
}))
write.csv(top_by_condition, paste0(out_prefix, "_top_genes_delta_por_condicion.csv"), row.names = FALSE)

readme <- c(
  "Analisis GM-CSF basal vs IL4/13 y hormonas puras en GSE30595",
  "",
  "Diseno usado:",
  "- Donantes pareados completos con GM-CSF basal: M24 y M27.",
  "- Condiciones principales: GM-CSF, GM-CSF+IL4/13, GM-CSF+E, GM-CSF+P.",
  "- Combo separado: GM-CSF+E/P/IL-10/4/13, disponible solo para M27 en este subconjunto; no se usa como hormona pura.",
  "",
  "Notas metodologicas:",
  "- Se uso gProcessedSignal de Agilent.",
  "- Se transformo como log2(signal + 1).",
  "- Si habia mas de una sonda por gen, se promedio la expresion log2 por gen.",
  "- La comparacion principal es descriptiva y pareada contra GM-CSF basal, no DEG formal, porque n=2 donantes.",
  "",
  paste0("Genes del panel solicitados: ", length(gene_panel)),
  paste0("Genes presentes en la plataforma/datos: ", nrow(mat)),
  "",
  "Archivos principales:",
  "- *_heatmap_expr_zscore_sin_combo.png/pdf",
  "- *_heatmap_delta_log2_vs_GMCSF_sin_combo.png/pdf",
  "- *_scores_M2_pareado.png/pdf",
  "- *_deltas_log2_vs_GMCSF_por_donante.csv",
  "- *_resumen_delta_log2_vs_GMCSF_por_gen.csv",
  "- *_scores_M2_delta_vs_GMCSF_resumen.csv"
)

writeLines(readme, paste0(out_prefix, "_README.txt"))

message("Analisis terminado.")
message("Genes presentes: ", nrow(mat), " / ", length(gene_panel))
message("Salida: ", analysis_dir)
