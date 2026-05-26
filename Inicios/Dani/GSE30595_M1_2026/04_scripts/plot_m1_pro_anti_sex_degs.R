overlap_path <- "M1_inmunometabolismo_expandida_DEG_overlap_by_sex.csv"
out_png <- "M1_inmunometabolismo_expandida_pro_anti_sex_vertical.png"
out_pdf <- "M1_inmunometabolismo_expandida_pro_anti_sex_vertical.pdf"
out_gene_table <- "M1_inmunometabolismo_expandida_pro_anti_sex_genes.csv"
out_summary <- "M1_inmunometabolismo_expandida_pro_anti_sex_summary.csv"

overlap <- read.csv(overlap_path, check.names = FALSE, stringsAsFactors = FALSE)

anti_genes <- c(
  "ABCG1", "ARG1", "ATG5", "BNIP3L", "CAT", "CD36", "EPAS1", "GPX4",
  "HPGD", "HPGDS", "LAMP2", "LIPA", "MSR1", "PTGER4", "RXRA", "SLC40A1",
  "SLC7A11", "SOD2", "SQSTM1", "TSC1", "TXN2", "UCP2", "ULK1"
)

pro_genes <- c(
  "ALOX5", "CYBB", "G6PD", "H6PD", "HK3", "HMGCR", "HMGCS1", "IDO1",
  "IKBKE", "KDM6B", "KYNU", "LTA4H", "NAMPT", "OLR1", "PDK3", "PDK4",
  "PFKP", "PGD", "PGK1", "PTGS2", "PYGL", "SLC2A1", "SLC2A3", "SLC2A6",
  "SLC7A1", "SLC7A5", "SLC38A1", "TBK1", "TFRC"
)

overlap$inflammatory_class <- "Context-dependent or unclassified"
overlap$inflammatory_class[overlap$gene_symbol %in% anti_genes] <- "Anti-inflammatory / resolving"
overlap$inflammatory_class[overlap$gene_symbol %in% pro_genes] <- "Pro-inflammatory / activation"
overlap$counted_for_plot <- overlap$inflammatory_class != "Context-dependent or unclassified"
overlap <- overlap[order(overlap$inflammatory_class, overlap$sex_overexpressed, overlap$gene_symbol), ]
write.csv(overlap, out_gene_table, row.names = FALSE)

summarise_target <- function(class_label, target_sex) {
  in_class <- overlap$inflammatory_class == class_label
  target <- in_class & overlap$sex_overexpressed == target_sex
  n_target <- sum(target)
  n_class <- sum(in_class)
  data.frame(
    metric = if (target_sex == "Male") {
      "Anti-inflammatory genes overexpressed in male"
    } else {
      "Pro-inflammatory genes overexpressed in female"
    },
    inflammatory_class = class_label,
    target_sex = target_sex,
    n_target = n_target,
    n_classified_class = n_class,
    percentage = 100 * n_target / n_class,
    genes = paste(overlap$gene_symbol[target], collapse = "; "),
    stringsAsFactors = FALSE
  )
}

summary_tbl <- rbind(
  summarise_target("Anti-inflammatory / resolving", "Male"),
  summarise_target("Pro-inflammatory / activation", "Female")
)
write.csv(summary_tbl, out_summary, row.names = FALSE)

vals <- summary_tbl$percentage
colors <- c("#2F64B5", "#C63D54")
label_text <- sprintf(
  "%s\n%d/%d genes\n(%.1f%%)",
  summary_tbl$target_sex,
  summary_tbl$n_target,
  summary_tbl$n_classified_class,
  summary_tbl$percentage
)

draw_plot <- function() {
  par(family = "sans", mar = c(5.0, 4.4, 3.0, 4.2), xpd = NA)
  x <- c(1.0, 2.15)
  bar_width <- 0.34
  plot(
    NA,
    xlim = c(0.45, 2.85),
    ylim = c(0, 105),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "Percentage of classified DEGs",
    bty = "l",
    cex.lab = 0.95
  )
  axis(2, at = seq(0, 100, 20), las = 1, cex.axis = 0.85)
  rect(x - bar_width / 2, 0, x + bar_width / 2, vals, col = colors, border = "black", lwd = 1.2)
  text(x, vals + 3.6, sprintf("%.1f%%", vals), font = 2, cex = 1.0)
  text(x + 0.25, pmin(vals + 14, 94), label_text, adj = c(0, 0.5), cex = 0.78)
  axis(
    1,
    at = x,
    labels = c("Anti-inflammatory\nmale OE", "Pro-inflammatory\nfemale OE"),
    tick = FALSE,
    line = 0.4,
    cex.axis = 0.88
  )
  mtext("Inflammatory direction in M1 immunometabolic DEGs", side = 3, line = 1.2, adj = 0, font = 2, cex = 1.0)
  mtext("Context-dependent genes excluded from denominator", side = 3, line = 0.05, adj = 0, cex = 0.78, col = "grey30")
}

png(out_png, width = 5.3, height = 4.4, units = "in", res = 600, bg = "white", pointsize = 8)
draw_plot()
dev.off()

pdf(out_pdf, width = 5.3, height = 4.4, bg = "white", pointsize = 8)
draw_plot()
dev.off()

print(summary_tbl)
cat("\nClassified DEG counts:\n")
print(table(overlap$inflammatory_class, overlap$sex_overexpressed))
cat("\nFiles written:\n", out_png, "\n", out_pdf, "\n", out_gene_table, "\n", out_summary, "\n", sep = "")
