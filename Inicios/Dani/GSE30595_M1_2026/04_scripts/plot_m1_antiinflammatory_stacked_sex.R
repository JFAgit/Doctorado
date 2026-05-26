overlap_path <- "M1_inmunometabolismo_expandida_DEG_overlap_by_sex.csv"
out_png <- "M1_antiinflammatory_genes_stacked_sex_vertical.png"
out_pdf <- "M1_antiinflammatory_genes_stacked_sex_vertical.pdf"
out_summary <- "M1_antiinflammatory_genes_stacked_sex_summary.csv"

overlap <- read.csv(overlap_path, check.names = FALSE, stringsAsFactors = FALSE)

anti_genes <- c(
  "ABCG1", "ARG1", "ATG5", "BNIP3L", "CAT", "CD36", "EPAS1", "GPX4",
  "HPGD", "HPGDS", "LAMP2", "LIPA", "MSR1", "PTGER4", "RXRA", "SLC40A1",
  "SLC7A11", "SOD2", "SQSTM1", "TSC1", "TXN2", "UCP2", "ULK1"
)

anti_overlap <- overlap[overlap$gene_symbol %in% anti_genes, ]
counts <- table(factor(anti_overlap$sex_overexpressed, levels = c("Female", "Male")))
total <- sum(counts)
pct_female <- 100 * as.numeric(counts["Female"]) / total
pct_male <- 100 * as.numeric(counts["Male"]) / total

summary_tbl <- data.frame(
  class = "Anti-inflammatory / resolving",
  sex_overexpressed = c("Male", "Female"),
  n = c(as.numeric(counts["Male"]), as.numeric(counts["Female"])),
  total = total,
  percentage = c(pct_male, pct_female),
  genes = c(
    paste(sort(anti_overlap$gene_symbol[anti_overlap$sex_overexpressed == "Male"]), collapse = "; "),
    paste(sort(anti_overlap$gene_symbol[anti_overlap$sex_overexpressed == "Female"]), collapse = "; ")
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_tbl, out_summary, row.names = FALSE)

draw_plot <- function() {
  par(family = "sans", mar = c(1.0, 1.0, 3.3, 2.6), xpd = NA)
  plot(
    NA,
    xlim = c(0, 1.55),
    ylim = c(0, 100),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "",
    bty = "n"
  )

  xleft <- 0.45
  xright <- 0.95
  rect(xleft, 0, xright, pct_female, col = "#ED1C24", border = NA)
  rect(xleft, pct_female, xright, 100, col = "#244A9B", border = NA)
  rect(xleft, 0, xright, 100, border = "black", lwd = 1.2)

  text(
    mean(c(xleft, xright)),
    pct_female / 2,
    sprintf("%.1f %%", pct_female),
    col = "white",
    cex = 0.98,
    font = 2
  )
  text(
    mean(c(xleft, xright)),
    pct_female + pct_male / 2,
    sprintf("%.1f %%", pct_male),
    col = "white",
    cex = 0.9,
    font = 2
  )

  text(xright + 0.05, pct_female / 2, "Female", adj = c(0, 0.5), cex = 0.92)
  text(xright + 0.05, pct_female + pct_male / 2, "Male", adj = c(0, 0.5), cex = 0.92)
  mtext("Anti-inflammatory\ngenes", side = 3, line = 0.35, cex = 0.95, font = 2)
}

png(out_png, width = 2.2, height = 3.4, units = "in", res = 600, bg = "white", pointsize = 9)
draw_plot()
dev.off()

pdf(out_pdf, width = 2.2, height = 3.4, bg = "white", pointsize = 9)
draw_plot()
dev.off()

print(summary_tbl)
cat("\nFiles written:\n", out_png, "\n", out_pdf, "\n", out_summary, "\n", sep = "")
