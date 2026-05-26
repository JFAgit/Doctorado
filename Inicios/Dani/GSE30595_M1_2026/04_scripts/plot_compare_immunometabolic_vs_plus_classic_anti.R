base_summary_path <- "M1_antiinflammatory_genes_stacked_sex_summary.csv"
plus_summary_path <- "M1_antiinflammatory_general_stacked_sex_summary.csv"

out_png <- "M1_compare_antiinflammatory_immunometabolic_vs_plus_classic.png"
out_pdf <- "M1_compare_antiinflammatory_immunometabolic_vs_plus_classic.pdf"
out_csv <- "M1_compare_antiinflammatory_immunometabolic_vs_plus_classic_summary.csv"

panels <- list(
  list(
    title = "Expanded\nimmunometabolic",
    list_label = "Expanded immunometabolic list\nanti-inflammatory subset",
    list_size = "403 genes screened",
    summary_path = base_summary_path
  ),
  list(
    title = "Expanded +\nclassic immune",
    list_label = "Immunometabolic subset plus\nclassic markers/interleukins",
    list_size = "91 anti-inflammatory genes screened",
    summary_path = plus_summary_path
  )
)

normalise_summary <- function(path, panel_name, list_label, list_size) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"gene_set" %in% names(df) && "class" %in% names(df)) {
    df$gene_set <- df$class
  }
  df$panel_name <- panel_name
  df$list_contrasted <- list_label
  df$list_size_note <- list_size
  df[, c("panel_name", "list_contrasted", "list_size_note", "gene_set", "sex_overexpressed", "n", "total", "percentage", "genes")]
}

combined <- do.call(rbind, lapply(panels, function(panel) {
  normalise_summary(panel$summary_path, panel$title, panel$list_label, panel$list_size)
}))
write.csv(combined, out_csv, row.names = FALSE)

draw_panel <- function(panel, add_left_label = FALSE) {
  df <- read.csv(panel$summary_path, stringsAsFactors = FALSE, check.names = FALSE)
  female <- df[df$sex_overexpressed == "Female", ]
  male <- df[df$sex_overexpressed == "Male", ]
  pct_female <- female$percentage
  pct_male <- male$percentage
  n_total <- female$total

  plot(
    NA,
    xlim = c(0, 1.55),
    ylim = c(-25, 112),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "",
    bty = "n"
  )

  xleft <- 0.46
  xright <- 0.96
  xmid <- mean(c(xleft, xright))
  rect(xleft, 0, xright, pct_female, col = "#ED1C24", border = NA)
  rect(xleft, pct_female, xright, 100, col = "#244A9B", border = NA)
  rect(xleft, 0, xright, 100, border = "black", lwd = 1.2)

  text(xmid, pct_female / 2, sprintf("%.1f %%", pct_female), col = "white", cex = 0.95, font = 2)
  text(xmid, pct_female + pct_male / 2, sprintf("%.1f %%", pct_male), col = "white", cex = 0.9, font = 2)
  text(xright + 0.05, pct_female / 2, sprintf("Female\n%d genes", female$n), adj = c(0, 0.5), cex = 0.75)
  text(xright + 0.05, pct_female + pct_male / 2, sprintf("Male\n%d genes", male$n), adj = c(0, 0.5), cex = 0.75)

  text(xmid, 108, panel$title, cex = 0.95, font = 2)
  text(xmid, -7, sprintf("Total evaluated:\n%d DEG genes", n_total), cex = 0.76, font = 2)
  text(xmid, -18, panel$list_label, cex = 0.62, col = "grey25")
  text(xmid, -24, panel$list_size, cex = 0.58, col = "grey35")

  if (add_left_label) {
    text(-0.08, 50, "Sex-overexpressed\nDEG proportion", srt = 90, cex = 0.74, xpd = NA)
  }
}

draw_figure <- function() {
  layout(matrix(1:2, nrow = 1), widths = c(1, 1))
  par(family = "sans", mar = c(1.0, 0.35, 1.1, 1.0), oma = c(0.5, 0.5, 1.0, 0.5), xpd = NA)
  draw_panel(panels[[1]], add_left_label = TRUE)
  draw_panel(panels[[2]])
  mtext("Anti-inflammatory M1 DEGs: immunometabolic list vs added classic immune genes", outer = TRUE, side = 3, line = -0.15, font = 2, cex = 0.94)
}

png(out_png, width = 5.4, height = 3.8, units = "in", res = 600, bg = "white", pointsize = 9)
draw_figure()
dev.off()

pdf(out_pdf, width = 5.4, height = 3.8, bg = "white", pointsize = 9)
draw_figure()
dev.off()

cat("Files written:\n", out_png, "\n", out_pdf, "\n", out_csv, "\n", sep = "")
