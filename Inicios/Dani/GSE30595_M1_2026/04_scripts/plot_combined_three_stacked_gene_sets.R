panels <- list(
  list(
    title = "Anti-inflammatory\ngenes",
    list_label = "List: expanded immunometabolic\nbibliography set (403 genes)",
    summary_path = "M1_antiinflammatory_genes_stacked_sex_summary.csv"
  ),
  list(
    title = "Anti-inflammatory\ngenes",
    list_label = "List: general anti-inflammatory\ncurated set (91 genes)",
    summary_path = "M1_antiinflammatory_general_stacked_sex_summary.csv"
  ),
  list(
    title = "Pro-inflammatory\ngenes",
    list_label = "List: general pro-inflammatory\ncurated set (133 genes)",
    summary_path = "M1_proinflammatory_general_stacked_sex_summary.csv"
  )
)

out_png <- "M1_combined_inflammatory_gene_sets_stacked_sex.png"
out_pdf <- "M1_combined_inflammatory_gene_sets_stacked_sex.pdf"
out_csv <- "M1_combined_inflammatory_gene_sets_stacked_sex_summary.csv"

panel_df <- do.call(rbind, lapply(seq_along(panels), function(i) {
  df <- read.csv(panels[[i]]$summary_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"gene_set" %in% names(df) && "class" %in% names(df)) {
    df$gene_set <- df$class
  }
  df <- df[, c("gene_set", "sex_overexpressed", "n", "total", "percentage", "genes")]
  df$panel <- i
  df$panel_title <- panels[[i]]$title
  df$contrast_list <- panels[[i]]$list_label
  df
}))
write.csv(panel_df, out_csv, row.names = FALSE)

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
    ylim = c(-27, 113),
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

  text(xmid, pct_female / 2, sprintf("%.1f %%", pct_female), col = "white", cex = 0.92, font = 2)
  text(xmid, pct_female + pct_male / 2, sprintf("%.1f %%", pct_male), col = "white", cex = 0.86, font = 2)
  text(xright + 0.05, pct_female / 2, sprintf("Female\n%d genes", female$n), adj = c(0, 0.5), cex = 0.72)
  text(xright + 0.05, pct_female + pct_male / 2, sprintf("Male\n%d genes", male$n), adj = c(0, 0.5), cex = 0.72)

  text(xmid, 108, panel$title, cex = 0.92, font = 2)
  text(xmid, -8, sprintf("Total evaluated:\n%d DEG genes", n_total), cex = 0.72, font = 2)
  text(xmid, -22, panel$list_label, cex = 0.58, col = "grey25")

  if (add_left_label) {
    text(-0.08, 50, "Sex-overexpressed\nDEG proportion", srt = 90, cex = 0.72, xpd = NA)
  }
}

draw_figure <- function() {
  layout(matrix(1:3, nrow = 1), widths = c(1, 1, 1))
  par(family = "sans", mar = c(1.1, 0.3, 1.2, 0.95), oma = c(0.5, 0.5, 1.1, 0.5), xpd = NA)
  draw_panel(panels[[1]], add_left_label = TRUE)
  draw_panel(panels[[2]])
  draw_panel(panels[[3]])
  mtext("M1 DEGs classified by sex and inflammatory gene set", outer = TRUE, side = 3, line = -0.2, font = 2, cex = 1.0)
}

png(out_png, width = 7.2, height = 3.8, units = "in", res = 600, bg = "white", pointsize = 9)
draw_figure()
dev.off()

pdf(out_pdf, width = 7.2, height = 3.8, bg = "white", pointsize = 9)
draw_figure()
dev.off()

cat("Files written:\n", out_png, "\n", out_pdf, "\n", out_csv, "\n", sep = "")
