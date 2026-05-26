female_deg_path <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Expresion/Criterio clasico/genes_sobreexpresadosM1_F_lfc0.58.csv"
male_deg_path <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Expresion/Criterio clasico/genes_sobreexpresadosM1_M_lfc0.58.csv"

out_list <- "proinflammatory_general_curated_gene_list.csv"
out_overlap <- "M1_proinflammatory_general_DEG_overlap_by_sex.csv"
out_summary <- "M1_proinflammatory_general_stacked_sex_summary.csv"
out_png <- "M1_proinflammatory_general_stacked_sex_vertical.png"
out_pdf <- "M1_proinflammatory_general_stacked_sex_vertical.pdf"

curated <- data.frame(
  gene_symbol = c(
    "AIM2","ALOX5","B2M","BCL2A1","BTK","C1QA","C1QB","C1QC","C2","C3","C5AR1","CASP1",
    "CASP4","CASP5","CCL2","CCL3","CCL4","CCL5","CCL7","CCL8","CCR1","CCR2","CCR5","CD40",
    "CD48","CD74","CD80","CD83","CD86","CLEC4E","COX2","CTSS","CXCL1","CXCL2","CXCL3","CXCL8",
    "CXCL9","CXCL10","CXCL11","CXCR4","CYBB","DUSP2","FCGR1A","FCGR3A","GBP1","GBP2","GBP5",
    "GCH1","HCK","HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DPA1","HLA-DPB1","HLA-DQA1",
    "HLA-DQB1","HLA-DRA","HLA-DRB1","ICAM1","IDO1","IFI16","IFI30","IFIT1","IFIT2","IFIT3",
    "IFNAR1","IFNAR2","IFNB1","IFNG","IFNGR1","IFNGR2","IKBKE","IL12A","IL12B","IL15","IL15RA",
    "IL18","IL18R1","IL1A","IL1B","IL1R1","IL23A","IL6","IRAK1","IRF1","IRF5","IRF7","ISG15",
    "ITGAM","ITGB2","JAK2","KYNU","LST1","LYZ","MMP9","MPO","MYD88","NAMPT","NFKB1","NFKB2",
    "NLRP3","NOS2","OAS1","OAS2","OAS3","OLR1","P2RX7","PDK3","PDK4","PFKP","PGD","PGK1",
    "PLA2G4A","PTGS2","RELA","S100A8","S100A9","STAT1","STAT2","STAT4","TBK1","TFRC","TLR2",
    "TLR4","TLR7","TLR8","TNF","TNFRSF1A","TNFSF10","TRAF6","VCAM1"
  ),
  module = c(
    "inflammasome","leukotriene/eicosanoid inflammation","antigen presentation MHC I","NF-kB survival inflammatory","TLR/BCR inflammatory kinase","complement","complement","complement","complement","complement","complement receptor","inflammasome caspase",
    "noncanonical inflammasome","noncanonical inflammasome","chemokine recruitment","chemokine recruitment","chemokine recruitment","chemokine recruitment","chemokine recruitment","chemokine recruitment","chemokine receptor","chemokine receptor","chemokine receptor","costimulation/TNF receptor",
    "immune activation","MHC II antigen presentation","costimulation","mature APC marker","costimulation","mincle inflammatory CLR","prostaglandin inflammation","lysosomal antigen processing","neutrophil chemokine","neutrophil chemokine","neutrophil chemokine","IL8 neutrophil chemokine",
    "IFNg chemokine","IFNg chemokine","IFNg chemokine","chemokine receptor","oxidative burst","MAPK inflammatory regulator","high affinity Fc receptor","Fc receptor","IFN-inducible GTPase","IFN-inducible GTPase","IFN-inducible GTPase",
    "NO/biopterin inflammatory metabolism","myeloid inflammatory kinase","MHC I","MHC I","MHC I","MHC II processing","MHC II processing","MHC II","MHC II","MHC II",
    "MHC II","MHC II","MHC II","adhesion/inflammation","tryptophan-kynurenine inflammatory axis","DNA sensing inflammasome","antigen processing","IFN-stimulated gene","IFN-stimulated gene","IFN-stimulated gene",
    "type I IFN receptor","type I IFN receptor","type I IFN cytokine","type II IFN cytokine","IFNg receptor","IFNg receptor","TLR glycolytic switch kinase","Th1 cytokine","Th1 cytokine","inflammatory cytokine","IL15 receptor",
    "inflammasome cytokine","IL18 receptor","IL1 cytokine","IL1 cytokine","IL1 receptor","Th17 inflammatory cytokine","inflammatory cytokine","TLR signaling","IFN/NFkB TF","M1 macrophage TF","type I IFN TF","IFN-stimulated ubiquitin-like",
    "CD11b myeloid activation","integrin inflammatory adhesion","JAK/IFN signaling","kynurenine inflammatory metabolism","myeloid inflammatory marker","lysozyme/macrophage activation","matrix inflammation","oxidative burst","TLR adaptor","inflammatory NAD metabolism","NF-kB TF","NF-kB TF",
    "inflammasome sensor","NO synthase M1","IFN-stimulated antiviral","IFN-stimulated antiviral","IFN-stimulated antiviral","oxLDL inflammatory receptor","inflammasome ATP receptor","HIF/glycolytic inflammatory metabolism","HIF/glycolytic inflammatory metabolism","glycolysis inflammatory metabolism","PPP inflammatory redox","glycolysis inflammatory metabolism",
    "arachidonic acid inflammation","COX2 prostaglandin inflammation","NF-kB TF","alarmin","alarmin","IFN/M1 TF","IFN TF","Th1 TF","TLR kinase","iron uptake inflammatory context","TLR",
    "TLR","TLR","TLR","TNF cytokine","TNF receptor","TRAIL inflammatory cytotoxic ligand","TLR/TNF adaptor","adhesion/inflammation"
  ),
  classification_note = "Broad pro-inflammatory/activation gene set; includes cytokines, chemokines, TLR/IFN/NF-kB machinery, antigen presentation, inflammasome, oxidative burst, adhesion and M1 metabolic activation genes.",
  stringsAsFactors = FALSE
)

curated <- curated[!duplicated(curated$gene_symbol), ]
curated$evidence_key <- "canonical_inflammatory_macrophage_M1_literature"
write.csv(curated, out_list, row.names = FALSE)

read_deg <- function(path, sex) {
  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  names(df)[1] <- "gene_symbol"
  df$sex_overexpressed <- sex
  df
}

deg <- rbind(read_deg(female_deg_path, "Female"), read_deg(male_deg_path, "Male"))
overlap <- merge(deg, curated, by = "gene_symbol")
overlap <- overlap[order(overlap$sex_overexpressed, overlap$gene_symbol), ]
write.csv(overlap, out_overlap, row.names = FALSE)

counts <- table(factor(overlap$sex_overexpressed, levels = c("Female", "Male")))
total <- sum(counts)
pct_female <- 100 * as.numeric(counts["Female"]) / total
pct_male <- 100 * as.numeric(counts["Male"]) / total

summary_tbl <- data.frame(
  gene_set = "General pro-inflammatory/activation",
  sex_overexpressed = c("Male", "Female"),
  n = c(as.numeric(counts["Male"]), as.numeric(counts["Female"])),
  total = total,
  percentage = c(pct_male, pct_female),
  genes = c(
    paste(sort(overlap$gene_symbol[overlap$sex_overexpressed == "Male"]), collapse = "; "),
    paste(sort(overlap$gene_symbol[overlap$sex_overexpressed == "Female"]), collapse = "; ")
  ),
  stringsAsFactors = FALSE
)
write.csv(summary_tbl, out_summary, row.names = FALSE)

draw_plot <- function() {
  par(family = "sans", mar = c(0.8, 0.8, 3.1, 2.6), xpd = NA)
  plot(NA, xlim = c(0, 1.55), ylim = c(0, 100), xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n")
  xleft <- 0.45
  xright <- 0.95
  rect(xleft, 0, xright, pct_female, col = "#ED1C24", border = NA)
  rect(xleft, pct_female, xright, 100, col = "#244A9B", border = NA)
  rect(xleft, 0, xright, 100, border = "black", lwd = 1.2)
  text(mean(c(xleft, xright)), pct_female / 2, sprintf("%.1f %%", pct_female), col = "white", cex = 0.98, font = 2)
  text(mean(c(xleft, xright)), pct_female + pct_male / 2, sprintf("%.1f %%", pct_male), col = "white", cex = 0.9, font = 2)
  text(xright + 0.05, pct_female / 2, "Female", adj = c(0, 0.5), cex = 0.92)
  text(xright + 0.05, pct_female + pct_male / 2, "Male", adj = c(0, 0.5), cex = 0.92)
  mtext("Pro-inflammatory\ngenes", side = 3, line = 0.35, cex = 0.95, font = 2)
}

png(out_png, width = 2.2, height = 3.4, units = "in", res = 600, bg = "white", pointsize = 9)
draw_plot()
dev.off()

pdf(out_pdf, width = 2.2, height = 3.4, bg = "white", pointsize = 9)
draw_plot()
dev.off()

cat("Curated genes:", nrow(curated), "\n")
cat("DEG overlap:", nrow(overlap), "\n")
print(summary_tbl)
cat("\nFiles written:\n", out_list, "\n", out_overlap, "\n", out_summary, "\n", out_png, "\n", out_pdf, "\n", sep = "")
