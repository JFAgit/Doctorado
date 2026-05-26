female_deg_path <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Expresion/Criterio clasico/genes_sobreexpresadosM1_F_lfc0.58.csv"
male_deg_path <- "C:/Users/fran_/Documents/Doctorado/Inicios/Dani/Fran/GSE30595 - Microarray 1 color/Expresion/Criterio clasico/genes_sobreexpresadosM1_M_lfc0.58.csv"

out_list <- "antiinflammatory_general_curated_gene_list.csv"
out_overlap <- "M1_antiinflammatory_general_DEG_overlap_by_sex.csv"
out_summary <- "M1_antiinflammatory_general_stacked_sex_summary.csv"
out_png <- "M1_antiinflammatory_general_stacked_sex_vertical.png"
out_pdf <- "M1_antiinflammatory_general_stacked_sex_vertical.pdf"

curated <- data.frame(
  gene_symbol = c(
    "ABCG1","ACKR2","ADORA2A","ALOX15","ALOX15B","ARG1","ARG2","ATF3","ATG5","AXL",
    "BCL3","BNIP3L","CAT","CCL18","CCL22","CD14","CD163","CD200R1","CD274","CD36",
    "CD68","CD93","CHID1","CLEC7A","CX3CR1","DUSP1","DUSP10","EPAS1","FCGR2B","FGL2",
    "GAS6","GILZ","GPX4","HAVCR2","HMOX1","HPGD","HPGDS","IL1R2","IL1RN","IL4R",
    "IL10","IL10RA","IL10RB","IL13RA1","IRAK3","KLF2","KLF4","LAMP2","LILRB1","LILRB2",
    "LILRB4","LIPA","MAF","MAFB","MARCO","MERTK","MRC1","MSR1","NFE2L2","NFKBIA",
    "NR1H3","PPARG","PTGER4","RCAN1","RXRA","SIGLEC10","SLC40A1","SLC7A11","SOCS1","SOCS3",
    "SOD2","SPHK1","SQSTM1","STAB1","STAB2","STAT3","STAT6","TGFB1","TGFB2","TGFB3",
    "TGFBR1","TGFBR2","TNFAIP3","TNIP1","TREM2","TSC1","TXN","TXN2","UCP2","ULK1",
    "VSIG4"
  ),
  module = c(
    "cholesterol efflux","chemokine scavenging","adenosine signaling","pro-resolving lipid mediator","pro-resolving lipid mediator","arginine alternative activation","arginine metabolism","negative inflammatory regulator","autophagy/mitophagy","efferocytosis TAM receptor",
    "NF-kB negative feedback","mitophagy","antioxidant redox","M2/tissue macrophage chemokine","M2/Th2 chemokine","monocyte/macrophage resolution context","M2/scavenger receptor","inhibitory immune receptor","checkpoint inhibitory ligand","efferocytosis/lipid uptake",
    "macrophage marker","efferocytosis","M2 marker","phagocytic receptor","patrolling/resolution receptor","MAPK negative feedback","MAPK negative feedback","HIF2 alternative activation","inhibitory Fc receptor","immunoregulatory mediator",
    "TAM ligand/efferocytosis","glucocorticoid anti-inflammatory mediator","antioxidant/anti-ferroptosis","checkpoint inhibitory receptor","heme/antioxidant resolution","prostaglandin catabolism","PGD2 synthesis/resolution","IL1 decoy receptor","IL1 antagonist","IL4/IL13 alternative activation receptor",
    "anti-inflammatory cytokine","IL10 receptor","IL10 receptor","IL13 receptor","TLR/IL1R negative regulator","anti-inflammatory transcription factor","alternative activation transcription factor","lysosome/autophagy","inhibitory myeloid receptor","inhibitory myeloid receptor",
    "inhibitory myeloid receptor","lysosomal lipid homeostasis","M2 transcription factor","macrophage differentiation/M2 TF","scavenger receptor","efferocytosis TAM receptor","CD206/M2 marker","scavenger receptor","NRF2 antioxidant regulator","NF-kB inhibitor",
    "LXR cholesterol/anti-inflammatory axis","PPARg alternative activation/lipid resolution","EP4 PGE2 anti-inflammatory receptor","calcineurin/NFAT feedback","RXR nuclear receptor","inhibitory/efferocytosis receptor","ferroportin iron export","cystine/glutathione redox","cytokine signaling suppressor","cytokine signaling suppressor",
    "mitochondrial antioxidant","sphingosine signaling/resolution context","autophagy receptor","scavenger/endocytic receptor","scavenger/endocytic receptor","IL10/TGFB signaling TF","IL4/IL13 alternative activation TF","anti-inflammatory cytokine","anti-inflammatory cytokine","anti-inflammatory cytokine",
    "TGFB receptor","TGFB receptor","A20 NF-kB negative regulator","A20-binding NF-kB regulator","lipid sensing/efferocytosis microglia-like macrophage","mTOR/M1 restraint","thioredoxin redox","mitochondrial thioredoxin redox","mitochondrial ROS restraint","autophagy initiation",
    "macrophage complement inhibitory receptor"
  ),
  evidence_key = c(
    "Wang2006_ATVB;YvanCharvet2008_Circulation","ACKR2_decoy_chemokine_resolution","adenosine_A2A_macrophage_antiinflammatory","SPM_lipoxin_resolvin_biosynthesis","SPM_lipoxin_resolvin_biosynthesis","Rath2014_FrontImmunol;M2_marker_literature","arginase_LXR_antiinflammatory_context","ATF3_TLR_negative_feedback","Liu2018_BBA;Cai2022_Autophagy","Grabiec2018_EJI;Cai2018_SciSignal",
    "NFkB_feedback_literature","mitophagy_resolution_literature","NRF2_redox_literature","M2_human_macrophage_marker","M2_human_macrophage_marker","monocyte_macrophage_context_marker","M2_scavenger_receptor_marker","CD200R_inhibitory_myeloid","PDL1_checkpoint_context","Fadok1998_JImmunol;Silverstein2009_SciSignal",
    "macrophage_marker_context","efferocytosis_receptor_literature","M2_marker_literature","dectin1_contextual_resolution","fractalkine_patrolling_resolution","DUSP1_MAPK_feedback","DUSP10_MAPK_feedback","HIF2_M2_context","FCGR2B_inhibitory_receptor","FGL2_immunoregulation",
    "Nepal2019_PNAS","glucocorticoid_TSC22D3_literature","GPX4_redox_ferroptosis_literature","TIM3_inhibitory_receptor","heme_oxygenase_antiinflammatory","eicosanoid_resolution_context","Virtue2015_IJO;Rajakariar2007_PNAS","IL1_decoy_receptor","IL1_receptor_antagonist","IL4_IL13_M2_axis",
    "IL10_canonical","IL10_receptor","IL10_receptor","IL13_M2_axis","IRAK3_negative_regulator","KLF2_antiinflammatory_myeloid","KLF4_M2_polarization","autophagy_lysosome_literature","LILRB_inhibitory_myeloid","LILRB_inhibitory_myeloid",
    "LILRB_inhibitory_myeloid","lysosomal_lipid_homeostasis","MAF_M2_literature","MAFB_macrophage_homeostasis","scavenger_receptor_context","Grabiec2018_EJI;Cai2018_SciSignal","CD206_M2_marker","scavenger_receptor_context","NRF2_redox_literature","NFkB_inhibitor",
    "Thomas2018_CellRep;Marathe2006_JBC","PPARG_alternative_activation","Gill2016_BJP;Heffron2021_JTH","RCAN1_inflammatory_feedback","Thomas2018_CellRep;Marathe2006_JBC","SIGLEC10_inhibitory_efferocytosis","Galli2004_BJH;iron_polarization_literature","glutathione_redox","SOCS1_cytokine_feedback","SOCS3_cytokine_feedback",
    "mitochondrial_redox_literature","SPHK1_resolution_context","autophagy_literature","STAB1_M2_scavenger","STAB2_scavenger_efferocytosis","IL10_TGFB_signaling","Nepal2019_PNAS","TGFB_canonical","TGFB_canonical","TGFB_canonical",
    "TGFB_signaling","TGFB_signaling","Altonsy2014_JBC","TNIP1_A20_pathway","TREM2_lipid_efferocytosis_context","Zhu2014_NatCommun","thioredoxin_redox","thioredoxin_redox","UCP2_ROS_literature","autophagy_literature",
    "VSIG4_macrophage_inhibitory"
  ),
  classification_note = "Broad anti-inflammatory/resolution gene set; includes canonical cytokines/receptors, inhibitory receptors, M2/resolution markers, efferocytosis receptors, negative inflammatory feedback and antioxidant/autophagy homeostasis genes.",
  stringsAsFactors = FALSE
)

curated <- curated[!duplicated(curated$gene_symbol), ]
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
  gene_set = "General anti-inflammatory/resolution",
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
  mtext("Anti-inflammatory\ngenes", side = 3, line = 0.35, cex = 0.95, font = 2)
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
