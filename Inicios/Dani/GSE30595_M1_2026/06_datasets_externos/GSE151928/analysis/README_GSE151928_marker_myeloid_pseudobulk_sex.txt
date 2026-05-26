GSE151928 - marker-based myeloid/macrophage pseudobulk por sexo
Fecha: 2026-05-13/14

Objetivo:
Repetir el analisis de sexo de GSE151928 enfocandolo en celulas myeloid/macrophage-like en vez de BAL total.

Metodo:
Como GEO no trae anotacion celular por celula, se hizo una anotacion operacional basada en marcadores.

Marcadores usados:
- alveolar_macrophage: MARCO, FABP4, PPARG, MRC1, MSR1, CD68, CD163, C1QA, C1QB, C1QC, APOC1, APOE, LIPA, MERTK.
- monocyte_like: LYZ, LST1, FCN1, VCAN, S100A8, S100A9, CTSS, TYROBP, FCER1G, AIF1, LGALS3.
- other_lineage/exclusion: CD3D, CD3E, TRAC, NKG7, GNLY, MS4A1, CD79A, FCER1A, CLEC10A, EPCAM, KRT8, KRT18, PECAM1, VWF, COL1A1.

Se retuvieron celulas con score myeloid >= 1 y mayor que el score other_lineage.
Luego se separaron en alveolar_macrophage_like o monocyte_like segun el score dominante.

Resultado de composicion:
BAL esta fuertemente enriquecido en celulas myeloid/macrophage-like: 95.6% a 100% de las celulas por sujeto fueron retenidas. Por eso los resultados de BAL total y marker_myeloid son muy parecidos.

Porcentaje antiinflamatorio/M2 por sexo:
- marker_myeloid: 78 genes detectados; 2 nominales p < 0.05 (2.56%); 1 mas alto en varones y 1 mas alto en mujeres; 0 FDR < 0.10.
- marker_alveolar_macrophage: 75 genes detectados; 2 nominales p < 0.05 (2.67%); 1 mas alto en varones y 1 mas alto en mujeres; 0 FDR < 0.10.
- marker_monocyte_like: 83 genes detectados; 3 nominales p < 0.05 (3.61%); 2 mas altos en varones y 1 mas alto en mujeres; 0 FDR < 0.10.

Genes nominales antiinflamatorios:
- marker_myeloid: CD68 mas alto en male; ALOX15B mas alto en female.
- marker_alveolar_macrophage: CD68 mas alto en male; ALOX15B mas alto en female.
- marker_monocyte_like: CD68 y HPGDS mas altos en male; ALOX15B mas alto en female.

Lectura:
Incluso enfocando el analisis en celulas marker-myeloid/macrophage-like, no aparece un sesgo antiinflamatorio fuerte por sexo en GSE151928. La senal nominal es chica y no sobrevive FDR. Esto puede significar que el dimorfismo de GSE30595 sea especifico de M1 in vitro/estimulado, mientras que BAL sano esta en homeostasis y con alta variabilidad interindividual.

Limitacion:
Esta anotacion es marker-based, no una reproduccion completa de clusters del paper. Para una version definitiva se recomienda correr Seurat/SingleR o recuperar anotaciones originales si estan en suplemento externo.

Archivos principales:
- run_gse151928_marker_myeloid_pseudobulk_sex.R
- GSE151928_marker_based_cell_population_summary_by_subject.csv
- GSE151928_marker_based_cell_scores.tsv
- GSE151928_marker_based_gene_set_overlap_summary.csv
- GSE151928_marker_based_antiinflammatory_percentage_by_sex.tsv
- GSE151928_marker_myeloid_antiinflammatory_general_sex_DE_overlap.csv
- GSE151928_marker_alveolar_macrophage_antiinflammatory_general_sex_DE_overlap.csv
- GSE151928_marker_monocyte_like_antiinflammatory_general_sex_DE_overlap.csv
