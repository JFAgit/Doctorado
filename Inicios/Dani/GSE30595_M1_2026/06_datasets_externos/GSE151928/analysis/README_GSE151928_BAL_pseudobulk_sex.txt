GSE151928 - analisis preliminar BAL total por sexo
Fecha: 2026-05-13/14

Pregunta:
Evaluar si GSE151928 puede analizarse de forma comparable a GSE30595 para genes inmunometabolicos, proinflamatorios y antiinflamatorios/M2.

Datos:
- 15 adultos sanos con BAL scRNA-seq.
- Sexo explicito en GEO: 9 male / 6 female.
- Edad: male media 31.2, female media 32.0.
- Matrices GEO: un archivo UMI_counts.csv.gz por sujeto.
- GEO RAW no trae anotacion celular por celula/cluster; por eso este primer analisis usa BAL total pseudobulk.

Metodo preliminar:
- Se sumaron UMI counts de todas las celulas por sujeto para crear pseudobulk BAL total.
- Se normalizo a logCPM.
- Se comparo male vs female con Welch t-test por gen.
- Se cruzaron los resultados con las mismas listas usadas en GSE30595:
  - inmunometabolismo expandida
  - antiinflammatory_general
  - proinflammatory_general

Resultados rapidos:
- Genes testeados con filtro CPM: 239 genes nominales p < 0.05 en todo el transcriptoma.
- FDR < 0.10 en todo el transcriptoma: solo ZFY-AS1, esperado por sexo cromosomico.
- En listas biologicas:
  - inmunometabolismo expandida: 360 genes detectados, 7 nominales p < 0.05, 0 FDR < 0.10.
  - antiinflammatory_general: 78 genes detectados, 2 nominales p < 0.05, 0 FDR < 0.10.
  - proinflammatory_general: 123 genes detectados, 5 nominales p < 0.05, 0 FDR < 0.10.

Lectura:
Este analisis confirma que GSE151928 sirve para una comparacion por sexo con buen n, pero el analisis biologicamente equivalente a GSE30595 deberia hacerse sobre macrofagos/monocyte-like pseudobulk, no sobre BAL total. Para eso falta anotar o recuperar anotaciones celulares.

Archivos:
- run_gse151928_bal_pseudobulk_sex.R
- GSE151928_BAL_total_sample_qc.csv
- GSE151928_BAL_total_pseudobulk_counts_by_subject.csv
- GSE151928_BAL_total_pseudobulk_logCPM_by_subject.csv
- GSE151928_BAL_total_sex_DE_welch_logCPM.csv
- GSE151928_BAL_total_gene_set_overlap_summary.csv
- GSE151928_BAL_total_inmunometabolismo_expandida_sex_DE_overlap.csv
- GSE151928_BAL_total_antiinflammatory_general_sex_DE_overlap.csv
- GSE151928_BAL_total_proinflammatory_general_sex_DE_overlap.csv

Siguiente paso recomendado:
Reprocesar/anotar celulas de GSE151928 para aislar:
- alveolar macrophages / airspace macrophages
- monocyte-like macrophages / monocytes
- opcionalmente subclusters proinflammatory, metallothionein/metal-binding, cycling myeloid

Luego repetir pseudobulk por sujeto y por poblacion celular.
