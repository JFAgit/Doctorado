GSE60424 - evaluacion para comparacion por sexo en monocitos/macrofagos humanos
Fecha: 2026-05-19

Fuente:
GEO GSE60424.

Tipo de dataset:
RNA-seq de sangre humana y subsets inmunes purificados.
No es un dataset de macrofagos M0/M1/M2 ni de macrofagos tisulares.

Celulas incluidas:
- Whole Blood: 20 muestras
- Neutrophils: 20 muestras
- Monocytes: 20 muestras
- B-cells: 20 muestras
- CD4: 20 muestras
- CD8: 20 muestras
- NK: 14 muestras

Condiciones/disease status:
- Healthy Control
- MS pretreatment
- MS posttreatment
- Type 1 Diabetes
- Sepsis
- ALS

Sexo:
La metadata trae gender explicito, aunque algunas muestras tienen gender vacio.
Se audito con marcadores X/Y en los normalized counts:
- Entre muestras con sexo metadata no vacio, coincidencia X/Y preliminar: 106/113.
- Las discrepancias se concentran en dos sujetos masculinos (subject 43 sepsis y subject 40 type 1 diabetes), donde la senal Y es baja en varias celulas. Para este dataset conviene usar metadata explicita antes que inferencia por normalized counts.

Monocitos:
Hay 20 muestras de Monocytes:
- Healthy Control: 4 female / 0 male
- MS pretreatment: 3 female / 0 male
- MS posttreatment: 3 female / 0 male
- Type 1 Diabetes: 3 female / 1 male
- Sepsis: 2 female / 1 male

Lectura para nuestro objetivo:
GSE60424 no sirve para comparar sexo en monocitos sanos/control porque Healthy Control Monocytes queda 4 female / 0 male.
Tampoco conviene comparar todos los monocitos male vs female porque los varones aparecen solo en enfermedad (sepsis y type 1 diabetes), entonces sexo queda confundido con disease status.

Conclusion:
Descartar como dataset principal para sexo en macrofagos/monocitos sanos.
Puede quedar como referencia de expresion basal por tipo celular inmune o para verificar si nuestros genes aparecen en monocitos humanos, pero no para diferencial por sexo comparable a GSE30595.

Archivos derivados:
- metadata/GSE60424_sample_metadata_sex_checked.csv
- metadata/GSE60424_sex_marker_normalized_counts.csv
- analysis/GSE60424_sample_counts_by_disease_celltype.csv
- analysis/GSE60424_sample_counts_by_celltype_sex_disease.csv
- analysis/GSE60424_subject_counts_by_sex_disease.csv
- analysis/GSE60424_healthy_control_monocytes_metadata.csv
- analysis/analyze_gse60424_metadata_sex.R
