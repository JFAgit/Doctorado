Datasets externos para comparacion de sexo en macrofagos humanos
Fecha: 2026-05-13

Objetivo:
Buscar datasets humanos de macrofagos M0/M1/M2 o estados equivalentes, sin enfermedad en las muestras de origen, y determinar sexo de donante con metadata y/o genes cromosomicos para compararlos luego con las listas inmunometabolicas, antiinflamatorias y proinflamatorias usadas en GSE30595.

Dataset 1: E-MTAB-7572
Fuente: ArrayExpress/BioStudies.
Archivos bajados:
- processed/entrez_counts.csv
- metadata/E-MTAB-7572.sdrf.txt
- metadata/E-MTAB-7572.idf.txt
- metadata/E-MTAB-7572_biostudies_api.json

Nota sobre raw:
El SDRF indica que los FASTQ crudos estan en EGA/EGAS00001003451, con acceso controlado por privacidad. Por eso, en esta carpeta se guardaron los raw read counts publicos (entrez_counts.csv) y la metadata publica.

Sexo:
El SDRF trae Characteristics[sex]. Se audito con genes cromosomicos en logCPM.
- d54: male
- d55: female
- d57: male
La inferencia por marcadores coincide con metadata en 24/24 muestras.

Condiciones utiles:
- none: M0/macrophage
- LPSIFNg, IFNg, LPSearly, LPSlate: estados inflamatorios/M1-like
- IL4, IL10, dex: estados alternativos/antiinflamatorios/M2-like o regulatorios

Archivos derivados:
- metadata/E-MTAB-7572_sample_metadata_sex_checked.csv
- metadata/E-MTAB-7572_sex_marker_logCPM.csv

Dataset 2: GSE35449
Fuente: GEO.
Archivos bajados:
- raw/GSE35449_RAW.tar
- processed/GSE35449_non-normalized.txt.gz
- metadata/GSE35449_series_matrix.txt.gz
- metadata/GPL6947_HumanHT-12_V3_0_R1_11283641_A.bgx.gz

Nota sobre raw:
El RAW.tar de GEO contiene el archivo de anotacion Illumina BGX. Las intensidades por muestra estan disponibles en la series matrix y en el archivo non-normalized de GEO.

Sexo:
GEO no trae sexo explicito. Se infirio usando marcadores Y dinamicos en la plataforma Illumina y XIST como apoyo. Marcadores revisados: XIST, DDX3Y, EIF1AY, KDM5D, RPS4Y1, RPS4Y2, TMSB4Y, USP9Y, UTY, ZFY, entre otros.

Inferencia por donante:
- Donor 1: male
- Donor 2: male
- Donor 3: male
- Donor 4: female
- Donor 5: male
- Donor 6: male
- Donor 7: male

La clasificacion de GSE35449 debe usarse como inferida, no como metadata declarada. Donor 4 es el mas claro por XIST alto y Y bajo; el resto muestra expresion de marcadores Y, especialmente RPS4Y1/2 y EIF1AY.

Condiciones utiles:
- M0 macrophages
- M1 macrophages
- M2 macrophages

Archivos derivados:
- metadata/GSE35449_sample_metadata_sex_inferred.csv
- metadata/GSE35449_donor_sex_inferred.csv
- metadata/GSE35449_sex_marker_probe_expression.csv

Script reproducible:
- infer_sex_chromosomal_markers.R

Siguiente paso sugerido:
Antes de hacer comparaciones biologicas, revisar balance:
- E-MTAB-7572: 2 hombres, 1 mujer, pareado por condiciones.
- GSE35449: 6 hombres, 1 mujer inferidos, pareado M0/M1/M2 por donante.

E-MTAB-7572 parece mejor para sexo por metadata explicita, aunque tiene pocos donantes. GSE35449 es biologicamente muy bueno para M0/M1/M2, pero muy desbalanceado por sexo si la inferencia se confirma.

Dataset 3: GSE18686
Fuente: GEO.
Archivos bajados:
- raw/GSE18686_RAW.tar
- processed/GSE18686_nonnormalized.txt.gz
- metadata/GSE18686_series_matrix.txt.gz
- metadata/GPL6947_HumanHT-12_V3_0_R1_11283641_A.bgx.gz

Nota sobre uso:
El dataset incluye cultivos de macrofagos derivados de buffy coats normales y biopsias de psoriasis. Para el objetivo actual se filtraron solo las muestras "Macrophages culture"; las biopsias de piel psoriasis quedaron fuera.

Condiciones utiles:
- control: M0-like
- IFNg: M1-like parcial
- LPS and IFNg: M1-like fuerte
- IL4: M2-like
- LPS, TNFa e IL17: condiciones inflamatorias adicionales

Sexo:
GEO no trae sexo explicito. Se infirio por marcadores cromosomicos en la plataforma Illumina, igual que GSE35449.

Inferencia por donante de cultivos macrofagicos:
- ID 1: female
- ID 2: female
- ID 3: female
- ID 4: female, con replica tecnica de todas las condiciones
- ID 5: male
- ID 6: male, sin muestra control

Balance:
- Por donante unico: 4 female / 2 male.
- Para IFNg, IL4, LPS, LPS+IFNg, TNFa e IL17: 4 donantes female y 2 male, con replica tecnica extra para ID 4.
- Para control: 4 donantes female y 1 male, porque ID 6 no tiene control.

Archivos derivados:
- metadata/GSE18686_sample_metadata_sex_inferred.csv
- metadata/GSE18686_donor_sex_inferred.csv
- metadata/GSE18686_sex_marker_probe_expression.csv

Lectura:
GSE18686 mejora el balance respecto de GSE35449, aunque sigue desbalanceado hacia femenino. Es util como dataset complementario para comparar IL4/M2-like vs LPS+IFNg/M1-like por sexo, pero el control/M0 queda flojo en masculino.

Busqueda extendida 2026-05-13/14: tejidos sanos/control, placenta/decidua/Hofbauer y MDM adicionales

Criterio ampliado:
Se aceptan macrofagos humanos M0/M1/M2 in vitro, macrofagos tisulares sanos/control y placenta/decidua/Hofbauer si el sexo es explicito o inferible por genes cromosomicos. Se excluyen datasets no humanos y se prioriza disponibilidad de matriz procesada o metadata clara.

Archivos de busqueda guardados:
- _busqueda_candidatos/candidatos_macrofagos_humanos_sexo_2026-05-13.tsv
- _busqueda_candidatos/GSE151928_series_matrix.txt.gz
- _busqueda_candidatos/GSE151928_SraRunInfo.csv
- _busqueda_candidatos/GSE174689_series_matrix.txt.gz
- _busqueda_candidatos/GSE174689_SraRunInfo.csv
- _busqueda_candidatos/GSE228087_series_matrix.txt.gz
- _busqueda_candidatos/GSE228087_SraRunInfo.csv
- _busqueda_candidatos/GSE61298_series_matrix.txt.gz
- _busqueda_candidatos/GSE199378_series_matrix.txt.gz
- _busqueda_candidatos/GSE199378_SraRunInfo.csv
- _busqueda_candidatos/GSE71253_series_matrix.txt.gz
- _busqueda_candidatos/GSE124350_series_matrix.txt.gz
- _busqueda_candidatos/GSE124350_SraRunInfo.csv
- _busqueda_candidatos/GSE228087_MoMF_counts.csv.gz
- _busqueda_candidatos/GSE228087_MoMF_tpm.csv.gz
- _busqueda_candidatos/GSE228087_sample_metadata_sex_inferred.csv
- _busqueda_candidatos/GSE228087_sex_marker_tpm.csv
- _busqueda_candidatos/GSE174689_All_logTPMs_exprTable.txt.gz
- _busqueda_candidatos/GSE174689_sample_metadata_sex_inferred.csv
- _busqueda_candidatos/GSE174689_sex_marker_logTPM.csv
- _busqueda_candidatos/infer_sex_candidate_rnaseq.R

Candidatos prioritarios:

1) GSE151928 - bronchoalveolar lavage sano, scRNA-seq
Fuente: GEO/BioProject PRJNA637757.
Resumen: 15 muestras de BAL de adultos sanos; el estudio compara programas de macrofagos/monocitos entre sexos.
Sexo: explicito en GEO series matrix.
- female: Subject 1, 5, 7, 8, 13, 14
- male: Subject 2, 3, 4, 6, 9, 10, 11, 12, 15
Balance: 9 male / 6 female.
Lectura: es el candidato tisular sano/control mas fuerte encontrado. No es bulk M0/M1/M2; requiere procesar scRNA-seq o usar matrices por muestra para hacer pseudobulk de macrofagos/monocitos.

2) GSE228087 - monocyte-derived macrophages, RNA-seq
Fuente: GEO/BioProject PRJNA948022.
Resumen: monocytes de donantes sanos diferenciados con M-CSF y polarizados con IL4+IL13 o LPS+IFNg; processed counts/TPM disponibles en GEO.
Sexo: no explicito en GEO ni SRA RunInfo. Inferencia preliminar por XIST/Y en TPM:
- donor_from_title 1: female
- donor_from_title 2: female
- donor_from_title 3: male
- donor_from_title 4: female
- donor_from_title 5: female, con una muestra de monocitos con Y intermedio pero XIST alto
Condiciones utiles: M-CSF unstim, M-CSF+IL4+IL13, M-CSF+LPS+IFNg.
Lectura: biologicamente muy cercano al objetivo M0/M1/M2. Para agrupar, usar donor_from_title extraido del titulo de muestra; el campo "donor id" de GEO no coincide de forma intuitiva con los titulos.
Archivos derivados:
- _busqueda_candidatos/GSE228087_sample_metadata_sex_inferred.csv
- _busqueda_candidatos/GSE228087_sex_marker_tpm.csv

3) GSE174689 - Hofbauer cells humanos, RNA-seq
Fuente: GEO/BioProject PRJNA731125.
Resumen: fetal placental macrophages/Hofbauer cells de placentas humanas a termino; controles no tratados, LPS+IFNg y L. monocytogenes.
Sexo: no explicito en GEO ni SRA RunInfo. La inferencia preliminar por XIST/Y en logTPM no separa de manera limpia: varios marcadores Y tienen expresion detectable en todas las muestras y el rango de XIST es bajo. No usar todavia para comparacion por sexo sin revisar raw, paper/suplementos o una matriz alternativa.
Condiciones utiles: controles no tratados 5 h/24 h y LPS+IFNg; las infecciones conviene excluirlas o analizarlas aparte.
Lectura: interesante para eje placenta/Hofbauer y estado antiinflamatorio/M2-like, pero con n pequeno y sexo incierto.
Archivos derivados:
- _busqueda_candidatos/GSE174689_sample_metadata_sex_inferred.csv
- _busqueda_candidatos/GSE174689_sex_marker_logTPM.csv

4) GSE61298 - monocyte-derived macrophages, array
Fuente: GEO.
Resumen: 3 donantes sanos; GM-CSF/M-CSF durante 7 dias y activacion con LPS+IFNg, IL4 o IL10.
Sexo: no explicito en GEO; inferible por marcadores cromosomicos si la plataforma conserva sondas X/Y utiles.
Lectura: excelente panel de polarizacion, pero solo 3 donantes.

Otros candidatos:
- GSE199378: scRNA-seq de MDM M1, M2a y M2c de 2 donantes sanos; no trae sexo explicito; util para firmas, menos fuerte para comparacion por sexo.
- GSE71253: MDM GM-CSF/M-CSF con/sin methotrexate, 3 donantes; no trae sexo explicito; util pero menos directo que LPS/IL4.
- GSE124350: macrofagos colonicos humanos CD206+ y CD206-; no trae sexo explicito en GEO/SRA; antes de usar hay que separar controles/no inflamados de muestras IBD.

Siguiente paso sugerido:
Priorizar GSE151928 para un analisis tisular sano con sexo explicito, haciendo pseudobulk de macrofagos/monocitos por sujeto. Como segundo dataset, usar GSE228087 para MDM M0/M1/M2-like con sexo inferido tentativo. GSE174689 queda como biologicamente interesante para Hofbauer, pero no prioritario para sexo hasta resolver la senal X/Y.

Dataset evaluado 2026-05-19: GSE60424
Fuente: GEO.
Tipo: RNA-seq de sangre humana y subsets inmunes purificados: Whole Blood, Neutrophils, Monocytes, B-cells, CD4, CD8 y NK. No es macrofagos M0/M1/M2 ni macrofagos tisulares.

Sexo:
La metadata trae gender explicito. Se audito de forma preliminar con marcadores X/Y en normalized counts; entre muestras con gender no vacio, la coincidencia fue 106/113. Las discrepancias se concentran en dos sujetos masculinos enfermos, probablemente por senal Y baja en normalized counts. Para este dataset conviene usar metadata explicita.

Monocitos:
- Monocytes totales: 20 muestras.
- Healthy Control Monocytes: 4 muestras, todas female.
- Todos los Monocytes: 18 female / 2 male, pero los varones estan en sepsis y type 1 diabetes.

Lectura:
GSE60424 no sirve para comparar sexo en monocitos sanos/control porque no hay varones en Healthy Control Monocytes. Tampoco conviene usar todos los monocitos para sexo porque sexo queda confundido con disease status. Queda como referencia de expresion en monocitos humanos, no como dataset principal para diferencial por sexo comparable a GSE30595.

Archivos:
- GSE60424/analysis/README_GSE60424_sexo_monocitos.txt
- GSE60424/metadata/GSE60424_sample_metadata_sex_checked.csv
- GSE60424/analysis/GSE60424_sample_counts_by_celltype_sex_disease.csv
