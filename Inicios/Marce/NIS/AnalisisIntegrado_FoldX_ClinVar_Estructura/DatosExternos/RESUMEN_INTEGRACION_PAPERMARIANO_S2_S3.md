# Integracion Paper Mariano/Nicola S2-S3

Se generaron dos propuestas a partir de la tabla final FoldX/ClinVar/estructura.

## Propuesta 1: trazable

Archivo: `PROPUESTA_1_TRAZABLE_CLINVAR_PAPERMARIANO.csv`

Incluye la clasificacion integrada y tambien las columnas separadas de ClinVar, PaperMariano Supplementary Table 2 (ACMG) y PaperMariano Supplementary Table 3 (actividad funcional). Es la tabla para auditar decisiones.

## Propuesta 2: final legible

Archivo: `PROPUESTA_2_FINAL_LEGIBLE_INTEGRADA.csv`

Incluye solamente: variante, clasificacion integrada, fuente de clasificacion, frecuencia alelica, DDG de las cuatro estructuras y clasificacion estructural consenso.

## Regla de integracion

1. Si ClinVar/NCBI/gnomAD ya tenia Pathogenic/Likely pathogenic o Benign/Likely benign, se conserva esa clasificacion.
2. Si ClinVar era incierto/no clasificado/conflictivo y PaperMariano Supplementary Table 2 tenia ACMG informativo, se usa Table 2.
3. Si no habia clasificacion clinica/ACMG informativa y PaperMariano Supplementary Table 3 tenia actividad funcional, se usa la etiqueta funcional.
4. Si nada de lo anterior aplica, queda incierta/no clasificada.

## Conteo por clasificacion integrada

| Clasificacion_integrada | N |
| --- | ---: |
| Uncertain significance / not classified in gnomAD extract | 734 |
| Uncertain significance | 72 |
| Likely Pathogenic | 15 |
| Likely Benign | 10 |
| Functional pathogenic | 8 |
| Functional benign | 6 |
| Pathogenic | 6 |
| Likely benign | 4 |
| Likely pathogenic | 4 |
| Pathogenic/Likely pathogenic | 3 |
| Functional intermediate | 3 |
| Conflicting classifications of pathogenicity | 2 |
| Benign/Likely benign | 2 |
| Benign | 1 |

## Conteo por fuente de clasificacion

| Fuente_clasificacion | N |
| --- | ---: |
| gnomAD extract | 750 |
| NCBI ClinVar | 93 |
| PaperMariano_SupplementaryTable3_FunctionalActivity | 17 |
| PaperMariano_SupplementaryTable2_ACMG | 10 |

Total de variantes unicas: 870