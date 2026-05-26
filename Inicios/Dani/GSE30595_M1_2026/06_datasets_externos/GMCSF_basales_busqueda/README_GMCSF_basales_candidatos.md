# Busqueda GM-CSF MDM basales humanos

Fecha: 2026-05-26

Objetivo: datasets humanos de macrofagos derivados de monocitos de sangre periferica, diferenciados con GM-CSF, en condicion basal/unstimulated/untreated/control, priorizando n femenino y masculino.

## Candidatos estrictos

| Dataset | Tipo | GM-CSF basal usable | Sexo | Valor para bulk |
|---|---:|---:|---|---|
| GSE160862 + GSE160863 | RNA-seq bulk, MDM GM-CSF, healthy donors, unactivated | 10 donantes, 10 muestras basales | Inferido por X/Y: 6F / 4M | Mejor candidato estricto: n razonable y ambos sexos. |
| GSE224845 | RNA-seq bulk, GM-CSF MDM, UNT/MOCK/SARS-CoV-2, 4 donantes x 3 tiempos UNT | 4 donantes UNT; 12 muestras si se modela tiempo | Inferido por Y: 1F / 3M | Bueno como apoyo; no contar los 12 tiempos como n independiente. |
| GSE232044 | RNA-seq bulk, GM-CSF MDM, JAK inhibitors | 4 donantes untreated | Inferido por Y: 2F / 2M | Muy limpio y balanceado, pero n chico. |
| GSE304218 | RNA-seq bulk, GM-CSF MDM, P. gingivalis infection | 5 controles no infectados | Inferido por X/Y: 0F / 5M | Basal usable, pero solo masculino. |
| GSE102492 | RNA-seq bulk/RPKM, GM-CSF vs M-CSF vs M. obuense MDM | 8 GM-CSF MDM | Inferido por Y: 0F / 8M | N bueno, pero solo masculino. |
| GSE156696 | RNA-seq FPKM, GM-CSF MDM + LXR modulators | 3 DMSO/vehicle | Inferido por Y: 0F / 3M | Vehicle, no basal puro; solo masculino. |
| GSE256208 | RNA-seq FPKM, GM-CSF MDM + CHIR99021 | 3 DMSO/control | Inferido por Y: 0F / 3M | Control usable, pero solo masculino. |
| GSE266236 | RNA-seq FPKM, GM-CSF MDM + siRNA GSK3 | 3 siRNA control | Inferido por Y: 1F / 2M | Control tecnico usable con cautela. |
| GSE188278 | RNA-seq, CD14/M-CSF/GM-CSF differentiation | 3 GM-MO donors | Sexo no explicito; inferible si se baja matriz | Candidato chico. |
| GSE99056 | RNA-seq/microarray reusado en literatura, GM-MO control/LPS | 3 GM-MO control | Sexo no explicito; inferible si se baja matriz | Candidato chico. |
| GSE135491 | RNA-seq, GM-CSF/M-CSF/CM conditions | 3 GM-CSF samples | Sexo no explicito | Candidato chico, diseno menos claro. |
| GSE27792 | Array, GM-CSF vs M-CSF MDM | 3 GM-CSF | Sexo no explicito; inferible por array | Historico, chico. |
| GSE68061 | Array, GM14/GM16 vs M14/M16 | 6 GM-CSF muestras: 3 donantes x 2 subsets | Sexo no explicito; inferible por array | Biologicamente interesante; no sumar subsets como donantes independientes. |
| GSE61298 | Array, GM-CSF/M-CSF +/- estimulos | 3 GM-CSF control | Sexo no explicito; inferible por array | Ya registrado; chico. |
| GSE71253 / GSE64531 | Array, GM-CSF/M-CSF +/- MTX | 3 GM-CSF control por dataset | Sexo no explicito; inferible por array | Cuidado: MTX/vehicle y datasets relacionados. |
| GSE3982 | Array atlas inmune, GM-CSF MDM unstimulated | 2 reps | Sexo no explicito | Muy chico; referencia exploratoria. |

## Candidatos grandes que NO entran en el criterio estricto

| Dataset | Motivo | N/Sexo |
|---|---|---|
| GSE269009 | Tiene muchos macrofagos derivados de monocitos y sexo explicito, pero los MP se diferenciaron con M-CSF. GM-CSF + IL4 se uso para DC, no para macrofagos. | MP total: 109; controles MP: 34F / 23M; casos MP: 25F / 27M. |

## Recomendacion practica

Para bulkear GM-CSF basal estricto, arrancar con `GSE160862 + GSE160863` como nucleo, y sumar `GSE232044` si aceptamos controles untreated de un experimento con inhibidores JAK. Despues se pueden agregar `GSE224845` usando un solo tiempo basal por donante o modelando donor/tiempo, pero no contaria sus 12 UNT como 12 individuos.

Para maximizar n por sexo, el mejor recurso encontrado es `GSE269009`, pero implica cambiar el criterio a M-CSF-derived macrophages.

## Conteo final por sexo

Resumen corto: con criterio **GM-CSF MDM basal estricto y donantes independientes**, el set mas razonable queda en **9F / 9M** si usamos el nucleo `GSE160862 + GSE160863`, sumamos `GSE232044`, y agregamos `GSE224845` contando un solo basal UNT por donante.

| Estrategia | Datasets incluidos | F | M | Total | Comentario |
|---|---|---:|---:|---:|---|
| Nucleo mas limpio | GSE160862 + GSE160863 | 6 | 4 | 10 | Basal/unactivated, GM-CSF MDM, healthy donors; mejor punto de partida. |
| Nucleo + balance | GSE160862 + GSE160863 + GSE232044 | 8 | 6 | 14 | Agrega untreated GM-CSF MDM; queda bastante balanceado. |
| Recomendado estricto | GSE160862 + GSE160863 + GSE232044 + GSE224845, usando 1 UNT por donante | 9 | 9 | 18 | Mejor compromiso: GM-CSF basal, ambos sexos balanceados, sin inflar tiempos. |
| Estricto ampliado con controles tecnicos | Recomendado estricto + GSE266236 siRNA control | 10 | 11 | 21 | Suma 3 controles siRNA; util, pero mas heterogeneo. |
| Todo lo usable con sexo inferido | Recomendado estricto + GSE266236 + GSE304218 + GSE102492 + GSE156696 + GSE256208 | 10 | 30 | 40 | Mucho mas n, pero queda fuertemente masculino y mezcla vehiculos/controles tecnicos. |
| Si se contaran todos los tiempos UNT de GSE224845 | Igual que recomendado, pero usando 12 UNT en vez de 4 donantes | 11 | 15 | 26 | No recomendado como n independiente; usar solo con modelo por donor/tiempo. |

Decision sugerida: para el analisis principal de sexo, usar **18 muestras/donantes: 9F y 9M**. Para sensibilidad, correr una segunda version ampliada con `GSE266236` y reportar que queda **10F / 11M**.

Archivos generados:
- `sex_inference_GMCSF_candidates.tsv`
- `sex_inference_GMCSF_raw_fpkm_candidates.tsv`
- `infer_sex_gmcsf_candidates.R`
- `infer_sex_gmcsf_fpkm_raw.R`
