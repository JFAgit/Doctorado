Analisis GM-CSF basal vs IL4/13 y hormonas puras en GSE30595

Diseno usado:
- Donantes pareados completos con GM-CSF basal: M24 y M27.
- Condiciones principales: GM-CSF, GM-CSF+IL4/13, GM-CSF+E, GM-CSF+P.
- Combo separado: GM-CSF+E/P/IL-10/4/13, disponible solo para M27 en este subconjunto; no se usa como hormona pura.

Notas metodologicas:
- Se uso gProcessedSignal de Agilent.
- Se transformo como log2(signal + 1).
- Si habia mas de una sonda por gen, se promedio la expresion log2 por gen.
- La comparacion principal es descriptiva y pareada contra GM-CSF basal, no DEG formal, porque n=2 donantes.

Genes del panel solicitados: 67
Genes presentes en la plataforma/datos: 44

Archivos principales:
- *_heatmap_expr_zscore_sin_combo.png/pdf
- *_heatmap_delta_log2_vs_GMCSF_sin_combo.png/pdf
- *_scores_M2_pareado.png/pdf
- *_deltas_log2_vs_GMCSF_por_donante.csv
- *_resumen_delta_log2_vs_GMCSF_por_gen.csv
- *_scores_M2_delta_vs_GMCSF_resumen.csv
