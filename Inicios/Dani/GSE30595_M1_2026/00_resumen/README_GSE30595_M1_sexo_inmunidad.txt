GSE30595 - M1 sex-stratified immune/inflammatory analysis
Fecha de organizacion: 2026-05-13

Objetivo general
----------------
Organizar los resultados generados durante el analisis exploratorio de GSE30595,
centrado en muestras M1 y en la clasificacion inferida de sexo.

Clasificacion de sexo usada
---------------------------
La clasificacion de sexo se infirio a partir de marcadores del cromosoma Y en la
matriz M1:

Masculino:
GSM758924, GSM758944, GSM758951, GSM758956

Femenino:
GSM758903, GSM758910, GSM758914, GSM758964, GSM758968

Analisis incluidos
------------------
1. PCA con todos los genes M1.
2. PCA con lista original de inmunometabolismo provista por el usuario.
3. PCA con lista de inmunometabolismo expandida por bibliografia funcional.
4. Cruce de genes inmunometabolicos expandidos contra DEGs M1 por sexo.
5. Figuras apiladas verticales para:
   - genes antiinflamatorios dentro de inmunometabolismo expandido,
   - genes antiinflamatorios generales,
   - genes proinflamatorios generales.
6. Figura combinada con los tres paneles anteriores.

Criterios principales
---------------------
DEGs M1 usados:
- genes_sobreexpresadosM1_F_lfc0.58.csv
- genes_sobreexpresadosM1_M_lfc0.58.csv

En los graficos apilados:
- Azul = genes sobreexpresados en muestras masculinas.
- Rojo = genes sobreexpresados en muestras femeninas.
- El denominador de cada panel es el total de genes DEG que cruzan con la lista
  curada correspondiente.

Resultados clave de los ultimos paneles
---------------------------------------
Anti-inflammatory genes, lista inmunometabolica expandida:
- Total evaluado: 23 DEG genes
- Masculino: 8 genes, 34.8%
- Femenino: 15 genes, 65.2%

Anti-inflammatory genes, lista antiinflamatoria general:
- Total evaluado: 57 DEG genes
- Masculino: 25 genes, 43.9%
- Femenino: 32 genes, 56.1%

Pro-inflammatory genes, lista proinflamatoria general:
- Total evaluado: 75 DEG genes
- Masculino: 56 genes, 74.7%
- Femenino: 19 genes, 25.3%

Estructura de carpetas sugerida
-------------------------------
00_resumen:
README y resumen combinado.

01_figuras:
Figuras PNG/PDF generadas.

02_tablas_y_listas:
Listas curadas, tablas de overlap y summaries.

03_pca:
Resultados de PCA y coordenadas.

04_scripts:
Scripts R reproducibles.

05_logs_bibliografia:
Busquedas y notas de curacion bibliografica.

Notas
-----
La lista antiinflamatoria general y la lista proinflamatoria general son curadas
para exploracion biologica. Conviene revisar/ajustar manualmente genes
contextuales antes de usarlas como definicion final en manuscrito.
