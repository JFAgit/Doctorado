Analisis integrado NIS / SLC5A5

Carpetas:

- Tablas: tablas finales y tablas intermedias curadas.
- QC: controles de calidad del mapeo, FoldX y anotaciones.
- Scripts: scripts usados para reconstruir, cruzar y ordenar los datos.
- PyMOL: scripts .pml para visualizar clasificacion estructural y membrana.
- Graficos: histogramas SVG.
- DatosExternos: descargas crudas de ClinVar/UniProt usadas para el cruce.

Archivos principales:

- Tablas/TABLA_FOLDX_NIS_FINAL_CON_INFO_ESTRUCTURAL.csv
  Tabla completa. Primeras columnas: variante, DDG por estructura,
  clasificacion estructural consenso y categorias por estructura.

- Tablas/TABLA_FOLDX_NIS_FINAL_SIMPLE_ESTRUCTURAL.csv
  Version reducida para mirar rapidamente.

- QC/QC_INTEGRACION_INFO_ESTRUCTURAL.csv
  Control de integracion de categorias estructurales por estructura.

- PyMOL/PyMOL_coloreo_estructural/colorear_estructural_7UUY.pml
- PyMOL/PyMOL_coloreo_estructural/colorear_estructural_7UUZ.pml
- PyMOL/PyMOL_coloreo_estructural/colorear_estructural_7UV0.pml
  Abren los PDB con membrana, colorean superficie/core/sitio activo.

- PyMOL/PyMOL_coloreo_estructural/colorear_estructural_AF.pml
  Abre AlphaFold y agrega planos de membrana aproximados.

Colores PyMOL:

- superficie: azul
- core: naranja
- sitio activo: rojo
- membrana/lipidos: gris translucido

Notas:

- La copia cruda grande del cluster queda en Documents/Doctorado/MarceNIS.
- Esta carpeta contiene productos integrados y ordenados para NIS.
