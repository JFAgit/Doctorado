import pandas as pd
import os
import re

# --- RUTAS ---
base_path = os.path.expanduser("~/MarceNIS/")
resultados_path = os.path.join(base_path, "Resultados_gnomAD")
analisis_path = os.path.join(base_path, "AnalisisEstructural")

reporte_file = os.path.join(resultados_path, "REPORTE_MAESTRO_GNOMAD.csv")
categorias_file = os.path.join(analisis_path, "residuos_clasificados.csv")

# 1. Cargar el reporte que generamos antes
if not os.path.exists(reporte_file):
    print("Error: No encontré el reporte maestro.")
    exit()
reporte = pd.read_csv(reporte_file)

# 2. Cargar las categorías
if not os.path.exists(categorias_file):
    print("Error: No encontré el archivo de residuos clasificados.")
    exit()
cat_df = pd.read_csv(categorias_file)

# --- PROCESAMIENTO PARA EL CRUCE ---

# Función para sacar solo los números (la posición) de un string
def extraer_posicion(texto):
    match = re.search(r'\d+', str(texto))
    return int(match.group()) if match else None

# Creamos una columna temporal de 'Posicion' en ambos para poder unirlos
reporte['Posicion'] = reporte['Variante'].apply(extraer_posicion)
cat_df['Posicion'] = cat_df['Residuo'].apply(extraer_posicion)

# Si un residuo tiene varias categorías (ej. Core y Sitio de Unión), las juntamos
cat_df_unique = cat_df.groupby('Posicion')['Categoría'].apply(lambda x: ', '.join(set(x))).reset_index()

# Unimos las tablas
reporte_final = pd.merge(reporte, cat_df_unique, on='Posicion', how='left')

# Limpieza final
reporte_final.drop('Posicion', axis=1, inplace=True)
reporte_final['Categoría'] = reporte_final['Categoría'].fillna('Sin Clasificar')

# Guardar
output_file = os.path.join(resultados_path, "REPORTE_MAESTRO_GNOMAD_CATEGORIZADO.csv")
reporte_final.to_csv(output_file, index=False)

print(f"¡Listo! Reporte final con categorías guardado en: {output_file}")
print(reporte_final.head(10))
