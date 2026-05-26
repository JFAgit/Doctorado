import pandas as pd
import re
import os

base_path = os.path.expanduser("~/MarceNIS")
csv_estabilidad = os.path.join(base_path, "Reporte_Final_Estabilidad_NIS.csv")
csv_clasificacion = os.path.join(base_path, "AnalisisEstructural/residuos_clasificados.csv")

# 1. Cargar datos
df_est = pd.read_csv(csv_estabilidad)
df_clasi = pd.read_csv(csv_clasificacion)

# 2. Función para extraer solo el NÚMERO de la posición
def extraer_pos_est(proteina):
    match = re.search(r'p\.[A-Za-z]{3}(\d+)', str(proteina))
    return int(match.group(1)) if match else None

def extraer_pos_clasi(residuo):
    match = re.search(r'\d+', str(residuo))
    return int(match.group(0)) if match else None

# 3. Preparar los DataFrames para el cruce
df_est['Posicion'] = df_est['Protein'].apply(extraer_pos_est)
df_clasi['Posicion'] = df_clasi['Residuo'].apply(extraer_pos_clasi)

# 4. Agrupar categorías (por si un residuo tiene más de una)
df_clasi_grouped = df_clasi.groupby('Posicion')['Categoría'].apply(lambda x: ' / '.join(set(x))).reset_index()

# 5. MERGE FINAL
df_final = pd.merge(df_est, df_clasi_grouped, on='Posicion', how='left')

# Limpiar columnas temporales y guardar
df_final.drop('Posicion', axis=1, inplace=True)
output_path = os.path.join(base_path, "REPORTE_MAESTRO_NIS_FOLDX.csv")
df_final.to_csv(output_path, index=False)

print(f"\n✅ ¡REPORTE MAESTRO GENERADO!")
print(f"Ubicación: {output_path}")
print("\nVista previa del cruce:")
print(df_final[['Protein', 'Result', 'ddG_Humano_AF', 'Categoría']].dropna(subset=['ddG_Humano_AF']).head(10))
