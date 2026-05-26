import pandas as pd
import re
import os

base_path = os.path.expanduser("~/MarceNIS")
csv_original = os.path.join(base_path, "Variantes/variantes_SLC5A5.csv")
csv_estabilidad = os.path.join(base_path, "FoldX/resultados_promediados_NIS.csv")

# Función para identificar mutaciones puntuales válidas
def get_mut_info(protein_str):
    if pd.isna(protein_str): return False
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(protein_str))
    return match is not None

# 1. Cargamos los datos
df_clinico = pd.read_csv(csv_original)
df_foldx = pd.read_csv(csv_estabilidad)

# 2. Rescatamos SOLO las mutaciones que mandamos a FoldX en el orden original
valid_proteins = df_clinico[df_clinico['Protein'].apply(get_mut_info)]['Protein'].tolist()

# 3. Mapeamos el ID de FoldX (1, 2, 3...) con el nombre real de la variante
id_to_protein = {i+1: prot for i, prot in enumerate(valid_proteins)}

# 4. Le asignamos el nombre correcto a los resultados de FoldX
df_foldx['Protein'] = df_foldx['ID'].map(id_to_protein)

# 5. Mergeamos cruzando por la columna 'Protein' (coincidencia exacta)
df_final = pd.merge(df_clinico, df_foldx, on='Protein', how='left')

# Limpiamos y guardamos
if 'ID' in df_final.columns:
    df_final.drop('ID', axis=1, inplace=True)

output_path = os.path.join(base_path, "Reporte_Final_Estabilidad_NIS.csv")
df_final.to_csv(output_path, index=False)

print(f"\n🚀 ¡Reporte final CORREGIDO generado en: {output_path}")
print("\nPrimeras filas de las mutaciones puntuales (con datos de FoldX):")
print(df_final[df_final['Protein'].apply(get_mut_info)][['Protein', 'Result', 'ddG_Humano_AF', 'ddG_7UUY']].head(10))
