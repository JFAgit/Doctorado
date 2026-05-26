import pandas as pd
import re
import os

# 1. ARCHIVOS DE ENTRADA
# Ajustá 'tabla_variantes_foldx.csv' al nombre real de tu archivo con gnomAD/FoldX
master_file = 'residuos_clasificados.csv' 

archivos_est = {
    '7UUY': 'residuos_clasificados_7uuy.csv',
    '7UUZ': 'residuos_clasificados_7uuz.csv',
    '7UV0': 'residuos_clasificados_7uv0.csv',
    'AF_Hum': 'residuos_clasificados_AF_Human.csv'
}

# 2. FUNCIÓN PARA CARGAR MAPEOS
def crear_mapa_posicion(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ Advertencia: No se encontró {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    mapa = {}
    for _, row in df.iterrows():
        # Extraer solo el número de 'ALA102A' -> 102
        match = re.search(r'\d+', str(row['Residuo']))
        if match:
            mapa[int(match.group())] = row['Categoría']
    return mapa

# Cargamos los 4 mapas
mapas = {id_est: crear_mapa_posicion(path) for id_est, path in archivos_est.items()}

# 3. CARGAR TABLA MAESTRA
df_final = pd.read_csv(master_file)

def extraer_posicion(protein_str):
    # De 'p.Gly18Arg' extrae 18
    if pd.isna(protein_str) or protein_str == "-": return None
    match = re.search(r'\d+', str(protein_str))
    return int(match.group()) if match else None

# 4. INTEGRAR COLUMNAS
for id_est, mapa in mapas.items():
    df_final[f'Cat_{id_est}'] = df_final['Protein'].apply(lambda x: mapa.get(extraer_posicion(x), "-"))

# 5. LÓGICA DE CONSENSO FINAL
def definir_consenso(row):
    # Miramos las categorías de las 4 estructuras
    cats = [row['Cat_7UUY'], row['Cat_7UUZ'], row['Cat_7UV0'], row['Cat_AF_Hum']]
    cats = [c for c in cats if c != "-"]
    
    if not cats: return "-"
    
    # Jerarquía: Sitio activo > Core > Superficie
    if "Sitio activo" in cats:
        return "Sitio activo"
    if "Core" in cats:
        return "Core"
    return "Superficie"

df_final['Categoría_Consenso'] = df_final.apply(definir_consenso, axis=1)

# 6. GUARDAR REPORTE FINAL
output_name = 'REPORTE_MAESTRO_ESTRUCTURAL_NIS.csv'
df_final.to_csv(output_name, index=False)

print(f"\n✅ ¡REPORTE GENERADO! Archivo: {output_name}")
print(df_final['Categoría_Consenso'].value_counts())
