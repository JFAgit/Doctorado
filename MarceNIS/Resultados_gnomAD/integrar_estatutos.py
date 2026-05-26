import pandas as pd
import re
import os
import sys

# 1. CONFIGURACIÓN
master_file = 'REPORTE_MAESTRO_GNOMAD.csv'
columna_variante = 'Variante'  # <--- CAMBIADO AQUÍ

archivos_est = {
    '7UUY': 'residuos_clasificados_7uuy.csv',
    '7UUZ': 'residuos_clasificados_7uuz.csv',
    '7UV0': 'residuos_clasificados_7uv0.csv',
    'AF_Hum': 'residuos_clasificados_AF_Human.csv'
}

# 2. CARGAR MAPEOS
mapas = {}
for id_est, path in archivos_est.items():
    if os.path.exists(path):
        df_tmp = pd.read_csv(path)
        m = {}
        for _, r in df_tmp.iterrows():
            pos = re.search(r'\d+', str(r['Residuo']))
            if pos: m[int(pos.group())] = r['Categoría']
        mapas[id_est] = m
    else:
        print(f"⚠️ No se encontró: {path}")

# 3. LEER REPORTE GNOMAD
df_master = pd.read_csv(master_file)
df_master.columns = df_master.columns.str.strip() # Limpiamos espacios por las dudas

if columna_variante not in df_master.columns:
    print(f"❌ ERROR: No existe la columna '{columna_variante}'")
    print(f"Columnas disponibles: {list(df_master.columns)}")
    sys.exit()

def get_pos(prot_str):
    if pd.isna(prot_str) or prot_str == "-": return None
    # Extrae el número de p.Gly18Arg o similar
    m = re.search(r'\d+', str(prot_str))
    return int(m.group()) if m else None

# 4. INTEGRAR COLUMNAS
for id_est, mapa in mapas.items():
    df_master[f'Cat_{id_est}'] = df_master[columna_variante].apply(lambda x: mapa.get(get_pos(x), "-"))

# 5. GENERAR CONSENSO FINAL
def definir_consenso(row):
    # Categorías de las 4 estructuras
    cats = [row.get('Cat_7UUY', "-"), row.get('Cat_7UUZ', "-"), 
            row.get('Cat_7UV0', "-"), row.get('Cat_AF_Hum', "-")]
    cats = [c for c in cats if c != "-" and c is not None]
    
    if not cats: return "-"
    
    if "Sitio activo" in cats: return "Sitio activo"
    if "Core" in cats: return "Core"
    return "Superficie"

df_master['Estructura_Consenso'] = df_master.apply(definir_consenso, axis=1)

# 6. GUARDAR
output_name = 'REPORTE_MAESTRO_GNOMAD_ESTRUCTURAL.csv'
df_master.to_csv(output_name, index=False)

print(f"\n✅ ¡Todo listo, rey! Reporte generado: {output_name}")
print(df_master['Estructura_Consenso'].value_counts())
