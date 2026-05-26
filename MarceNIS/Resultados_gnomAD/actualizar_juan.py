import pandas as pd
import re
import os

# 1. RUTAS DE ARCHIVOS
juan_file = '../ResultadosVariantesJuan/REPORTE_FINAL_PULIDO.csv'
archivos_est = {
    '7UUY': 'residuos_clasificados_7uuy.csv',
    '7UUZ': 'residuos_clasificados_7uuz.csv',
    '7UV0': 'residuos_clasificados_7uv0.csv',
    'AF_Hum': 'residuos_clasificados_AF_Human.csv'
}

# 2. CARGAR MAPAS DE ESTRUCTURA
mapas = {}
for id_est, path in archivos_est.items():
    if os.path.exists(path):
        df_tmp = pd.read_csv(path)
        m = {}
        for _, r in df_tmp.iterrows():
            # Extraer posición: 'ALA102A' -> 102
            pos = re.search(r'\d+', str(r['Residuo']))
            if pos:
                m[int(pos.group())] = r['Categoría']
        mapas[id_est] = m
    else:
        print(f"⚠️ No se encontró: {path}")

# 3. PROCESAR REPORTE DE JUAN
df_juan = pd.read_csv(juan_file)
df_juan.columns = df_juan.columns.str.strip()

# Identificar columna de proteína (ajustar si es 'Protein' o 'Variante')
col_prot = 'Protein' if 'Protein' in df_juan.columns else 'Variante'

def extraer_pos(texto):
    if pd.isna(texto) or texto == "-": return None
    # De 'p.Gly18Arg' o 'G18R' extrae 18
    m = re.search(r'\d+', str(texto))
    return int(m.group()) if m else None

# 4. APLICAR CLASIFICACIÓN
for id_est, mapa in mapas.items():
    df_juan[f'Cat_{id_est}'] = df_juan[col_prot].apply(lambda x: mapa.get(extraer_pos(x), "-"))

# 5. CONSENSO ESTRUCTURAL
def definir_consenso(row):
    cats = [row.get(f'Cat_{e}', "-") for e in archivos_est.keys()]
    cats = [c for c in cats if c != "-" and c is not None]
    
    if not cats: return "-"
    if "Sitio activo" in cats: return "Sitio activo"
    if "Core" in cats: return "Core"
    return "Superficie"

df_juan['Estructura_Consenso_Actualizada'] = df_juan.apply(definir_consenso, axis=1)

# 6. GUARDAR
output_juan = '../ResultadosVariantesJuan/REPORTE_JUAN_ESTRUCTURAL_ACTUALIZADO.csv'
df_juan.to_csv(output_juan, index=False)

print(f"\n✅ ¡Reporte de Juan actualizado! Archivo: {output_juan}")
print(df_juan['Estructura_Consenso_Actualizada'].value_counts())
