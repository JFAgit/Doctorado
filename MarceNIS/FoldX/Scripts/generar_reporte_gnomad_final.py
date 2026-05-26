import pandas as pd
import os
import re

# --- RUTAS ---
base_path = os.path.expanduser("~/MarceNIS/")
variantes_path = os.path.join(base_path, "Variantes")
resultados_path = os.path.join(base_path, "Resultados_gnomAD")
metadata_gnomad = os.path.join(variantes_path, "gnomAD_NIS_Missense.csv")

# 1. Mapeo de variantes (Limpia el 'AA102S;' -> 'A102S')
def obtener_mapeo_variantes(prefix):
    file_path = os.path.join(variantes_path, f"individual_list_{prefix}.txt")
    with open(file_path, 'r') as f:
        # Quitamos el ';' y convertimos AA102S a A102S
        lineas = [line.strip().replace(';', '') for line in f if line.strip()]
        variantes_limpias = [v[0] + v[2:] for v in lineas]
    return dict(enumerate(variantes_limpias, 1))

# 2. Procesar los .fxout (Ahora sin buscar el .pdb)
def procesar_fxout(struct_name, file_pattern, mapeo):
    file_path = os.path.join(resultados_path, struct_name, f"Average_{file_pattern}_Repair.fxout")
    if not os.path.exists(file_path): 
        print(f"Ojo: No encontré el archivo para {struct_name}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, sep='\t', skiprows=8)
    df.columns = [c.strip() for c in df.columns]
    
    # Buscamos el número después del último guion bajo (ej: _1, _2, etc.)
    df['idx_str'] = df['Pdb'].str.extract(r'_(\d+)$')
    df['idx'] = pd.to_numeric(df['idx_str'], errors='coerce')
    df = df.dropna(subset=['idx'])
    df['idx'] = df['idx'].astype(int)
    
    df['Variante'] = df['idx'].map(mapeo)
    res = df[['Variante', 'total energy']].copy()
    res.columns = ['Variante', f'ddG_{struct_name}']
    
    # Agrupamos y promediamos por si hay repetidos
    return res.groupby('Variante').mean()

# --- EJECUCIÓN ---
print("Levantando datos de FoldX...")
mapeo = obtener_mapeo_variantes("AF")

df_af = procesar_fxout("AF", "AF-Q92911model", mapeo)
df_7uuy = procesar_fxout("7UUY", "7UUY", mapeo)
df_7uuz = procesar_fxout("7UUZ", "7UUZ", mapeo)
df_7uv0 = procesar_fxout("7UV0", "7UV0", mapeo)

# Unimos las 4 tablas
reporte = pd.concat([df_af, df_7uuy, df_7uuz, df_7uv0], axis=1)

# --- CRUCE CON PATHOGENICITY ---
if os.path.exists(metadata_gnomad):
    print("Cruzando con ClinVar...")
    meta = pd.read_csv(metadata_gnomad)
    # Limpiamos p.Ala102Ser a A102S
    d3to1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G','His':'H',
             'Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
    
    def clean_v(v):
        v = str(v).replace('p.', '')
        for k, val in d3to1.items(): v = v.replace(k, val)
        return v

    meta['Variante_ID'] = meta['Protein Consequence'].apply(clean_v)
    meta_sub = meta[['Variante_ID', 'ClinVar Germline Classification']].drop_duplicates('Variante_ID')
    
    reporte = reporte.reset_index()
    reporte = pd.merge(reporte, meta_sub, left_on='Variante', right_on='Variante_ID', how='left')
    reporte.drop('Variante_ID', axis=1, inplace=True)
    reporte.rename(columns={'ClinVar Germline Classification': 'Pathogenicity'}, inplace=True)
    reporte['Pathogenicity'] = reporte['Pathogenicity'].fillna('VUS/Not Classified')

# Guardar
output_file = os.path.join(resultados_path, "REPORTE_MAESTRO_GNOMAD.csv")
reporte.to_csv(output_file, index=False)

print(f"¡Listo! Ahora sí hay datos. Guardado en: {output_file}")
print(reporte.head())
