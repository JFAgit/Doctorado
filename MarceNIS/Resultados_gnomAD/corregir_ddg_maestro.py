import pandas as pd
import numpy as np
import os

# 1. Rutas y archivos
reporte_maestro = 'REPORTE_MAESTRO_GNOMAD_ESTRUCTURAL.csv'
carpetas = {
    'ddg_7UUY': '7UUY/Dif_7UUY_Repair.fxout',
    'ddg_7UUZ': '7UUZ/Dif_7UUZ_Repair.fxout',
    'ddg_7UV0': '7UV0/Dif_7UV0_Repair.fxout',
    'ddg_AF': 'AF/Dif_AF-Q92911model_Repair.fxout'
}

def parsear_foldx_dif(path):
    if not os.path.exists(path):
        print(f"⚠️ No se encontró {path}")
        return {}
    
    # Leemos el archivo saltando las 8 líneas de encabezado
    # El delimitador de FoldX suele ser tabulación o múltiples espacios
    df = pd.read_csv(path, sep='\t', skipinitialspace=True, skiprows=8)
    
    # La columna 0 tiene el nombre (ej: 7UV0_Repair_1_0.pdb)
    # La columna 1 tiene el Total Energy (el ddG real)
    # FoldX a veces pega las columnas, así que leemos por posición
    raw_data = []
    with open(path, 'r') as f:
        lines = f.readlines()[9:] # Empezamos en la data
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                name = parts[0]
                ddg = float(parts[1])
                # Sacamos el ID de la variante (el número entre los últimos guiones)
                # ej: 7UV0_Repair_45_0.pdb -> ID 45
                idx = int(name.split('_')[-2])
                raw_data.append({'ID': idx, 'ddG': ddg})
    
    # Promediamos las 3 corridas por cada ID
    return pd.DataFrame(raw_data).groupby('ID')['ddG'].mean().to_dict()

# 2. Cargar el reporte que ya tiene las categorías
df_final = pd.read_csv(reporte_maestro)

# 3. Actualizar las columnas de ddG
for col_name, file_path in carpetas.items():
    print(f"🔄 Procesando {col_name}...")
    mapa_ddg = parsear_foldx_dif(file_path)
    
    # Mapeamos usando el índice de la fila (FoldX arranca en 0 o 1 según la lista)
    # Si tu 'ID' en el reporte coincide con la posición en individual_list:
    if 'ID' in df_final.columns:
        df_final[col_name] = df_final['ID'].map(mapa_ddg)
    else:
        # Si no tenés columna ID, usamos el índice de la fila (empezando en 1)
        df_final[col_name] = (df_final.index + 1).map(mapa_ddg)

# 4. Guardar la versión corregida (LA POSTA)
output_final = 'REPORTE_MAESTRO_FINAL_CORREGIDO.csv'
df_final.to_csv(output_final, index=False)

print(f"\n✅ ¡REPORTAZO CORREGIDO! Archivo generado: {output_final}")
print(df_final[['Variante', 'ddG_7UV0', 'Estructura_Consenso']].head(10))
