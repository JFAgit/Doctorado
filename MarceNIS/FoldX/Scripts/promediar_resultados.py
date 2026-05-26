import os
import pandas as pd
import glob

base_path = os.path.expanduser("~/MarceNIS/FoldX")
carpetas = {
    "Estructura1": "7UUY",
    "Estructura2": "7UUZ",
    "Estructura3": "7UV0",
    "EstructuraAlphaFold": "Humano_AF"
}

resultados_finales = []

for folder, nombre in carpetas.items():
    path = os.path.join(base_path, folder)
    # Buscamos el archivo de diferencias (Dif_...)
    files = glob.glob(os.path.join(path, "Dif_*.fxout"))
    
    if not files:
        print(f"⚠️ No se encontró archivo .fxout en {folder}")
        continue
    
    # Leemos el archivo (saltando el header de FoldX)
    df = pd.read_csv(files[0], sep='\t', skipinitialspace=True, skiprows=8)
    
    # El nombre del PDB tiene el formato: AF-Q92911model_Repair_1_0.pdb
    # Queremos agrupar por el ID de la variante (el '1' en este caso)
    def extract_variant_id(pdb_name):
        parts = pdb_name.split('_')
        return parts[-2] # El anteúltimo elemento es el ID de la variante

    df['Variant_ID'] = df['Pdb'].apply(extract_variant_id)
    
    # Agrupamos por Variant_ID y promediamos la columna 'total energy'
    res = df.groupby('Variant_ID')['total energy'].mean().reset_index()
    res.columns = ['ID', f'ddG_{nombre}']
    
    # Guardamos para mergear después
    res['ID'] = res['ID'].astype(int)
    resultados_finales.append(res.set_index('ID'))

# Combinamos todo en una sola tabla
tabla_final = pd.concat(resultados_finales, axis=1).sort_index()

# Guardamos el CSV
output_csv = os.path.join(base_path, "resultados_promediados_NIS.csv")
tabla_final.to_csv(output_csv)

print(f"\n✅ ¡Listo! Se generó el archivo: {output_csv}")
print(tabla_final.head(10))
