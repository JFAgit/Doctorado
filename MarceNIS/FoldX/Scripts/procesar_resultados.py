import pandas as pd
import os
import glob
import re

# Rutas de las carpetas
base_path = os.path.expanduser("~/MarceNIS/FoldX")
output_path = os.path.join(base_path, "Resultados_Finales")
folders = {
    "EstructuraAlphaFold": "Humano_AF",
    "Estructura1": "Raton_7UUY",
    "Estructura2": "Raton_7UUZ",
    "Estructura3": "Raton_7UV0"
}

all_results = []

print("Iniciando recolección de datos...")

for folder, name in folders.items():
    folder_path = os.path.join(base_path, folder)
    # Buscamos el archivo Raw_differences que contiene los promedios
    files = glob.glob(os.path.join(folder_path, "Raw_differences*.fxout"))
    
    if files:
        # FoldX usa tabulaciones. Leemos y quitamos espacios en blanco de los nombres de columnas
        df = pd.read_csv(files[0], sep='\t')
        df.columns = df.columns.str.strip()
        
        # Seleccionamos la columna del nombre de la mutante y la de energía total (ddG)
        # Usamos iloc por si el header tiene caracteres raros, FoldX suele poner ddG en la col 1
        temp_df = df.iloc[:, [0, 1]].copy()
        temp_df.columns = ['Mutant', f'ddG_{name}']
        
        # Limpiamos el nombre: de "7UUY_Repair_G18R_1" a solo "G18R"
        # Esta regex busca una letra, números y otra letra (formato AA-Pos-AA)
        temp_df['Mutant'] = temp_df['Mutant'].astype(str).str.extract(r'([A-Z]\d+[A-Z])')
        
        all_results.append(temp_df)
        print(f"✅ Cargados resultados de: {name}")
    else:
        print(f"⚠️ No se encontraron archivos en {folder}")

# Combinar todo
if all_results:
    final_df = all_results[0]
    for next_df in all_results[1:]:
        final_df = pd.merge(final_df, next_df, on='Mutant', how='outer')
    
    # Calcular un promedio de ratón para comparar fácil contra humano
    raton_cols = [c for c in final_df.columns if 'Raton' in c]
    if raton_cols:
        final_df['Promedio_Raton'] = final_df[raton_cols].mean(axis=1)
    
    # Guardar el CSV final
    output_file = os.path.join(output_path, "comparativa_ddG_NIS.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n🚀 PROCESO COMPLETADO")
    print(f"Archivo guardado en: {output_file}")
    print("\nPrimeras filas del resultado:")
    print(final_df.head())
else:
    print("\n❌ Error: No se pudo procesar ningún archivo. ¿Ya terminó la corrida?")
