import pandas as pd
import sys

# 1. CARGAR LOS ARCHIVOS
juan_file = '../ResultadosVariantesJuan/REPORTE_JUAN_FINAL_NOMENCLATURA.csv'
gnomad_file = 'REPORTE_MAESTRO_FINAL_CORREGIDO.csv'

try:
    df_j = pd.read_csv(juan_file)
    df_g = pd.read_csv(gnomad_file)
except Exception as e:
    print(f"❌ Error al leer archivos: {e}")
    sys.exit()

# 2. ESTANDARIZAR COLUMNAS PARA EL CRUCE
# Limpiamos nombres de columnas (espacios y demas)
df_j.columns = df_j.columns.str.strip()
df_g.columns = df_g.columns.str.strip()

# En Juan usamos 'Variante_Corta', en gnomAD es 'Variante'
if 'Variante_Corta' in df_j.columns:
    df_j = df_j.rename(columns={'Variante_Corta': 'Variante'})

# 3. MARCAR LA FUENTE
df_j['Fuente_Juan'] = True
df_g['Fuente_gnomAD'] = True

# 4. UNIFICAR (Outer Join)
df_unificado = pd.merge(df_g, df_j, on='Variante', how='outer', suffixes=('_gnomAD', '_Juan'))

# 5. CREAR LA COLUMNA DE FUENTE FINAL
def determinar_fuente(row):
    j = row.get('Fuente_Juan') == True
    g = row.get('Fuente_gnomAD') == True
    if j and g: return 'Juan & gnomAD'
    if j: return 'Juan'
    return 'gnomAD'

df_unificado['Fuente'] = df_unificado.apply(determinar_fuente, axis=1)

# 6. REPARAR CONSENSO (si la variante solo estaba en Juan)
if 'Estructura_Consenso' in df_unificado.columns:
    if 'Estructura_Consenso_Actualizada' in df_unificado.columns:
        df_unificado['Estructura_Consenso'] = df_unificado['Estructura_Consenso'].fillna(df_unificado['Estructura_Consenso_Actualizada'])

# 7. ORDENAR COLUMNAS (Fix para el KeyError)
# Buscamos cual de las dos versiones de ddg existe
col_ddg = 'ddg_7UV0' if 'ddg_7UV0' in df_unificado.columns else 'ddG_7UV0'

cols_priority = ['Variante', 'Fuente', 'Estructura_Consenso', 'Pathogenicity']
# Agregamos la de ddg solo si existe
if col_ddg in df_unificado.columns:
    cols_priority.append(col_ddg)

# Filtramos las columnas que realmente existen en el dataframe
existentes = [c for c in cols_priority if c in df_unificado.columns]
otras_cols = [c for c in df_unificado.columns if c not in existentes]

df_unificado = df_unificado[existentes + otras_cols]

# 8. GUARDAR
output_name = 'SUPER_REPORTE_NIS_UNIFICADO.csv'
df_unificado.to_csv(output_name, index=False)

print(f"--- UNIFICACIÓN COMPLETADA ---")
print(df_unificado['Fuente'].value_counts())
print(f"\n✅ Archivo generado: {output_name}")
