import pandas as pd

# 1. Cargamos las dos tablas
# Ajustá los nombres de las columnas si no se llaman 'Variante' o 'Protein'
# Por esto (con la 'd'):
juan_file = '../ResultadosVariantesJuan/REPORTE_FINAL_PULIDO.csv'
master_file = 'REPORTE_MAESTRO_FINAL_CORREGIDO.csv'

df_juan = pd.read_csv(juan_file)
df_master = pd.read_csv(master_file)

# Limpiamos nombres de columnas por si las moscas
df_juan.columns = df_juan.columns.str.strip()
df_master.columns = df_master.columns.str.strip()

# Identificamos las columnas de las variantes (asumo 'Protein' en Juan y 'Variante' en el tuyo)
col_juan = 'Protein' if 'Protein' in df_juan.columns else 'Variante'
col_master = 'Variante'

vars_juan = set(df_juan[col_juan].dropna().unique())
vars_master = set(df_master[col_master].dropna().unique())

# 2. Análisis de conjuntos
comunes = vars_juan.intersection(vars_master)
faltantes = vars_juan - vars_master
extras = vars_master - vars_juan

print(f"--- RESULTADOS DEL CRUCE ---")
print(f"✅ Variantes de Juan encontradas en la tabla nueva: {len(comunes)}")
print(f"❌ Variantes de Juan que NO están en la tabla nueva: {len(faltantes)}")
print(f"✨ Variantes nuevas (gnomAD) que no estaban en lo de Juan: {len(extras)}")

if faltantes:
    print(f"\n⚠️ Las que faltan son: {faltantes}")
else:
    print("\n🎉 ¡Impecable! Todas las variantes de Juan están adentro de la nueva tabla.")
