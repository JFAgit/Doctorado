import pandas as pd
import os

path = os.path.expanduser("~/MarceNIS/REPORTE_MAESTRO_NIS_FOLDX.csv")
df = pd.read_csv(path)

# 1. Identificamos las columnas de ddG
cols_ddg = ['ddG_7UUY', 'ddG_7UUZ', 'ddG_7UV0', 'ddG_Humano_AF']

# 2. Limpiamos los números: nos aseguramos de que sean floats y redondeamos
for col in cols_ddg:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

# 3. Reemplazamos los NaNs por algo más amigable para Marce
df.fillna("-", inplace=True)

# 4. Guardamos la versión final "pulida"
output = os.path.expanduser("~/MarceNIS/REPORTE_FINAL_PULIDO.csv")
df.to_csv(output, index=False)
print(f"✅ ¡Archivo pulido generado en: {output}")
