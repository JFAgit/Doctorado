import pandas as pd
import re

# 1. Cargar el reporte unificado
df = pd.read_csv("SUPER_REPORTE_NIS_UNIFICADO.csv")

# 2. Filtrar las que NO tienen ddg (las de Juan que faltan)
# Usamos ddg_7UV0 como referencia, si ese es NaN, le falta la energía
col_ref = 'ddg_7UV0' if 'ddg_7UV0' in df.columns else 'ddG_7UV0'
faltantes = df[df[col_ref].isna() & (df['Fuente'].str.contains('Juan'))].copy()

def format_foldx(variante):
    # De 'G18R' a 'GA18R;' (Formato FoldX: AminoOriginal Posicion AminoMutante ;)
    m = re.search(r'([A-Z])(\d+)([A-Z])', str(variante))
    if m:
        return f"{m.group(1)}A{m.group(2)}{m.group(3)};" # Agregamos la 'A' de la cadena si es necesario
    return None

lista_foldx = faltantes['Variante'].apply(format_foldx).dropna().tolist()

# 3. Guardar el archivo para FoldX
with open("individual_list_Juan_faltantes.txt", "w") as f:
    for mut in lista_foldx:
        f.write(mut + "\n")

print(f"✅ Lista generada con {len(lista_foldx)} variantes.")
