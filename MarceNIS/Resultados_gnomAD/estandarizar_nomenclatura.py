import pandas as pd
import re

# Diccionario de conversión de aminoácidos
aa_map = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': 'X', '*': 'X'
}

def acortar_nombre(nombre_largo):
    if pd.isna(nombre_largo) or nombre_largo == "-":
        return "-"
    
    # Caso para Missense: p.Gly18Arg -> G18R
    # Buscamos el patrón: p.(Amino1)(Posicion)(Amino2)
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*|\=)', str(nombre_largo))
    if match:
        aa1 = aa_map.get(match.group(1), match.group(1))
        pos = match.group(2)
        aa2 = aa_map.get(match.group(3), match.group(3))
        if aa2 == "=": aa2 = aa1 # Sinónima
        return f"{aa1}{pos}{aa2}"
    
    # Caso para cuando ya está en formato corto pero con 'p.' (p.G18R)
    match_corto = re.search(r'p\.([A-Z])(\d+)([A-Z]|\*|\=)', str(nombre_largo))
    if match_corto:
        return f"{match_corto.group(1)}{match_corto.group(2)}{match_corto.group(3)}"

    return nombre_largo # Si es un del, ins o frameshift complejo, lo dejamos igual

# 1. Cargamos el último reporte de Juan
juan_file = '../ResultadosVariantesJuan/REPORTE_JUAN_ESTRUCTURAL_ACTUALIZADO.csv'
df_juan = pd.read_csv(juan_file)

# 2. Creamos la columna estandarizada
# Asumimos que la columna original se llama 'Protein'
col_original = 'Protein' if 'Protein' in df_juan.columns else 'Variante'
df_juan['Variante_Corta'] = df_juan[col_original].apply(acortar_nombre)

# 3. Guardamos
output_name = '../ResultadosVariantesJuan/REPORTE_JUAN_FINAL_NOMENCLATURA.csv'
df_juan.to_csv(output_name, index=False)

print(f"✅ Nomenclatura procesada!")
print(df_juan[[col_original, 'Variante_Corta']].head(10))
