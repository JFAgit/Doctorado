import pandas as pd
import re

# Cargamos tus datos
df = pd.read_csv('variantes_SLC5A5.csv')

def get_mut_info(protein_str):
    if pd.isna(protein_str) or 'p.' not in protein_str: return None
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', protein_str)
    if match:
        aa_map = {'Ala':'A', 'Arg':'R', 'Asn':'N', 'Asp':'D', 'Cys':'C', 'Gln':'Q', 'Glu':'E', 'Gly':'G', 'His':'H', 'Ile':'I', 'Leu':'L', 'Lys':'K', 'Met':'M', 'Phe':'F', 'Pro':'P', 'Ser':'S', 'Thr':'T', 'Trp':'W', 'Tyr':'Y', 'Val':'V'}
        return {'orig': aa_map.get(match.group(1)), 'pos': int(match.group(2)), 'mut': aa_map.get(match.group(3))}
    return None

muts = df['Protein'].apply(get_mut_info).dropna().tolist()

# Generar archivos
def write_foldx(filename, offset, min_pos):
    with open(filename, 'w') as f:
        count = 0
        for m in muts:
            new_pos = m['pos'] + offset
            if m['pos'] >= min_pos: # Filtramos las que no están en el PDB
                f.write(f"{m['orig']}A{new_pos}{m['mut']};\n")
                count += 1
    print(f"Creado {filename} con {count} variantes.")

# 1. Humano AF (Offset 0)
write_foldx('individual_list_AF.txt', 0, 1)
# 2. 7UUY (Offset -4, empieza aprox en pos 10 humana)
write_foldx('individual_list_7UUY.txt', -4, 10)
# 3. 7UUZ y 7UV0 (Offset -5, empieza aprox en pos 10 humana)
write_foldx('individual_list_RAT.txt', -5, 10)
