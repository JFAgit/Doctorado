import pandas as pd
import re

def obtener_residuos_pdb(pdb_file):
    residuos = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                # Extraemos el numero de residuo (columnas 23-26)
                res_num = int(line[22:26].strip())
                residuos.add(res_num)
    return residuos

def format_foldx(variante):
    m = re.search(r'([A-Z])(\d+)([A-Z])', str(variante))
    if m: return (int(m.group(2)), f"{m.group(1)}A{m.group(2)}{m.group(3)};")
    return (None, None)

# 1. Cargar reporte
df = pd.read_csv("SUPER_REPORTE_NIS_UNIFICADO.csv")
faltantes = df[df['Fuente'].str.contains('Juan', na=False) & df['ddg_7UV0'].isna()]

pdbs = {
    '7UUY': '7UUY_Repair.pdb',
    '7UUZ': '7UUZ_Repair.pdb',
    '7UV0': '7UV0_Repair.pdb',
    'AF': 'AF-Q92911model_Repair.pdb'
}

for nombre, pdb_path in pdbs.items():
    try:
        res_presentes = obtener_residuos_pdb(pdb_path)
        lista_final = []
        for v in faltantes['Variante']:
            pos, fmt = format_foldx(v)
            if pos in res_presentes:
                lista_final.append(fmt)
        
        with open(f"list_Juan_{nombre}.txt", "w") as f:
            for m in lista_final: f.write(m + "\n")
        print(f"✅ Lista para {nombre} creada con {len(lista_final)} variantes.")
    except:
        print(f"❌ No se pudo procesar {nombre} (¿está el PDB?)")
