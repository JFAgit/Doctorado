import pandas as pd
import os
import re

def load_mutation_map(list_path):
    """Mapea el índice (1, 2, 3...) a la (Posición, Aminoácido Mutante)."""
    mapping = {}
    if not os.path.exists(list_path): return {}
    with open(list_path, 'r') as f:
        for i, line in enumerate(f, 1):
            # Limpieza extrema: solo letras y números
            mut_str = line.strip().replace(';', '').replace(' ', '').upper()
            if mut_str:
                # Extraemos la posición y el aminoácido final (ej: RA9W -> 9, W)
                match = re.search(r'(\d+)([A-Z])$', mut_str)
                if match:
                    pos, mut_aa = match.groups()
                    mapping[str(i)] = (int(pos), mut_aa)
    return mapping

def load_foldx_data(file_path, mut_map):
    """Carga los DDG y los indexa por la tupla (Posicion, Mutante)."""
    results = {}
    if not os.path.exists(file_path): return {}
    raw_data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith(("-", "Pdb", "FoldX", "PDB", "Output")) or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # Buscamos el ID del PDB: ej '7UUY_Repair_1_0.pdb' -> indice '1'
                    match = re.search(r'_(\d+)_\d+\.pdb', parts[0])
                    if match:
                        idx = match.group(1)
                        pos_mut = mut_map.get(idx)
                        if pos_mut:
                            try:
                                energy = float(parts[1])
                                if pos_mut not in raw_data: raw_data[pos_mut] = []
                                raw_data[pos_mut].append(energy)
                            except: continue
        # Promediamos las 3 corridas
        return {k: sum(v)/len(v) for k, v in raw_data.items()}
    except: return {}

def main():
    csv_path = "../../Variantes/gnomAD_v4_UNIFICADO_FINAL.csv"
    configs = {
        "7UUY": {"dif": "../Estructura1/Dif_7UUY_Repair.fxout", "list": "individual_list_7UUY.txt"},
        "7UUZ": {"dif": "../Estructura2/Dif_7UUZ_Repair.fxout", "list": "individual_list_7UUZ.txt"},
        "7UV0": {"dif": "../Estructura3/Dif_7UV0_Repair.fxout", "list": "individual_list_7UV0.txt"},
        "AF":   {"dif": "../EstructuraAlphaFold/Dif_AF-Q92911model_Repair.fxout", "list": "individual_list_AF.txt"}
    }

    # 1. Cargar datos de FoldX (indexados por posicion y aminoacido mutante)
    final_data = {}
    for name, paths in configs.items():
        print(f"Cargando {name}...")
        m_map = load_mutation_map(paths['list'])
        final_data[name] = load_foldx_data(paths['dif'], m_map)
        print(f"  -> {len(final_data[name])} variantes listas para unificar.")

    # 2. Diccionario de Aminoácidos
    d31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
           'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
           'TYR':'Y','VAL':'V','TER':'X','STP':'X','*':'X'}

    # 3. Unificar con el CSV
    df = pd.read_csv(csv_path, low_memory=False)
    output = []
    
    for _, row in df.iterrows():
        pc = str(row['Protein Consequence']).upper()
        if not pc.startswith('P.'): continue
        
        # Extraer posición y mutante del nombre 'P.ARG9TRP'
        match = re.search(r'(\d+)([A-Z]+|\*)', pc)
        if not match: continue
        pos_h, mut_3 = match.groups()
        mut_h = d31.get(mut_3, 'X')
        
        res_row = {'Variante': pc, 'ClinVar': row.get('ClinVar Germline Classification', 'ND')}
        
        for name in configs.keys():
            # Buscamos en el diccionario por (Posición, Mutante)
            # Probamos la posición directa y también +/- 1 por si hay desfase
            found_val = "NaN"
            for offset in [0, -1, 1]: # Esto arregla automáticamente los descalces de ratón
                key = (int(pos_h) + offset, mut_h)
                if key in final_data[name]:
                    found_val = round(final_data[name][key], 4)
                    break
            res_row[f"DDG_{name}"] = found_val
            
        output.append(res_row)

    pd.DataFrame(output).to_csv("TABLA_MAESTRA_FOLDX_NIS.csv", index=False)
    print("\n¡Listo! El archivo se actualizó. Ahora tiene que tener muchísimos menos NaNs.")

if __name__ == "__main__":
    main()
