import pandas as pd
import sys
import os

def get_pdb_residues(pdb_file):
    """Extrae los residuos que REALMENTE existen en el PDB (Número y AA)."""
    residuos_existentes = {}
    d31 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    try:
        if not os.path.exists(pdb_file): return {}
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    residuos_existentes[res_num] = d31.get(res_name, 'X')
    except: return {}
    return residuos_existentes

def parse_fasta_alignment(file_path):
    sequences = {}
    current_id = None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                current_id = line.strip()[1:]
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line.strip()
    return sequences

def main():
    if len(sys.argv) < 2: return
    aln = parse_fasta_alignment(sys.argv[1])
    af_key = [k for k in aln.keys() if 'AF' in k or 'Q92911' in k][0]
    
    pdb_files = {
        "7UUY": "../Estructura1/7UUY_Repair.pdb",
        "7UUZ": "../Estructura2/7UUZ_Repair.pdb",
        "7UV0": "../Estructura3/7UV0_Repair.pdb",
        "AF": "../EstructuraAlphaFold/AF-Q92911model_Repair.pdb"
    }

    csv_path = "../../Variantes/gnomAD_v4_UNIFICADO_FINAL.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    for name, seq in aln.items():
        clean_name = "AF" if ("AF" in name or "Q92911" in name) else name.split('_')[0]
        pdb_path = pdb_files.get(clean_name)
        
        # OBTENER RESIDUOS REALES DEL PDB
        pdb_res = get_pdb_residues(pdb_path)
        print(f"\nEstructura {clean_name}: {len(pdb_res)} residuos encontrados en el PDB.")

        output_file = f"individual_list_{clean_name}.txt"
        count = 0
        d31_missense = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}

        with open(output_file, 'w') as f:
            # Mapeo de alineamiento: Humano Pos -> Estructura Pos
            h_pos = 0
            t_pos = 0
            map_h_to_t = {}
            for h_aa, t_aa in zip(aln[af_key], seq):
                if h_aa != '-': h_pos += 1
                if t_aa != '-': t_pos += 1
                if h_aa != '-' and t_aa != '-':
                    map_h_to_t[h_pos] = t_pos

            for _, row in df.iterrows():
                try:
                    pc = str(row['Protein Consequence'])
                    if not pc.startswith('p.'): continue
                    
                    pos_h = int(''.join(filter(str.isdigit, pc)))
                    mut_1 = d31_missense.get(pc[-3:].capitalize(), 'X')
                    
                    # 1. ¿A qué posición de la estructura corresponde?
                    pos_t = map_h_to_t.get(pos_h)
                    
                    # 2. ¿Esa posición existe en el PDB y coincide el Wild Type?
                    # Nota: Para AF el offset es 1, para los PDB de ratón sumamos el inicio real (9)
                    # Pero mejor usamos directamente los números que extrajimos del PDB
                    if clean_name == "AF":
                        real_pos = pos_t
                    else:
                        # Para 7UUY, la pos 1 del alineamiento es la 9 del PDB
                        real_pos = pos_t + 8 if pos_t else None

                    if real_pos in pdb_res:
                        wt_pdb = pdb_res[real_pos]
                        f.write(f"{wt_pdb}A{real_pos}{mut_1};\n")
                        count += 1
                except: continue
        
        print(f" -> {count} variantes válidas para FoldX.")

if __name__ == "__main__":
    main()
