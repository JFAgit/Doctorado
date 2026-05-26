import csv
import re
import os

base_path = os.path.expanduser("~/MarceNIS/Variantes")
foldx_path = os.path.expanduser("~/MarceNIS/FoldX")
csv_path = os.path.join(base_path, "gnomAD_NIS_Missense.csv")
aln_path = os.path.join(base_path, "alineamiento_NIS.aln")

# Diccionario de AA 3 a 1 para chequear el PDB
aa_3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

def get_pdb_residues(pdb_file):
    """Retorna un set con los numeros de residuos que realmente existen en el PDB."""
    presentes = {}
    if not os.path.exists(pdb_file): return presentes
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                presentes[res_num] = aa_3to1.get(res_name, 'X')
    return presentes

def parse_clustal(filepath):
    seqs = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("CLUSTAL") and not line.startswith(" "):
                parts = line.split()
                if len(parts) >= 2:
                    name, seq_chunk = parts[0], parts[1]
                    seqs[name] = seqs.get(name, "") + seq_chunk
    return seqs

def build_mapping(hum_seq, rat_seq):
    mapping = {} # hum_pos -> (rat_pos, rat_aa)
    hum_pos, rat_pos = 1, 1
    for i in range(len(hum_seq)):
        h_aa, r_aa = hum_seq[i], rat_seq[i]
        if h_aa != '-':
            if r_aa != '-': mapping[hum_pos] = (rat_pos, r_aa)
            hum_pos += 1
        if r_aa != '-': rat_pos += 1
    return mapping

def get_mut_info(protein_str):
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', str(protein_str))
    if match:
        d = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
        return {'orig': d.get(match.group(1)), 'pos': int(match.group(2)), 'mut': d.get(match.group(3))}
    return None

def main():
    seqs = parse_clustal(aln_path)
    hum_aln, rat_aln = seqs['sp|Q92911|'], seqs['7UUY_1|Cha']
    pos_map = build_mapping(hum_aln, rat_aln)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        mutations = [get_mut_info(row['Protein Consequence']) for row in csv.DictReader(f) if get_mut_info(row.get('Protein Consequence'))]

    pdbs = {
        "AF": os.path.join(foldx_path, "EstructuraAlphaFold/AF-Q92911model_Repair.pdb"),
        "7UUY": os.path.join(foldx_path, "Estructura1/7UUY_Repair.pdb"),
        "7UUZ": os.path.join(foldx_path, "Estructura2/7UUZ_Repair.pdb"),
        "7UV0": os.path.join(foldx_path, "Estructura3/7UV0_Repair.pdb")
    }

    for name, path in pdbs.items():
        pdb_data = get_pdb_residues(path)
        output = os.path.join(base_path, f"individual_list_{name}.txt")
        count = 0
        with open(output, 'w') as f:
            for m in mutations:
                if name == "AF":
                    if m['pos'] in pdb_data:
                        f.write(f"{pdb_data[m['pos']]}A{m['pos']}{m['mut']};\n")
                        count += 1
                else:
                    if m['pos'] in pos_map:
                        r_pos, r_aa = pos_map[m['pos']]
                        r_pos_final = r_pos - 16 # Ajuste de offset PDB
                        if r_pos_final in pdb_data:
                            # IMPORTANTE: Usamos r_aa (el real de la rata)
                            f.write(f"{pdb_data[r_pos_final]}A{r_pos_final}{m['mut']};\n")
                            count += 1
        print(f"Lista para {name}: {count} variantes (filtradas por gaps del PDB).")

if __name__ == "__main__": main()
