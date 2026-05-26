import sys
import os

def pdb_to_fasta(pdb_file):
    # Mapa de aminoácidos
    aa_map = {
        'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
        'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
        'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
    }
    
    if not os.path.exists(pdb_file):
        return None

    # Usamos el nombre del archivo como etiqueta (label)
    label = os.path.basename(pdb_file).replace('.pdb', '')
    seq, last = [], None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_num = line[22:26].strip()
                res_name = line[17:20].strip()
                if res_num != last:
                    seq.append(aa_map.get(res_name, 'X'))
                    last = res_num
    
    return f'>{label}\n' + ''.join(seq) + '\n'

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 extraer_fasta.py archivo.pdb")
    else:
        pdb_path = sys.argv[1]
        resultado = pdb_to_fasta(pdb_path)
        if resultado:
            print(resultado, end='')
