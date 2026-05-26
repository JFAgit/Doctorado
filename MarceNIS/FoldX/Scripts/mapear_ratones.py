import pandas as pd
import re
import os

# Secuencias completas del Clustal que pasaste
alignment = {
    "Humano_AF":  "MEAVETGERPTFGAWDYGVFALMLLVSTGIGLWVGLARGGQRSAEDFFTGGRRLAALPVGLSLSASFMSAVQVLGVPSEAYRYGLKFLWMCLGQLLNSVLTALLFMPVFYRLGLTSTYEYLEMRFSRAVRLCGTLQYIVATMLYTGIVIYAPALILNQVTGLDIWASLLSTGIICTFYTAVGGMKAVVWTDVFQVVVMLSGFWVVLARGVMLVGGPRQVLTLAQNHSRINLMDFNPDPRSRYTFWTFVVGGTLVWLSMYGVNQAQVQRYVACRTEKQAKLALLINQVGLFLIVSSAACCGIVMFVFYTDCDPLLLGRISAPDQYMPLLVLDIFEDLPGVPGLFLACAYSGTLSTASTSINAMAAVTVEDLIKPRLRSLAPRKLVIISKGLSLIYGSACLTVAALSSLLGGGVLQGSFTVMGVISGPLLGAFILGMFLPACNTPGVLAGLGAGLALSLWVALGATLYPPSEQTMRVLPSSAARCVALSVNASGLLDPALLPANDSSRAPSSGMDASRPALADSFYAISYLYYGALGTLTTVLCGALISCLTGPTKRSTLAPGLLWWDLARQTASVAPKEEVAILDDNLVKGPEELPTGNKKPPGFLPTNEDRLFFLGQKELEGAGSWTPCVGHDGGRDQQETNL",
    "Raton_7UUY": "--------RATFGAWDYGVFATMLLVSTGIGLWQL-------------------AAVPVGLSLAASFMSAVQVLGVPAEAARYGLKFLWMCAGQLLNSLLTAFLFLPIFYRLGLTSTYQYLELRFSRAVRLCGTLQYLVATMLYTGIVIYAPALILNQVTGLDIWASLLSTGIICTLYTTVGGMK----VWTDVFQVVVMLVGFWVILARGVILLGGPRNVLSLAQQHSRINLMDFDPDPRSRYTFWTFIVGGTLVWLSMYGVNQAQVQRYVACHTEGKAKLALLVNQLGLFLIVASAACCGIVMFVYYKDCDPLLTGRISAPDQYMPLLVLDIFEDLPGVPGLFLACAYSGTLSTASTSINAMAAVTVEDLIKPRMPGLAPRKLVFISKGLSFIYGSACLTVAALSSLLGGGVLQGSFTVMGVISGPLLGAFTLGMLLPACNTPGVLSGLAAGLAVSLWVAVGATLYPPGEQTMGVLPTSAAGC-------------------------------GRPALADTFYAISYLYYGALGTLTTMLCGALISYLTGPTKRSSLGPGLLWWD-------------------------------------------",
    "Raton_7UV0": "---------ATFGAWDYGVFATMLLVSTGIGLWVG-------------------LAVPVGLSLAASFMSAVQVLGVPAEAARYGLKFLWMCAGQLLNSLLTAFLFLPIFYRLGLTSTYQYLELRFSRAVRLCGTLQYLVATMLYTGIVIYAPALILNQVTGLDIWASLLSTGIICTLYTTVGGM------TDVFQVVVMLVGFWVILARGVILLGGPRNVLSLAQQHSRINLMDFDPDPRSRYTFWTFIVGGTLVWLSMYGVNQAQVQRYVACHTEGKAKLALLVNQLGLFLIVASAACCGIVMFVYYKDCDPLLTGRISAPDQYMPLLVLDIFEDLPGVPGLFLACAYSGTLSTASTSINAMAAVTVEDLIKPRMPGLAPRKLVFISKGLSFIYGSACLTVAALSSLLGGGVLQGSFTVMGVISGPLLGAFTLGMLLPACNTPGVLSGLAAGLAVSLWVAVGATLYPPGEQTMGVLPTSAAGC-------------------------------GRPALADTFYAISYLYYGALGTLTTMLCGALISYLTGPTKRSSLGPGLLWWD-------------------------------------------",
    "Raton_7UUZ": "---------ATFGAWDYGVFATMLLVSTGIGLWVGLA----------------LAAVPVGLSLAASFMSAVQVLGVPAEAARYGLKFLWMCAGQLLNSLLTAFLFLPIFYRLGLTSTYQYLELRFSRAVRLCGTLQYLVATMLYTGIVIYAPALILNQVTGLDIWASLLSTGIICTLYTTVGGMK----VWTDVFQVVVMLVGFWVILARGVILLGGPRNVLSLAQQHSRINLMDFDPDPRSRYTFWTFIVGGTLVWLSMYGVNQAQVQRYVACHTEGKAKLALLVNQLGLFLIVASAACCGIVMFVYYKDCDPLLTGRISAPDQYMPLLVLDIFEDLPGVPGLFLACAYSGTLSTASTSINAMAAVTVEDLIKPRMPGLAPRKLVFISKGLSFIYGSACLTVAALSSLLGGGVLQGSFTVMGVISGPLLGAFTLGMLLPACNTPGVLSGLAAGLAVSLWVAVGATLYPPGEQTMGVLPTSAAG----------------------------------PALADTFYAISYLYYGALGTLTTMLCGALISYLTGPTKRSSLGPGLLWW--------------------------------------------"
}

def crear_mapeo():
    mapping = []
    # Contadores de aminoácidos reales (según lo que FoldX indexa)
    # 7UUY empieza en la columna 9 del alignment, que es el index 1 del PDB
    counters = {k: 0 for k in alignment.keys()}
    
    for i in range(len(alignment["Humano_AF"])):
        res_h = alignment["Humano_AF"][i]
        res_y = alignment["Raton_7UUY"][i]
        res_v0 = alignment["Raton_7UV0"][i]
        res_uz = alignment["Raton_7UUZ"][i]
        
        if res_h != '-': counters["Humano_AF"] += 1
        if res_y != '-': counters["Raton_7UUY"] += 1
        if res_v0 != '-': counters["Raton_7UV0"] += 1
        if res_uz != '-': counters["Raton_7UUZ"] += 1
        
        mapping.append({
            "h_pos": counters["Humano_AF"] if res_h != '-' else None,
            "7UUY_pos": counters["Raton_7UUY"] if res_y != '-' else None,
            "7UUY_res": res_y,
            "7UV0_pos": counters["Raton_7UV0"] if res_v0 != '-' else None,
            "7UV0_res": res_v0,
            "7UUZ_pos": counters["Raton_7UUZ"] if res_uz != '-' else None,
            "7UUZ_res": res_uz
        })
    return pd.DataFrame(mapping)

df_map = crear_mapeo()
csv_path = os.path.expanduser("~/MarceNIS/Variantes/variantes_SLC5A5.csv")
variantes = pd.read_csv(csv_path)
aa_map = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D', 'Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}

for struct, folder in [("7UUY", "Estructura1"), ("7UV0", "Estructura3"), ("7UUZ", "Estructura2")]:
    out_list = []
    for _, var in variantes.iterrows():
        match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', var['Protein'])
        if match:
            h_pos = int(match.group(2))
            mut_aa = aa_map[match.group(3)]
            
            # Buscar en el mapeo dinámico
            row = df_map[df_map['h_pos'] == h_pos]
            if not row.empty:
                r_pos = row[f"{struct}_pos"].values[0]
                r_res = row[f"{struct}_res"].values[0]
                
                if r_pos and r_res != '-':
                    out_list.append(f"{r_res}A{int(r_pos)}{mut_aa};")
    
    with open(f"/home/faguilera/MarceNIS/FoldX/{folder}/individual_list.txt", 'w') as f:
        f.write("\n".join(out_list))
    print(f"✅ {folder} lista.")
