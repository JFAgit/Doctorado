from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\MarceNIS")
FOLDX = BASE / "FoldX"
SCRIPTS = FOLDX / "Scripts"
VARIANTS_CSV = BASE / "Variantes" / "gnomAD_v4_UNIFICADO_FINAL.csv"
ALIGNMENT = SCRIPTS / "alineamiento_NIS.aln"

OUT_DETALLE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\QC_MAPEO_FOLDX_VS_ALINEAMIENTO_DETALLE.csv")
OUT_RESUMEN = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\QC_MAPEO_FOLDX_VS_ALINEAMIENTO_RESUMEN.csv")


STRUCTURES = {
    "7UUY": {
        "prefix": "7UUY",
        "pdb": FOLDX / "Estructura1" / "7UUY_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UUY.txt",
    },
    "7UUZ": {
        "prefix": "7UUZ",
        "pdb": FOLDX / "Estructura2" / "7UUZ_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UUZ.txt",
    },
    "7UV0": {
        "prefix": "7UV0",
        "pdb": FOLDX / "Estructura3" / "7UV0_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UV0.txt",
    },
    "AF": {
        "prefix": "AF",
        "pdb": FOLDX / "EstructuraAlphaFold" / "AF-Q92911model_Repair.pdb",
        "list": SCRIPTS / "individual_list_AF.txt",
    },
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_fasta_alignment(path):
    seqs = {}
    key = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                key = line[1:]
                seqs[key] = ""
            elif key is not None:
                seqs[key] += line
    return seqs


def find_key(seqs, structure, prefix):
    if structure == "AF":
        for key in seqs:
            if "AF" in key or "Q92911" in key:
                return key
    for key in seqs:
        if key.startswith(prefix) or key.split("_")[0] == prefix:
            return key
    raise KeyError(f"No encontre secuencia para {structure}")


def build_alignment_map(ref_seq, target_seq, structure):
    h_pos = 0
    t_pos = 0
    rows = {}
    for aln_i, (h_aa, t_aa) in enumerate(zip(ref_seq, target_seq), start=1):
        if h_aa != "-":
            h_pos += 1
        if t_aa != "-":
            t_pos += 1
        if h_aa != "-" and t_aa != "-":
            pdb_pos = t_pos if structure == "AF" else t_pos + 8
            rows[h_pos] = {
                "Alignment_Column": aln_i,
                "Human_AA_Alignment": h_aa,
                "Target_AA_Alignment": t_aa,
                "Target_Seq_Pos_NoOffset": t_pos,
                "Expected_PDB_Pos": pdb_pos,
            }
    return rows


def pdb_ca_residues(path):
    residues = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    res_name = line[17:20].strip().upper()
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue
                residues[res_num] = AA3_TO_1.get(res_name, "X")
    return residues


def read_list(path):
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            item = raw.strip().replace(" ", "").rstrip(";")
            if item:
                out.append(item)
    return out


def parse_foldx_input(value):
    match = re.match(r"^([A-Z])A(\d+)([A-Z])$", value)
    if not match:
        return None
    wt, pos, mut = match.groups()
    return wt, int(pos), mut


def parse_variant(pc):
    pc = "" if pd.isna(pc) else str(pc).strip()
    if not pc.startswith("p."):
        return None
    digits = "".join(ch for ch in pc if ch.isdigit())
    if not digits:
        return None
    mut3 = pc[-3:].upper()
    return {
        "Variant": pc,
        "Human_Pos": int(digits),
        "Mut_1": AA3_TO_1.get(mut3, "X"),
    }


def main():
    seqs = parse_fasta_alignment(ALIGNMENT)
    ref_key = find_key(seqs, "AF", "AF")
    ref_seq = seqs[ref_key]

    variants = pd.read_csv(VARIANTS_CSV, low_memory=False)
    parsed_variants = []
    for source_idx, row in variants.iterrows():
        parsed = parse_variant(row.get("Protein Consequence"))
        if parsed:
            parsed["Source_Row"] = source_idx + 1
            parsed_variants.append(parsed)

    detail_rows = []
    summary_rows = []

    for structure, cfg in STRUCTURES.items():
        seq_key = find_key(seqs, structure, cfg["prefix"])
        aln_map = build_alignment_map(ref_seq, seqs[seq_key], structure)
        pdb_res = pdb_ca_residues(cfg["pdb"])
        foldx_list = read_list(cfg["list"])

        expected_entries = []
        for parsed in parsed_variants:
            map_info = aln_map.get(parsed["Human_Pos"])
            if not map_info:
                continue
            pdb_pos = map_info["Expected_PDB_Pos"]
            wt_pdb = pdb_res.get(pdb_pos)
            if wt_pdb is None:
                continue
            expected_entries.append(
                {
                    **parsed,
                    **map_info,
                    "Expected_Input": f"{wt_pdb}A{pdb_pos}{parsed['Mut_1']}",
                    "PDB_AA": wt_pdb,
                    "PDB_Pos": pdb_pos,
                }
            )

        ok = 0
        bad = 0
        for idx, expected in enumerate(expected_entries, start=1):
            actual = foldx_list[idx - 1] if idx <= len(foldx_list) else ""
            parsed_input = parse_foldx_input(actual)
            actual_wt = actual_pos = actual_mut = None
            pdb_has_actual = False
            pdb_actual_aa = ""
            if parsed_input:
                actual_wt, actual_pos, actual_mut = parsed_input
                pdb_actual_aa = pdb_res.get(actual_pos, "")
                pdb_has_actual = actual_pos in pdb_res

            status = "OK"
            checks = {
                "List_Equals_Expected": actual == expected["Expected_Input"],
                "Actual_Pos_Equals_Alignment_Pos": actual_pos == expected["PDB_Pos"],
                "Actual_WT_Equals_PDB": actual_wt == pdb_actual_aa,
                "Actual_Mut_Equals_Variant_Mut": actual_mut == expected["Mut_1"],
                "PDB_Position_Exists": pdb_has_actual,
            }
            if not all(checks.values()):
                status = "MISMATCH"
                bad += 1
            else:
                ok += 1

            detail_rows.append(
                {
                    "Structure": structure,
                    "FoldX_Index": idx,
                    "Source_Row": expected["Source_Row"],
                    "Variant": expected["Variant"],
                    "Human_Pos": expected["Human_Pos"],
                    "Human_AA_Alignment": expected["Human_AA_Alignment"],
                    "Target_AA_Alignment": expected["Target_AA_Alignment"],
                    "Alignment_Column": expected["Alignment_Column"],
                    "Expected_PDB_Pos_From_Alignment": expected["PDB_Pos"],
                    "Expected_Input_From_Alignment_And_PDB": expected["Expected_Input"],
                    "Actual_FoldX_Input": actual,
                    "Actual_PDB_Pos": actual_pos,
                    "PDB_AA_At_Actual_Pos": pdb_actual_aa,
                    "Status": status,
                    **checks,
                }
            )

        summary_rows.append(
            {
                "Structure": structure,
                "Alignment_Key": seq_key,
                "PDB_CA_Residues": len(pdb_res),
                "Expected_Valid_Inputs_From_Alignment_And_PDB": len(expected_entries),
                "Actual_FoldX_List_Inputs": len(foldx_list),
                "OK": ok,
                "MISMATCH": bad,
                "Length_Matches": len(expected_entries) == len(foldx_list),
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUT_DETALLE, index=False)
    summary.to_csv(OUT_RESUMEN, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote: {OUT_RESUMEN}")
    print(f"Wrote: {OUT_DETALLE}")


if __name__ == "__main__":
    main()
