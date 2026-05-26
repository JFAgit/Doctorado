from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\MarceNIS")
FOLDX = BASE / "FoldX"
SCRIPTS = FOLDX / "Scripts"
VARIANTS_CSV = BASE / "Variantes" / "gnomAD_v4_UNIFICADO_FINAL.csv"
ALIGNMENT = SCRIPTS / "alineamiento_NIS.aln"

OUTPUT_CSV = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\TABLA_MAESTRA_FOLDX_NIS_RECONSTRUIDA.csv")
QC_CSV = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\QC_TABLA_MAESTRA_FOLDX_NIS_RECONSTRUIDA.csv")


STRUCTURES = {
    "7UUY": {
        "seq_key_prefix": "7UUY",
        "pdb": FOLDX / "Estructura1" / "7UUY_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UUY.txt",
        "dif": FOLDX / "Estructura1" / "Dif_7UUY_Repair.fxout",
    },
    "7UUZ": {
        "seq_key_prefix": "7UUZ",
        "pdb": FOLDX / "Estructura2" / "7UUZ_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UUZ.txt",
        "dif": FOLDX / "Estructura2" / "Dif_7UUZ_Repair.fxout",
    },
    "7UV0": {
        "seq_key_prefix": "7UV0",
        "pdb": FOLDX / "Estructura3" / "7UV0_Repair.pdb",
        "list": SCRIPTS / "individual_list_7UV0.txt",
        "dif": FOLDX / "Estructura3" / "Dif_7UV0_Repair.fxout",
    },
    "AF": {
        "seq_key_prefix": "AF",
        "pdb": FOLDX / "EstructuraAlphaFold" / "AF-Q92911model_Repair.pdb",
        "list": SCRIPTS / "individual_list_AF.txt",
        "dif": FOLDX / "EstructuraAlphaFold" / "Dif_AF-Q92911model_Repair.fxout",
    },
}


AA3_TO_1_TITLE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}

AA3_TO_1_UPPER = {k.upper(): v for k, v in AA3_TO_1_TITLE.items()}
PDB_AA3_TO_1 = AA3_TO_1_UPPER.copy()


def parse_fasta_alignment(path):
    sequences = {}
    current = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:]
                sequences[current] = ""
            elif current is not None:
                sequences[current] += line
    return sequences


def find_sequence_key(alignment, structure_name, prefix):
    if structure_name == "AF":
        for key in alignment:
            if "AF" in key or "Q92911" in key:
                return key
    for key in alignment:
        if key.split("_")[0] == prefix or key.startswith(prefix):
            return key
    raise KeyError(f"No sequence found for {structure_name} with prefix {prefix}")


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
                residues[res_num] = PDB_AA3_TO_1.get(res_name, "X")
    return residues


def human_to_target_map(ref_seq, target_seq, structure_name):
    human_pos = 0
    target_pos = 0
    mapping = {}

    for human_aa, target_aa in zip(ref_seq, target_seq):
        if human_aa != "-":
            human_pos += 1
        if target_aa != "-":
            target_pos += 1
        if human_aa != "-" and target_aa != "-":
            if structure_name == "AF":
                mapping[human_pos] = target_pos
            else:
                # This mirrors the generator that produced the successful FoldX lists.
                mapping[human_pos] = target_pos + 8
    return mapping


def parse_protein_consequence(value):
    pc = "" if pd.isna(value) else str(value).strip()
    if not pc.startswith("p."):
        return None

    upper = pc.upper()
    variant_type = "other"
    foldx_applicable = True
    caution = ""

    if "FS" in upper:
        variant_type = "frameshift"
        foldx_applicable = False
        caution = "Frameshift: FoldX substitution energy is not biologically interpretable."
    elif "*" in pc or "TER" in upper or "STOP" in upper:
        variant_type = "stop_gained"
        foldx_applicable = False
        caution = "Stop/nonsense: FoldX does not model protein truncation."
    elif "?" in pc:
        variant_type = "uncertain_protein_start_or_unknown"
        foldx_applicable = False
        caution = "Protein consequence contains '?'; not a standard missense substitution."

    digits = "".join(ch for ch in pc if ch.isdigit())
    if not digits:
        return {
            "variant": pc,
            "wt_3": None,
            "pos": None,
            "mut_3": None,
            "wt_1": "X",
            "mut_1": "X",
            "variant_type": variant_type,
            "foldx_applicable": False,
            "caution": caution or "Could not parse as p.Aaa123Bbb.",
        }

    # This intentionally mirrors generar_mapeo_total.py, which used fixed
    # string slices and pc[-3:] to build the FoldX input lists.
    wt_3 = pc[2:5]
    pos = int(digits)
    mut_3 = pc[-3:]
    wt_1 = AA3_TO_1_TITLE.get(wt_3.capitalize(), "X")
    mut_1 = AA3_TO_1_TITLE.get(mut_3.capitalize(), "X")

    if variant_type == "other":
        if wt_1 != "X" and mut_1 != "X":
            variant_type = "missense"
        else:
            variant_type = "non_missense_or_unknown"
            foldx_applicable = False
            caution = "Not a standard amino-acid substitution."

    return {
        "variant": pc,
        "wt_3": wt_3.capitalize(),
        "pos": pos,
        "mut_3": mut_3.capitalize(),
        "wt_1": wt_1,
        "mut_1": mut_1,
        "variant_type": variant_type,
        "foldx_applicable": foldx_applicable,
        "caution": caution,
    }


def read_individual_list(path):
    entries = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            item = line.strip().replace(" ", "")
            if item:
                entries.append(item.rstrip(";"))
    return entries


def load_dif_by_index(path):
    # FoldX may append to an existing Dif_*.fxout if the same folder is reused.
    # Keep the last value for each (mutation index, run index), which matches the
    # current PDB files left in the folder.
    by_pair = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("FoldX", "by ", "Jesper", "Luis", "-", "Pdb", "PDB", "Output")):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            match = re.search(r"_(\d+)_(\d+)\.pdb$", parts[0])
            if not match:
                continue
            try:
                energy = float(parts[1])
            except ValueError:
                continue
            idx = int(match.group(1))
            run = int(match.group(2))
            by_pair[(idx, run)] = energy

    by_index = {}
    for (idx, _run), energy in by_pair.items():
        by_index.setdefault(idx, []).append(energy)

    return {
        idx: {
            "ddg": sum(values) / len(values),
            "runs": len(values),
        }
        for idx, values in by_index.items()
    }


def build_expected_inputs(variants_df, alignment):
    ref_key = find_sequence_key(alignment, "AF", "AF")
    ref_seq = alignment[ref_key]

    expected = {name: [] for name in STRUCTURES}
    rows = []

    for source_row, row in variants_df.iterrows():
        parsed = parse_protein_consequence(row.get("Protein Consequence"))
        if parsed is None:
            continue

        rows.append((source_row, parsed))

    for name, cfg in STRUCTURES.items():
        seq_key = find_sequence_key(alignment, name, cfg["seq_key_prefix"])
        mapping = human_to_target_map(ref_seq, alignment[seq_key], name)
        pdb_residues = pdb_ca_residues(cfg["pdb"])

        for source_row, parsed in rows:
            input_value = ""
            real_pos = None
            if parsed["pos"] is not None:
                real_pos = mapping.get(parsed["pos"])
                if real_pos in pdb_residues:
                    wt_pdb = pdb_residues[real_pos]
                    input_value = f"{wt_pdb}A{real_pos}{parsed['mut_1']}"

            if input_value:
                expected[name].append(
                    {
                        "source_row": source_row,
                        "variant": parsed["variant"],
                        "input": input_value,
                        "real_pos": real_pos,
                    }
                )

    return rows, expected


def main():
    variants = pd.read_csv(VARIANTS_CSV, low_memory=False)
    alignment = parse_fasta_alignment(ALIGNMENT)
    parsed_rows, expected = build_expected_inputs(variants, alignment)

    out_rows = []
    parsed_by_source_row = {source_row: parsed for source_row, parsed in parsed_rows}

    for source_row, parsed in parsed_rows:
        original = variants.loc[source_row]
        out = {
            "Source_Row": int(source_row) + 1,
            "Variante": parsed["variant"],
            "Variant_Type": parsed["variant_type"],
            "FoldX_Applicable": parsed["foldx_applicable"],
            "FoldX_Caution": parsed["caution"],
            "ClinVar": original.get("ClinVar Germline Classification", ""),
            "gnomAD_ID": original.get("gnomAD ID", ""),
            "rsIDs": original.get("rsIDs", ""),
            "Allele_Frequency": original.get("Allele Frequency", ""),
            "Protein_Consequence": original.get("Protein Consequence", ""),
            "HGVS_Consequence": original.get("HGVS Consequence", ""),
        }
        for name in STRUCTURES:
            out[f"FoldX_Input_{name}"] = ""
            out[f"FoldX_Index_{name}"] = ""
            out[f"DDG_{name}"] = ""
            out[f"FoldX_Runs_{name}"] = ""
            out[f"FoldX_Status_{name}"] = "not_mapped_to_structure"
        out_rows.append(out)

    row_by_source = {row["Source_Row"] - 1: row for row in out_rows}
    qc_rows = []

    for name, cfg in STRUCTURES.items():
        actual_list = read_individual_list(cfg["list"])
        expected_list = [entry["input"] for entry in expected[name]]
        dif = load_dif_by_index(cfg["dif"])

        mismatches = sum(
            1 for expected_item, actual_item in zip(expected_list, actual_list)
            if expected_item != actual_item
        )
        length_matches = len(expected_list) == len(actual_list)

        qc_rows.append(
            {
                "Structure": name,
                "Expected_Input_Count": len(expected_list),
                "Actual_List_Count": len(actual_list),
                "Dif_Index_Count": len(dif),
                "Length_Matches": length_matches,
                "Line_By_Line_Mismatches": mismatches,
                "Max_Runs_Per_Index": max((v["runs"] for v in dif.values()), default=0),
                "Min_Runs_Per_Index": min((v["runs"] for v in dif.values()), default=0),
            }
        )

        # Use the actual FoldX list order. The QC tells us whether it matches the
        # recomputed mapping. If it does, each index can be assigned to a human row.
        for idx, entry in enumerate(expected[name], start=1):
            target = row_by_source[entry["source_row"]]
            target[f"FoldX_Input_{name}"] = actual_list[idx - 1] if idx <= len(actual_list) else entry["input"]
            target[f"FoldX_Index_{name}"] = idx
            if idx in dif:
                target[f"DDG_{name}"] = round(dif[idx]["ddg"], 6)
                target[f"FoldX_Runs_{name}"] = dif[idx]["runs"]
                target[f"FoldX_Status_{name}"] = "ok"
            else:
                target[f"FoldX_Status_{name}"] = "input_without_dif_result"

    final = pd.DataFrame(out_rows)

    # Put the key columns first, then the per-structure columns.
    key_cols = [
        "Source_Row",
        "Variante",
        "Variant_Type",
        "FoldX_Applicable",
        "FoldX_Caution",
        "ClinVar",
        "gnomAD_ID",
        "rsIDs",
        "Allele_Frequency",
        "Protein_Consequence",
        "HGVS_Consequence",
    ]
    structure_cols = []
    for name in STRUCTURES:
        structure_cols.extend(
            [
                f"FoldX_Input_{name}",
                f"FoldX_Index_{name}",
                f"DDG_{name}",
                f"FoldX_Runs_{name}",
                f"FoldX_Status_{name}",
            ]
        )
    final = final[key_cols + structure_cols]

    final.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(qc_rows).to_csv(QC_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {QC_CSV}")
    print()
    print(pd.DataFrame(qc_rows).to_string(index=False))
    print()
    ddg_cols = [f"DDG_{name}" for name in STRUCTURES]
    for col in ddg_cols:
        print(f"{col}: {(final[col].astype(str) != '').sum()} non-empty cells")


if __name__ == "__main__":
    main()
