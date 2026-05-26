from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
NIS_STRUCT = BASE / "NIS" / "ClasificacionEstructural"
INPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_UNIPROT_LITERATURA.csv"
OUTPUT = BASE / "TABLA_FOLDX_NIS_FINAL_CON_INFO_ESTRUCTURAL.csv"
OUTPUT_SIMPLE = BASE / "TABLA_FOLDX_NIS_FINAL_SIMPLE_ESTRUCTURAL.csv"
QC = BASE / "QC_INTEGRACION_INFO_ESTRUCTURAL.csv"

STRUCT_FILES = {
    "7UUY": NIS_STRUCT / "residuos_clasificados_7uuy.csv",
    "7UUZ": NIS_STRUCT / "residuos_clasificados_7uuz.csv",
    "7UV0": NIS_STRUCT / "residuos_clasificados_7uv0.csv",
    "AF": NIS_STRUCT / "residuos_clasificados_AF_Human.csv",
}


def extract_pos(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def normalize_category(value):
    text = str(value).strip().lower()
    if text in {"sitio activo", "active site", "sitio_activo"}:
        return "sitio activo"
    if text == "core":
        return "core"
    if text in {"superficie", "surface"}:
        return "superficie"
    return text


def consensus(categories):
    cats = {normalize_category(c) for c in categories if not pd.isna(c) and str(c).strip()}
    if "sitio activo" in cats:
        return "sitio activo"
    if "core" in cats:
        return "core"
    if "superficie" in cats:
        return "superficie"
    return ""


def load_structural_map(path):
    df = pd.read_csv(path)
    cat_col = "Categoría" if "Categoría" in df.columns else "CategorÃ­a"
    df["pos"] = df["Residuo"].map(extract_pos)
    df["cat"] = df[cat_col].map(normalize_category)
    grouped = (
        df.dropna(subset=["pos"])
        .groupby("pos")
        .agg(
            category=("cat", consensus),
            all_categories=("cat", lambda x: ";".join(sorted(set(map(str, x))))),
            residues=("Residuo", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    return grouped


df = pd.read_csv(INPUT)

qc_rows = []
consensus_inputs = []
for structure, path in STRUCT_FILES.items():
    mapping = load_structural_map(path)
    input_col = f"FoldX_Input_{structure}"
    pos_col = f"Structural_Position_{structure}"
    cat_col = f"Structural_{structure}"
    all_col = f"Structural_All_{structure}"
    res_col = f"Structural_Residue_{structure}"

    df[pos_col] = df[input_col].map(extract_pos) if input_col in df.columns else None
    df[cat_col] = df[pos_col].map(lambda p: mapping.loc[p, "category"] if pd.notna(p) and p in mapping.index else "")
    df[all_col] = df[pos_col].map(lambda p: mapping.loc[p, "all_categories"] if pd.notna(p) and p in mapping.index else "")
    df[res_col] = df[pos_col].map(lambda p: mapping.loc[p, "residues"] if pd.notna(p) and p in mapping.index else "")
    consensus_inputs.append(cat_col)

    qc_rows.append({"Metric": f"{structure}: classified residue positions in source", "Value": int(mapping.shape[0])})
    qc_rows.append({"Metric": f"{structure}: rows with FoldX input", "Value": int(df[input_col].notna().sum()) if input_col in df.columns else 0})
    qc_rows.append({"Metric": f"{structure}: rows with structural category", "Value": int((df[cat_col].astype(str) != "").sum())})
    for cat, count in df[cat_col].replace("", "sin anotacion").value_counts().items():
        qc_rows.append({"Metric": f"{structure}: rows category {cat}", "Value": int(count)})

df["Structural_Category_Consensus"] = df[consensus_inputs].apply(lambda row: consensus(row.tolist()), axis=1)
df["Structural_Categories_By_Structure"] = df[consensus_inputs].apply(
    lambda row: "; ".join(f"{col.replace('Structural_', '')}={val}" for col, val in row.items() if str(val).strip()),
    axis=1,
)

front_cols = [
    "Variante",
    "DDG_7UUY",
    "DDG_7UUZ",
    "DDG_7UV0",
    "DDG_AF",
    "Structural_Category_Consensus",
    "Structural_Categories_By_Structure",
    "Structural_7UUY",
    "Structural_7UUZ",
    "Structural_7UV0",
    "Structural_AF",
]

remaining = [c for c in df.columns if c not in front_cols]
df = df[front_cols + remaining]
df.to_csv(OUTPUT, index=False)

simple_cols = [
    "Variante",
    "DDG_7UUY",
    "DDG_7UUZ",
    "DDG_7UV0",
    "DDG_AF",
    "Structural_Category_Consensus",
    "Structural_Categories_By_Structure",
    "Structural_7UUY",
    "Structural_7UUZ",
    "Structural_7UV0",
    "Structural_AF",
    "ClinVar_Final_Classification",
    "Literature_Label",
    "Allele_Frequency",
]
df[simple_cols].to_csv(OUTPUT_SIMPLE, index=False)

qc_rows.insert(0, {"Metric": "Input rows", "Value": len(df)})
qc_rows.insert(1, {"Metric": "Rows with structural consensus", "Value": int((df["Structural_Category_Consensus"].astype(str) != "").sum())})
for cat, count in df["Structural_Category_Consensus"].replace("", "sin anotacion").value_counts().items():
    qc_rows.append({"Metric": f"Consensus: rows category {cat}", "Value": int(count)})

pd.DataFrame(qc_rows).to_csv(QC, index=False)

print(f"Wrote: {OUTPUT}")
print(f"Wrote: {OUTPUT_SIMPLE}")
print(f"Wrote: {QC}")
print(pd.DataFrame(qc_rows).to_string(index=False))
