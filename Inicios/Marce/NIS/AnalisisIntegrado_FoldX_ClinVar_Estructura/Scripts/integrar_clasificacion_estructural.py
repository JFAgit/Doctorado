from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
STRUCTURAL = Path(r"C:\Users\fran_\Documents\Doctorado\MarceNIS\AnalisisEstructural\residuos_clasificados.csv")
INPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_UNIPROT_LITERATURA.csv"
OUTPUT = BASE / "TABLA_FOLDX_NIS_FINAL_CON_INFO_ESTRUCTURAL.csv"
OUTPUT_SIMPLE = BASE / "TABLA_FOLDX_NIS_FINAL_SIMPLE_ESTRUCTURAL.csv"
QC = BASE / "QC_INTEGRACION_INFO_ESTRUCTURAL.csv"


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
    cats = {normalize_category(c) for c in categories if str(c).strip()}
    if "sitio activo" in cats:
        return "sitio activo"
    if "core" in cats:
        return "core"
    if "superficie" in cats:
        return "superficie"
    return ""


struct = pd.read_csv(STRUCTURAL)
struct["Residuo"] = struct["Residuo"].astype(str).str.strip()
struct["Categoria_Normalizada"] = struct["Categoría"].map(normalize_category)
struct["Structural_Position"] = struct["Residuo"].map(extract_pos)

grouped = (
    struct.dropna(subset=["Structural_Position"])
    .groupby("Structural_Position")
    .agg(
        Structural_Residues=("Residuo", lambda x: ";".join(sorted(set(map(str, x))))),
        Structural_Categories_All=("Categoria_Normalizada", lambda x: ";".join(sorted(set(map(str, x))))),
        Structural_Category=("Categoria_Normalizada", consensus),
    )
    .reset_index()
)
grouped["Structural_Position"] = grouped["Structural_Position"].astype(int)

df = pd.read_csv(INPUT)
df["Variant_Position"] = df["Variante"].map(extract_pos)
df = df.merge(grouped, left_on="Variant_Position", right_on="Structural_Position", how="left")
df["Structural_Annotation_Source"] = df["Structural_Category"].notna().map(
    {True: str(STRUCTURAL), False: ""}
)

front_cols = [
    "Variante",
    "DDG_7UUY",
    "DDG_7UUZ",
    "DDG_7UV0",
    "DDG_AF",
    "Structural_Category",
    "Structural_Categories_All",
    "Structural_Residues",
    "Structural_Position",
    "Structural_Annotation_Source",
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
    "Structural_Category",
    "Structural_Categories_All",
    "ClinVar_Final_Classification",
    "Literature_Label",
    "Allele_Frequency",
]
df[simple_cols].to_csv(OUTPUT_SIMPLE, index=False)

qc_rows = [
    {"Metric": "Input rows", "Value": len(df)},
    {"Metric": "Rows with structural category", "Value": int(df["Structural_Category"].notna().sum())},
    {"Metric": "Rows without structural category", "Value": int(df["Structural_Category"].isna().sum())},
    {"Metric": "Unique structural positions classified", "Value": int(grouped["Structural_Position"].nunique())},
]
for cat, count in df["Structural_Category"].fillna("sin anotacion").value_counts().items():
    qc_rows.append({"Metric": f"Rows category: {cat}", "Value": int(count)})

pd.DataFrame(qc_rows).to_csv(QC, index=False)

print(f"Wrote: {OUTPUT}")
print(f"Wrote: {OUTPUT_SIMPLE}")
print(f"Wrote: {QC}")
print(pd.DataFrame(qc_rows).to_string(index=False))
