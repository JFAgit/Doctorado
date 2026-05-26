from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
INPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI_UNIPROT.csv"
OUT_ROWS = BASE / "CONTEO_CLASIFICACIONES_NORMALIZADAS_POR_FILAS.csv"
OUT_UNIQUE = BASE / "CONTEO_CLASIFICACIONES_NORMALIZADAS_VARIANTES_UNICAS.csv"


def normalize_classification(value):
    text = "" if pd.isna(value) else str(value).strip()
    lower = text.lower()
    if lower in {"", "nan"}:
        return "Sin dato"
    if "not classified" in lower or "sin clasificar" in lower:
        return "VUS / Sin clasificar"
    if lower == "uncertain significance":
        return "Uncertain significance"
    if lower == "likely pathogenic":
        return "Likely pathogenic"
    if lower == "pathogenic":
        return "Pathogenic"
    if lower in {"pathogenic/likely pathogenic", "likely pathogenic/pathogenic"}:
        return "Pathogenic/Likely pathogenic"
    if lower == "likely benign":
        return "Likely benign"
    if lower == "benign":
        return "Benign"
    if lower in {"benign/likely benign", "likely benign/benign"}:
        return "Benign/Likely benign"
    if "conflicting" in lower:
        return "Conflicting classifications"
    return text


df = pd.read_csv(INPUT)
df["Variante_normalizada"] = df["Variante"].astype(str).str.strip().str.lower()
df["Clasificacion_normalizada"] = df["ClinVar_Final_Classification"].map(normalize_classification)

count_rows = (
    df["Clasificacion_normalizada"]
    .value_counts()
    .rename_axis("Clasificacion")
    .reset_index(name="Conteo_filas")
)

# One row per protein-variant name, ignoring case. If duplicates disagree, keep
# a deterministic representative after sorting by normalized class.
unique = df.sort_values(["Variante_normalizada", "Clasificacion_normalizada"]).drop_duplicates("Variante_normalizada")
count_unique = (
    unique["Clasificacion_normalizada"]
    .value_counts()
    .rename_axis("Clasificacion")
    .reset_index(name="Conteo_variantes_unicas")
)

count_rows.to_csv(OUT_ROWS, index=False)
count_unique.to_csv(OUT_UNIQUE, index=False)

print(f"Filas totales: {len(df)}")
print(f"Variantes unicas por nombre, ignorando mayus/minus: {df['Variante_normalizada'].nunique()}")
print()
print("Conteo por filas:")
print(count_rows.to_string(index=False))
print()
print("Conteo por variantes unicas:")
print(count_unique.to_string(index=False))
print()
print(f"Wrote: {OUT_ROWS}")
print(f"Wrote: {OUT_UNIQUE}")
