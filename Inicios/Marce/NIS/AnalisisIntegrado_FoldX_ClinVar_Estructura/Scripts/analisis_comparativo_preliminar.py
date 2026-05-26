from pathlib import Path
import math

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\NIS\AnalisisIntegrado_FoldX_ClinVar_Estructura")
INPUT = BASE / "Tablas" / "TABLA_FOLDX_NIS_FINAL_CON_INFO_ESTRUCTURAL.csv"
OUTDIR = BASE / "AnalisisComparativo"
OUTDIR.mkdir(exist_ok=True)

DDG_COLS = ["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0", "DDG_AF"]
STATE_LABELS = {
    "DDG_7UUY": "7UUY_apo",
    "DDG_7UUZ": "7UUZ_ReO4_Na",
    "DDG_7UV0": "7UV0_I_Na",
    "DDG_AF": "AF_humano",
}


def markdown_table(frame, floatfmt=".3f"):
    if frame.empty:
        return "_Sin filas_"
    cols = list(frame.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    vals.append("")
                else:
                    vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def normalize_clinical_group(value):
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text in {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}:
        return "Pathogenic_or_likely"
    if text in {"benign", "likely benign", "benign/likely benign"}:
        return "Benign_or_likely"
    if "conflicting" in text:
        return "Conflicting"
    if text == "uncertain significance":
        return "Uncertain_significance"
    return "VUS_or_unclassified"


def summarize_numeric(df, group_cols, value_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row = {**key_dict, "Structure": STATE_LABELS[col], "Column": col, "N": int(vals.shape[0])}
            if vals.empty:
                row.update({"Mean": math.nan, "Median": math.nan, "Q1": math.nan, "Q3": math.nan, "Min": math.nan, "Max": math.nan})
                row.update({"Pct_DDG_gt_1": math.nan, "Pct_DDG_gt_2": math.nan, "Pct_DDG_gt_3": math.nan})
            else:
                row.update(
                    {
                        "Mean": vals.mean(),
                        "Median": vals.median(),
                        "Q1": vals.quantile(0.25),
                        "Q3": vals.quantile(0.75),
                        "Min": vals.min(),
                        "Max": vals.max(),
                        "Pct_DDG_gt_1": 100 * (vals > 1).mean(),
                        "Pct_DDG_gt_2": 100 * (vals > 2).mean(),
                        "Pct_DDG_gt_3": 100 * (vals > 3).mean(),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT)
    for col in DDG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Clinical_Group"] = df["ClinVar_Final_Classification"].map(normalize_clinical_group)
    df["Any_Experimental_DDG"] = df[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].notna().any(axis=1)
    df["All_Experimental_DDG"] = df[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].notna().all(axis=1)
    df["All_4_DDG"] = df[DDG_COLS].notna().all(axis=1)
    df["Experimental_Mean_DDG"] = df[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].mean(axis=1, skipna=True)
    df["Experimental_Max_DDG"] = df[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].max(axis=1, skipna=True)
    df["Experimental_Min_DDG"] = df[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].min(axis=1, skipna=True)
    df["Conformational_Range_Experimental"] = df["Experimental_Max_DDG"] - df["Experimental_Min_DDG"]

    def most_destabilizing(row):
        vals = row[["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0"]].dropna()
        if vals.empty:
            return ""
        return STATE_LABELS[vals.idxmax()]

    df["Most_Destabilizing_Experimental_State"] = df.apply(most_destabilizing, axis=1)

    summary_clinical = summarize_numeric(df, ["Clinical_Group"], DDG_COLS)
    summary_structural = summarize_numeric(df, ["Structural_Category_Consensus"], DDG_COLS)
    summary_clinical_structural = summarize_numeric(df, ["Clinical_Group", "Structural_Category_Consensus"], DDG_COLS)

    counts = []
    counts.append({"Metric": "Rows", "Value": len(df)})
    for group, count in df["Clinical_Group"].value_counts().items():
        counts.append({"Metric": f"Clinical group: {group}", "Value": int(count)})
    for cat, count in df["Structural_Category_Consensus"].fillna("sin_anotacion").value_counts().items():
        counts.append({"Metric": f"Structural consensus: {cat}", "Value": int(count)})
    for state, count in df["Most_Destabilizing_Experimental_State"].replace("", "no_experimental_ddg").value_counts().items():
        counts.append({"Metric": f"Most destabilizing experimental state: {state}", "Value": int(count)})

    top_destabilizing = df.sort_values("Experimental_Max_DDG", ascending=False).head(50)
    top_sensitive = df.sort_values("Conformational_Range_Experimental", ascending=False).head(50)
    pathogenic_or_likely = df[df["Clinical_Group"].eq("Pathogenic_or_likely")].sort_values("Experimental_Max_DDG", ascending=False)
    benign_or_likely = df[df["Clinical_Group"].eq("Benign_or_likely")].sort_values("Experimental_Max_DDG", ascending=False)

    cols_front = [
        "Variante",
        "Clinical_Group",
        "ClinVar_Final_Classification",
        "Structural_Category_Consensus",
        "Structural_Categories_By_Structure",
        "DDG_7UUY",
        "DDG_7UUZ",
        "DDG_7UV0",
        "DDG_AF",
        "Experimental_Mean_DDG",
        "Experimental_Max_DDG",
        "Conformational_Range_Experimental",
        "Most_Destabilizing_Experimental_State",
        "Allele_Frequency",
        "Literature_Label",
        "Functional_Summary",
    ]
    cols_front = [c for c in cols_front if c in df.columns]

    summary_clinical.to_csv(OUTDIR / "summary_ddg_by_clinical_group.csv", index=False)
    summary_structural.to_csv(OUTDIR / "summary_ddg_by_structural_category.csv", index=False)
    summary_clinical_structural.to_csv(OUTDIR / "summary_ddg_by_clinical_and_structural.csv", index=False)
    pd.DataFrame(counts).to_csv(OUTDIR / "summary_counts.csv", index=False)
    df[cols_front].to_csv(OUTDIR / "tabla_analisis_comparativo_preliminar.csv", index=False)
    top_destabilizing[cols_front].to_csv(OUTDIR / "top50_mas_desestabilizantes_experimental.csv", index=False)
    top_sensitive[cols_front].to_csv(OUTDIR / "top50_mayor_sensibilidad_conformacional.csv", index=False)
    pathogenic_or_likely[cols_front].to_csv(OUTDIR / "variantes_patogenicas_resumen.csv", index=False)
    benign_or_likely[cols_front].to_csv(OUTDIR / "variantes_benignas_resumen.csv", index=False)

    md = []
    md.append("# Analisis comparativo preliminar NIS / FoldX")
    md.append("")
    md.append("## Conteos")
    md.append(markdown_table(pd.DataFrame(counts)))
    md.append("")
    md.append("## DDG por grupo clinico")
    md.append(markdown_table(summary_clinical))
    md.append("")
    md.append("## DDG por categoria estructural")
    md.append(markdown_table(summary_structural))
    md.append("")
    md.append("## Top 20 por DDG experimental maximo")
    md.append(markdown_table(top_destabilizing[cols_front].head(20)))
    md.append("")
    md.append("## Top 20 por sensibilidad conformacional experimental")
    md.append(markdown_table(top_sensitive[cols_front].head(20)))
    (OUTDIR / "RESUMEN_ANALISIS_COMPARATIVO_PRELIMINAR.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote outputs to: {OUTDIR}")
    print(pd.DataFrame(counts).to_string(index=False))
    print("\nClinical summary:")
    print(summary_clinical.to_string(index=False))


if __name__ == "__main__":
    main()
