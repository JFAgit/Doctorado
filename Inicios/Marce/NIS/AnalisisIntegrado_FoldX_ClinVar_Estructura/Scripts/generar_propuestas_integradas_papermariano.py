from pathlib import Path
import re

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TABLAS = BASE / "Tablas"
DATOS = BASE / "DatosExternos"

FINAL = TABLAS / "TABLA_FOLDX_NIS_FINAL_CON_INFO_ESTRUCTURAL.csv"
S2_MATCHES = DATOS / "SAVY_SUPPLEMENTARY_TABLE2_MATCHES_TO_FINAL.csv"
S3_MATCHES = DATOS / "SAVY_SUPPLEMENTARY_TABLE3_MATCHES_TO_FINAL.csv"

OUT_DETALLADA = TABLAS / "PROPUESTA_1_TRAZABLE_CLINVAR_PAPERMARIANO.csv"
OUT_LEGIBLE = TABLAS / "PROPUESTA_2_FINAL_LEGIBLE_INTEGRADA.csv"
OUT_XLSX = TABLAS / "PROPUESTAS_INTEGRACION_PAPERMARIANO.xlsx"
OUT_RESUMEN = BASE / "DatosExternos" / "RESUMEN_INTEGRACION_PAPERMARIANO_S2_S3.md"


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm_variant(value):
    return clean_text(value).replace(" ", "").upper()


def protein_position(variant):
    match = re.search(r"(\d+)", clean_text(variant))
    return int(match.group(1)) if match else 10**9


def to_float(value):
    value = clean_text(value)
    if not value or value.lower() == "nan":
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def clinical_group(label):
    label_l = clean_text(label).lower()
    if "conflicting" in label_l:
        return "Conflicting"
    if "pathogenic" in label_l:
        return "Pathogenic_or_likely"
    if "benign" in label_l:
        return "Benign_or_likely"
    if "uncertain" in label_l or "not classified" in label_l:
        return "Uncertain_or_unclassified"
    return "Other"


def is_informative_acmg(label):
    group = clinical_group(label)
    return group in {"Pathogenic_or_likely", "Benign_or_likely"}


def functional_integrated_label(label):
    label_l = clean_text(label).lower()
    if "pathogenic" in label_l:
        return "Functional pathogenic"
    if "benign" in label_l:
        return "Functional benign"
    if "intermediate" in label_l:
        return "Functional intermediate"
    return ""


def first_nonempty(values):
    for value in values:
        value = clean_text(value)
        if value:
            return value
    return ""


def counts_to_markdown(series, name_col, count_col="N"):
    lines = [f"| {name_col} | {count_col} |", "| --- | ---: |"]
    for name, count in series.items():
        lines.append(f"| {clean_text(name)} | {count} |")
    return "\n".join(lines)


def load_matches(path, key_col="Variante_Final"):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, row in df.iterrows():
        key = norm_variant(row.get(key_col, ""))
        if key and key not in out:
            out[key] = row.to_dict()
    return out


def write_xlsx(detailed, legible, counts, source_counts):
    resumen = pd.DataFrame(
        [
            {"Seccion": "Total", "Campo": "Variantes unicas", "Valor": len(legible)},
            *[
                {"Seccion": "Clasificacion integrada", "Campo": idx, "Valor": int(val)}
                for idx, val in counts.items()
            ],
            *[
                {"Seccion": "Fuente clasificacion", "Campo": idx, "Valor": int(val)}
                for idx, val in source_counts.items()
            ],
        ]
    )
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        legible.to_excel(writer, index=False, sheet_name="Propuesta_2_Final")
        detailed.to_excel(writer, index=False, sheet_name="Propuesta_1_Trazable")
        resumen.to_excel(writer, index=False, sheet_name="Resumen")

        for sheet_name, width_limits in {
            "Propuesta_2_Final": (12, 36),
            "Propuesta_1_Trazable": (12, 48),
            "Resumen": (12, 56),
        }.items():
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            min_w, max_w = width_limits
            for column_cells in ws.columns:
                header = clean_text(column_cells[0].value)
                max_len = max(len(clean_text(cell.value)) for cell in column_cells[:200])
                width = max(min_w, min(max_w, max(max_len + 2, len(header) + 2)))
                ws.column_dimensions[column_cells[0].column_letter].width = width


def integrated_classification(row, s2, s3):
    clinvar = clean_text(row.get("ClinVar_Final_Classification", ""))
    clinvar_source = clean_text(row.get("ClinVar_Final_Source", ""))
    clinvar_group = clinical_group(clinvar)

    if clinvar_group in {"Pathogenic_or_likely", "Benign_or_likely"}:
        return clinvar, clinvar_source or "ClinVar/gnomAD/NCBI"

    s2_label = clean_text(s2.get("ACMG_Result_SAVY", ""))
    if is_informative_acmg(s2_label):
        return s2_label, "PaperMariano_SupplementaryTable2_ACMG"

    s3_label_raw = clean_text(s3.get("Functional_Label_From_Activity", ""))
    s3_label = functional_integrated_label(s3_label_raw)
    if s3_label:
        return s3_label, "PaperMariano_SupplementaryTable3_FunctionalActivity"

    if clinvar_group == "Conflicting":
        return clinvar, clinvar_source or "ClinVar/gnomAD/NCBI"

    if clinvar:
        return clinvar, clinvar_source or "gnomAD extract"

    return "Uncertain significance / not classified", "No informative external classification"


def choose_representative(group):
    group = group.copy()
    for col in ["Allele_Frequency", "DDG_7UUY", "DDG_7UUZ", "DDG_7UV0", "DDG_AF"]:
        if col in group.columns:
            group[f"_{col}_num"] = group[col].map(to_float)
    # Prefer the row with the highest allele frequency when duplicated.
    if "_Allele_Frequency_num" in group.columns:
        group = group.sort_values("_Allele_Frequency_num", ascending=False, na_position="last")
    return group.iloc[0]


def main():
    final = pd.read_csv(FINAL, dtype=str).fillna("")
    s2_by_variant = load_matches(S2_MATCHES)
    s3_by_variant = load_matches(S3_MATCHES)

    final["_Variant_Key"] = final["Variante"].map(norm_variant)
    rows = []

    for key, group in final.groupby("_Variant_Key", sort=False):
        rep = choose_representative(group)
        s2 = s2_by_variant.get(key, {})
        s3 = s3_by_variant.get(key, {})
        integrated_label, integrated_source = integrated_classification(rep, s2, s3)

        ddg_7uuy = first_nonempty(group["DDG_7UUY"].tolist())
        ddg_7uuz = first_nonempty(group["DDG_7UUZ"].tolist())
        ddg_7uv0 = first_nonempty(group["DDG_7UV0"].tolist())
        ddg_af = first_nonempty(group["DDG_AF"].tolist())

        allele_values = [to_float(v) for v in group.get("Allele_Frequency", pd.Series(dtype=str)).tolist()]
        allele_values = [v for v in allele_values if v is not None]
        allele_frequency = max(allele_values) if allele_values else ""

        row = {
            "Variante": clean_text(rep["Variante"]),
            "Clasificacion_integrada": integrated_label,
            "Fuente_clasificacion": integrated_source,
            "Frecuencia_alelica": allele_frequency,
            "DDG_7UUY_apo": ddg_7uuy,
            "DDG_7UUZ_ReO4_Na": ddg_7uuz,
            "DDG_7UV0_I_Na": ddg_7uv0,
            "DDG_AF": ddg_af,
            "Clasificacion_estructural": clean_text(rep.get("Structural_Category_Consensus", "")),
            "ClinVar_Final_Classification": clean_text(rep.get("ClinVar_Final_Classification", "")),
            "ClinVar_Final_Source": clean_text(rep.get("ClinVar_Final_Source", "")),
            "PaperMariano_SuppTable2_ACMG_Classification": clean_text(s2.get("ACMG_Result_SAVY", "")),
            "PaperMariano_SuppTable2_ACMG_Criteria": clean_text(s2.get("ACMG_Criteria_SAVY", "")),
            "PaperMariano_SuppTable2_References": clean_text(s2.get("References_SAVY", "")),
            "PaperMariano_SuppTable3_Functional_Activity": clean_text(s3.get("Activity_num_SAVY_S3", "")),
            "PaperMariano_SuppTable3_Functional_Label": clean_text(s3.get("Functional_Label_From_Activity", "")),
            "PaperMariano_SuppTable3_References": clean_text(s3.get("References_SAVY_S3", "")),
        }
        rows.append(row)

    detailed = pd.DataFrame(rows)
    detailed["_pos"] = detailed["Variante"].map(protein_position)
    detailed = detailed.sort_values(["_pos", "Variante"]).drop(columns=["_pos"])

    legible_cols = [
        "Variante",
        "Clasificacion_integrada",
        "Fuente_clasificacion",
        "Frecuencia_alelica",
        "DDG_7UUY_apo",
        "DDG_7UUZ_ReO4_Na",
        "DDG_7UV0_I_Na",
        "DDG_AF",
        "Clasificacion_estructural",
    ]
    legible = detailed[legible_cols].copy()

    detailed.to_csv(OUT_DETALLADA, index=False)
    legible.to_csv(OUT_LEGIBLE, index=False)

    counts = legible["Clasificacion_integrada"].value_counts(dropna=False)
    source_counts = legible["Fuente_clasificacion"].value_counts(dropna=False)
    write_xlsx(detailed, legible, counts, source_counts)

    summary = [
        "# Integracion Paper Mariano/Nicola S2-S3",
        "",
        "Se generaron dos propuestas a partir de la tabla final FoldX/ClinVar/estructura.",
        "",
        "## Propuesta 1: trazable",
        "",
        f"Archivo: `{OUT_DETALLADA.name}`",
        "",
        "Incluye la clasificacion integrada y tambien las columnas separadas de ClinVar, PaperMariano Supplementary Table 2 (ACMG) y PaperMariano Supplementary Table 3 (actividad funcional). Es la tabla para auditar decisiones.",
        "",
        "## Propuesta 2: final legible",
        "",
        f"Archivo: `{OUT_LEGIBLE.name}`",
        "",
        "Incluye solamente: variante, clasificacion integrada, fuente de clasificacion, frecuencia alelica, DDG de las cuatro estructuras y clasificacion estructural consenso.",
        "",
        "## Regla de integracion",
        "",
        "1. Si ClinVar/NCBI/gnomAD ya tenia Pathogenic/Likely pathogenic o Benign/Likely benign, se conserva esa clasificacion.",
        "2. Si ClinVar era incierto/no clasificado/conflictivo y PaperMariano Supplementary Table 2 tenia ACMG informativo, se usa Table 2.",
        "3. Si no habia clasificacion clinica/ACMG informativa y PaperMariano Supplementary Table 3 tenia actividad funcional, se usa la etiqueta funcional.",
        "4. Si nada de lo anterior aplica, queda incierta/no clasificada.",
        "",
        "## Conteo por clasificacion integrada",
        "",
        counts_to_markdown(counts, "Clasificacion_integrada"),
        "",
        "## Conteo por fuente de clasificacion",
        "",
        counts_to_markdown(source_counts, "Fuente_clasificacion"),
        "",
        f"Total de variantes unicas: {len(legible)}",
    ]
    OUT_RESUMEN.write_text("\n".join(summary), encoding="utf-8")

    print(f"Variantes unicas: {len(legible)}")
    print("\nClasificacion integrada:")
    print(counts.to_string())
    print("\nFuente:")
    print(source_counts.to_string())
    print(f"\nEscrito:\n- {OUT_DETALLADA}\n- {OUT_LEGIBLE}\n- {OUT_XLSX}\n- {OUT_RESUMEN}")


if __name__ == "__main__":
    main()
