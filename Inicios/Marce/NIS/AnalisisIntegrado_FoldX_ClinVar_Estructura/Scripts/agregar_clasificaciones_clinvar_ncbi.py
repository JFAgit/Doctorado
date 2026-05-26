from pathlib import Path
import json
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
INPUT = BASE / "TABLA_MAESTRA_FOLDX_NIS_RECONSTRUIDA.csv"
CLINVAR_JSONS = [
    BASE / "clinvar_slc5a5_esummary_0_500.json",
    BASE / "clinvar_slc5a5_esummary_500_1000.json",
]

OUTPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI.csv"
CLINVAR_PARSED = BASE / "ClinVar_NCBI_SLC5A5_parseado.csv"
SUMMARY = BASE / "RESUMEN_CRUCE_CLINVAR_NCBI.txt"


def normalize_classification(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value == "VUS / Sin Clasificar":
        return "Uncertain significance / not classified in gnomAD extract"
    return value


def parse_gnomad_id(value):
    if pd.isna(value):
        return ""
    parts = str(value).strip().split("-")
    if len(parts) != 4:
        return ""
    chrom, pos, ref, alt = parts
    return f"{chrom}:{pos}:{ref}:{alt}".upper()


def parse_spdi(value):
    if not value:
        return ""
    parts = str(value).split(":")
    if len(parts) != 4:
        return ""
    seq, zero_based_pos, ref, alt = parts
    if not seq.startswith("NC_000019"):
        return ""
    try:
        one_based_pos = int(zero_based_pos) + 1
    except ValueError:
        return ""
    return f"19:{one_based_pos}:{ref}:{alt}".upper()


def protein_key(value):
    if pd.isna(value):
        return ""
    text = str(value)
    match = re.search(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*|\?)", text)
    if not match:
        return ""
    aa1, pos, aa2 = match.groups()
    return f"p.{aa1.capitalize()}{pos}{aa2.capitalize()}"


def cdna_short(value):
    if pd.isna(value):
        return ""
    text = str(value)
    match = re.search(r"(c\.[A-Za-z0-9_+\-*?>=]+)", text)
    return match.group(1) if match else ""


def trait_names(germline):
    traits = germline.get("trait_set") or []
    return "; ".join(t.get("trait_name", "") for t in traits if t.get("trait_name"))


def parse_clinvar_records():
    records = []
    for path in CLINVAR_JSONS:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data.get("result", {})
        for uid in result.get("uids", []):
            rec = result.get(uid, {})
            germline = rec.get("germline_classification") or {}
            variation_set = rec.get("variation_set") or []
            first_var = variation_set[0] if variation_set else {}
            canonical_spdi = first_var.get("canonical_spdi", "")
            variation_xrefs = first_var.get("variation_xrefs") or []
            rsids = []
            for xref in variation_xrefs:
                db = str(xref.get("db_source", "")).lower()
                x_id = str(xref.get("db_id", ""))
                if db in {"dbsnp", "dbsnp"} and x_id:
                    rsids.append(x_id if x_id.startswith("rs") else f"rs{x_id}")

            title = rec.get("title", "")
            protein = rec.get("protein_change", "") or protein_key(title)
            cdna = first_var.get("cdna_change", "") or title
            records.append(
                {
                    "ClinVar_Variation_ID": uid,
                    "ClinVar_Accession": rec.get("accession_version", rec.get("accession", "")),
                    "ClinVar_Title": title,
                    "ClinVar_Classification_NCBI": germline.get("description", ""),
                    "ClinVar_Review_Status_NCBI": germline.get("review_status", ""),
                    "ClinVar_Last_Evaluated_NCBI": germline.get("last_evaluated", ""),
                    "ClinVar_Trait_NCBI": trait_names(germline),
                    "ClinVar_Molecular_Consequence_NCBI": "; ".join(rec.get("molecular_consequence_list") or []),
                    "ClinVar_Protein_Change_NCBI": protein,
                    "ClinVar_Protein_Key": protein_key(protein or title),
                    "ClinVar_cDNA_NCBI": cdna,
                    "ClinVar_cDNA_Key": cdna_short(cdna),
                    "ClinVar_Canonical_SPDI": canonical_spdi,
                    "ClinVar_Genomic_Key": parse_spdi(canonical_spdi),
                    "ClinVar_rsIDs": ";".join(sorted(set(rsids))),
                    "ClinVar_URL": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/",
                }
            )
    return pd.DataFrame(records)


def best_record(records):
    if records.empty:
        return None

    priority = {
        "Pathogenic": 1,
        "Likely pathogenic": 2,
        "Pathogenic/Likely pathogenic": 2,
        "Likely benign": 3,
        "Benign": 4,
        "Benign/Likely benign": 4,
        "Uncertain significance": 5,
        "Conflicting classifications of pathogenicity": 6,
        "not provided": 9,
        "": 10,
    }
    tmp = records.copy()
    tmp["_priority"] = tmp["ClinVar_Classification_NCBI"].map(priority).fillna(7)
    tmp["_review_len"] = tmp["ClinVar_Review_Status_NCBI"].fillna("").str.len()
    return tmp.sort_values(["_priority", "_review_len"], ascending=[True, False]).iloc[0]


def main():
    source = pd.read_csv(INPUT)
    ddg_cols = ["DDG_7UUY", "DDG_7UUZ", "DDG_7UV0", "DDG_AF"]
    source = source[source[ddg_cols].notna().any(axis=1)].copy()
    source["gnomAD_Genomic_Key"] = source["gnomAD_ID"].map(parse_gnomad_id)
    source["Protein_Key"] = source["Protein_Consequence"].map(protein_key)
    source["cDNA_Key"] = source["HGVS_Consequence"].map(cdna_short)

    clinvar = parse_clinvar_records()
    clinvar.to_csv(CLINVAR_PARSED, index=False)

    genomic_groups = {k: v for k, v in clinvar.groupby("ClinVar_Genomic_Key") if k}
    protein_groups = {k: v for k, v in clinvar.groupby("ClinVar_Protein_Key") if k}
    cdna_groups = {k: v for k, v in clinvar.groupby("ClinVar_cDNA_Key") if k}
    out_rows = []
    for _, row in source.iterrows():
        match_method = ""
        candidates = pd.DataFrame()

        if row["gnomAD_Genomic_Key"] and row["gnomAD_Genomic_Key"] in genomic_groups:
            candidates = genomic_groups[row["gnomAD_Genomic_Key"]]
            match_method = "genomic_key"
        elif row["cDNA_Key"] and row["cDNA_Key"] in cdna_groups:
            candidates = cdna_groups[row["cDNA_Key"]]
            match_method = "cDNA"
        elif row["Protein_Key"] and row["Protein_Key"] in protein_groups:
            candidates = protein_groups[row["Protein_Key"]]
            match_method = "protein_change"

        best = best_record(candidates)
        new = row.to_dict()
        new["ClinVar_gnomAD_original"] = normalize_classification(row.get("ClinVar"))
        if best is not None:
            for col in [
                "ClinVar_Variation_ID",
                "ClinVar_Accession",
                "ClinVar_Title",
                "ClinVar_Classification_NCBI",
                "ClinVar_Review_Status_NCBI",
                "ClinVar_Last_Evaluated_NCBI",
                "ClinVar_Trait_NCBI",
                "ClinVar_Molecular_Consequence_NCBI",
                "ClinVar_Protein_Change_NCBI",
                "ClinVar_cDNA_NCBI",
                "ClinVar_Canonical_SPDI",
                "ClinVar_URL",
            ]:
                new[col] = best.get(col, "")
            new["ClinVar_Match_Method_NCBI"] = match_method
            new["ClinVar_Match_Count_NCBI"] = len(candidates)
            new["ClinVar_Final_Classification"] = best.get("ClinVar_Classification_NCBI", "") or normalize_classification(row.get("ClinVar"))
            new["ClinVar_Final_Source"] = "NCBI ClinVar"
        else:
            for col in [
                "ClinVar_Variation_ID",
                "ClinVar_Accession",
                "ClinVar_Title",
                "ClinVar_Classification_NCBI",
                "ClinVar_Review_Status_NCBI",
                "ClinVar_Last_Evaluated_NCBI",
                "ClinVar_Trait_NCBI",
                "ClinVar_Molecular_Consequence_NCBI",
                "ClinVar_Protein_Change_NCBI",
                "ClinVar_cDNA_NCBI",
                "ClinVar_Canonical_SPDI",
                "ClinVar_URL",
            ]:
                new[col] = ""
            new["ClinVar_Match_Method_NCBI"] = ""
            new["ClinVar_Match_Count_NCBI"] = 0
            new["ClinVar_Final_Classification"] = normalize_classification(row.get("ClinVar"))
            new["ClinVar_Final_Source"] = "gnomAD extract"
        out_rows.append(new)

    out = pd.DataFrame(out_rows)

    preferred = [
        "Variante",
        "ClinVar_Final_Classification",
        "ClinVar_Final_Source",
        "ClinVar_gnomAD_original",
        "ClinVar_Classification_NCBI",
        "ClinVar_Review_Status_NCBI",
        "ClinVar_Last_Evaluated_NCBI",
        "ClinVar_Match_Method_NCBI",
        "ClinVar_Match_Count_NCBI",
        "ClinVar_Variation_ID",
        "ClinVar_Accession",
        "ClinVar_Trait_NCBI",
        "ClinVar_URL",
        "DDG_7UUY",
        "DDG_7UUZ",
        "DDG_7UV0",
        "DDG_AF",
        "Allele_Frequency",
        "gnomAD_ID",
        "rsIDs",
        "Protein_Consequence",
        "HGVS_Consequence",
        "Variant_Type",
        "FoldX_Applicable",
        "FoldX_Caution",
    ]
    remaining = [c for c in out.columns if c not in preferred]
    out = out[preferred + remaining]
    out.to_csv(OUTPUT, index=False)

    summary_lines = []
    summary_lines.append(f"Input variants with any FoldX DDG: {len(source)}")
    summary_lines.append(f"ClinVar SLC5A5 records downloaded from NCBI: {len(clinvar)}")
    summary_lines.append(f"Rows matched to NCBI ClinVar: {(out['ClinVar_Match_Count_NCBI'] > 0).sum()}")
    summary_lines.append("")
    summary_lines.append("Final classification counts:")
    summary_lines.append(out["ClinVar_Final_Classification"].fillna("").value_counts().to_string())
    summary_lines.append("")
    summary_lines.append("NCBI match methods:")
    summary_lines.append(out["ClinVar_Match_Method_NCBI"].replace("", "no_match").value_counts().to_string())
    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote: {OUTPUT}")
    print(f"Wrote: {CLINVAR_PARSED}")
    print(f"Wrote: {SUMMARY}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
