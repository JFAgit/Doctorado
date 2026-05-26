from pathlib import Path
import json
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
INPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI.csv"
UNIPROT_JSON = BASE / "uniprot_Q92911.json"
OUTPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI_UNIPROT.csv"
OUTPUT_SIMPLE = BASE / "TABLA_FOLDX_NIS_CLASIFICACION_EXTENDIDA_SIMPLE.csv"
UNIPROT_PARSED = BASE / "UniProt_Q92911_variantes_parseadas.csv"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def protein_one_letter_key(value):
    if pd.isna(value):
        return ""
    text = str(value)
    match = re.search(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*|\?)", text)
    if not match:
        return ""
    wt3, pos, mut3 = match.groups()
    wt = AA3_TO_1.get(wt3.upper(), "X")
    mut = AA3_TO_1.get(mut3.upper(), "X")
    return f"{wt}{pos}{mut}"


def parse_uniprot():
    data = json.loads(UNIPROT_JSON.read_text(encoding="utf-8"))
    rows = []
    for feature in data.get("features", []):
        if feature.get("type") != "Natural variant":
            continue
        loc = feature.get("location", {})
        pos = loc.get("start", {}).get("value")
        alt = feature.get("alternativeSequence", {})
        wt = alt.get("originalSequence", "")
        alts = alt.get("alternativeSequences", []) or []
        xrefs = feature.get("featureCrossReferences", []) or []
        rsids = []
        for x in xrefs:
            if str(x.get("database", "")).lower() == "dbsnp":
                rsids.append(x.get("id", ""))
        pubmed = []
        for ev in feature.get("evidences", []) or []:
            if ev.get("source") == "PubMed" and ev.get("id"):
                pubmed.append(ev["id"])
        for mut in alts:
            rows.append(
                {
                    "UniProt_Feature_ID": feature.get("featureId", ""),
                    "UniProt_Pos": pos,
                    "UniProt_WT": wt,
                    "UniProt_MUT": mut,
                    "UniProt_Key": f"{wt}{pos}{mut}",
                    "UniProt_Description": feature.get("description", ""),
                    "UniProt_rsIDs": ";".join(sorted(set(rsids))),
                    "UniProt_PubMed": ";".join(sorted(set(pubmed))),
                    "UniProt_URL": f"https://web.expasy.org/variant_pages/{feature.get('featureId', '')}.html",
                }
            )
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT)
    uv = parse_uniprot()
    uv.to_csv(UNIPROT_PARSED, index=False)

    by_key = {k: v.iloc[0] for k, v in uv.groupby("UniProt_Key") if k}
    new_rows = []
    for _, row in df.iterrows():
        key = protein_one_letter_key(row.get("Protein_Consequence"))
        hit = by_key.get(key)
        match_method = "protein_change" if hit is not None else ""

        out = row.to_dict()
        if hit is not None:
            for col in [
                "UniProt_Feature_ID",
                "UniProt_Description",
                "UniProt_rsIDs",
                "UniProt_PubMed",
                "UniProt_URL",
            ]:
                out[col] = hit.get(col, "")
            out["UniProt_Match_Method"] = match_method
            out["UniProt_Disease_Annotated"] = "TDH1" in str(hit.get("UniProt_Description", ""))
        else:
            for col in [
                "UniProt_Feature_ID",
                "UniProt_Description",
                "UniProt_rsIDs",
                "UniProt_PubMed",
                "UniProt_URL",
            ]:
                out[col] = ""
            out["UniProt_Match_Method"] = ""
            out["UniProt_Disease_Annotated"] = False
        new_rows.append(out)

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(OUTPUT, index=False)

    simple_cols = [
        "Variante",
        "ClinVar_Final_Classification",
        "ClinVar_Final_Source",
        "ClinVar_Classification_NCBI",
        "ClinVar_Review_Status_NCBI",
        "ClinVar_Variation_ID",
        "ClinVar_URL",
        "UniProt_Feature_ID",
        "UniProt_Description",
        "UniProt_PubMed",
        "DDG_7UUY",
        "DDG_7UUZ",
        "DDG_7UV0",
        "DDG_AF",
    ]
    out_df[simple_cols].to_csv(OUTPUT_SIMPLE, index=False)

    print(f"Wrote: {OUTPUT}")
    print(f"Wrote: {OUTPUT_SIMPLE}")
    print(f"Wrote: {UNIPROT_PARSED}")
    print(f"UniProt natural variants parsed: {len(uv)}")
    print(f"Rows matched to UniProt: {(out_df['UniProt_Match_Method'] != '').sum()}")
    print(out_df["ClinVar_Final_Classification"].fillna("").value_counts().to_string())


if __name__ == "__main__":
    main()
