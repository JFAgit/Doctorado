from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
TABLE = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI_UNIPROT.csv"
CLINVAR = BASE / "ClinVar_NCBI_SLC5A5_parseado.csv"
OUT = BASE / "CANDIDATOS_CLINVAR_POR_RSID_NO_ASIGNADOS_AUTOMATICAMENTE.csv"


df = pd.read_csv(TABLE)
cv = pd.read_csv(CLINVAR)

rs_rows = []
for _, rec in cv.iterrows():
    for rsid in str(rec.get("ClinVar_rsIDs", "")).split(";"):
        if rsid and rsid != "nan":
            rs_rows.append((rsid, rec))

out = []
for _, row in df.iterrows():
    if str(row.get("ClinVar_Match_Method_NCBI", "")) != "nan" and str(row.get("ClinVar_Match_Method_NCBI", "")):
        continue
    if not isinstance(row.get("rsIDs"), str):
        continue
    row_rsids = [x for x in re.split(r"[;, ]+", row["rsIDs"]) if x]
    for rsid, rec in rs_rows:
        if rsid in row_rsids:
            out.append(
                {
                    "Variante": row.get("Variante", ""),
                    "gnomAD_ID": row.get("gnomAD_ID", ""),
                    "rsID": rsid,
                    "ClinVar_Candidate_Classification": rec.get("ClinVar_Classification_NCBI", ""),
                    "ClinVar_Candidate_Title": rec.get("ClinVar_Title", ""),
                    "ClinVar_Candidate_Variation_ID": rec.get("ClinVar_Variation_ID", ""),
                    "ClinVar_Candidate_URL": rec.get("ClinVar_URL", ""),
                    "Reason_Not_Automatically_Assigned": "rsID-only match; rsIDs can be multiallelic, so exact allele/protein/cDNA match is required before assigning.",
                }
            )

pd.DataFrame(out).drop_duplicates().to_csv(OUT, index=False)
print(OUT)
print(f"candidate rows: {len(out)}")
