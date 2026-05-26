from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce")
INPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_NCBI_UNIPROT.csv"
OUTPUT = BASE / "TABLA_FOLDX_NIS_CON_CLINVAR_UNIPROT_LITERATURA.csv"
OUTPUT_SIMPLE = BASE / "TABLA_FOLDX_NIS_LITERATURA_SIMPLE.csv"
LIT_CSV = BASE / "EVIDENCIA_LITERATURA_SLC5A5_CURADA.csv"


def norm_variant(value):
    return str(value).strip().lower()


# Curated from public papers/reviews found during this run. Keep this file
# auditable: it stores source, kind of evidence, and a cautious interpretation.
LITERATURE = [
    {
        "Variante": "p.Val59Glu",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "reviewed functional disease variant",
        "Functional_Summary": "Reported NIS mutation associated with iodide transport defect/thyroid dyshormonogenesis in reviews.",
        "Frequency_Statement_From_Source": "",
        "Source": "Review of iodide transport defect mutations in SLC5A5/NIS",
        "PMID_or_DOI": "",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1219868/",
    },
    {
        "Variante": "p.Gly93Arg",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "natural variant; reviewed functional disease variant",
        "Functional_Summary": "TDH1/NIS disease variant reported in UniProt and reviews; associated with impaired iodide transport.",
        "Frequency_Statement_From_Source": "",
        "Source": "UniProt Q92911; NIS mutation reviews",
        "PMID_or_DOI": "PMID:9745458",
        "URL": "https://rest.uniprot.org/uniprotkb/Q92911",
    },
    {
        "Variante": "p.Arg124His",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "functional disease variant",
        "Functional_Summary": "NIS mutation reported with iodide transport defect; described in congenital iodide transport defect literature.",
        "Frequency_Statement_From_Source": "",
        "Source": "NIS mutation/iodide transport defect literature",
        "PMID_or_DOI": "",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1219868/",
    },
    {
        "Variante": "p.Gln263Leu",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "Reported with congenital hypothyroidism; in vitro assays showed reduced/altered NIS function.",
        "Frequency_Statement_From_Source": "",
        "Source": "Novel compound heterozygous pathogenic variants in SLC5A5 causing dyshormonogenetic congenital hypothyroidism",
        "PMID_or_DOI": "DOI:10.1111/cen.15189",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/?term=10.1111%2Fcen.15189",
    },
    {
        "Variante": "p.Gln267Glu",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "natural variant; reviewed functional disease variant",
        "Functional_Summary": "TDH1/NIS disease variant reported in UniProt and reviews; associated with impaired iodide transport.",
        "Frequency_Statement_From_Source": "",
        "Source": "UniProt Q92911; NIS mutation reviews",
        "PMID_or_DOI": "PMID:9486973",
        "URL": "https://rest.uniprot.org/uniprotkb/Q92911",
    },
    {
        "Variante": "p.Cys272*",
        "Literature_Label": "Pathogenic truncating",
        "Evidence_Type": "reviewed disease variant",
        "Functional_Summary": "Premature stop/truncating NIS mutation reported in iodide transport defect literature.",
        "Frequency_Statement_From_Source": "",
        "Source": "NIS mutation reviews",
        "PMID_or_DOI": "",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1219868/",
    },
    {
        "Variante": "p.Asp331Asn",
        "Literature_Label": "Likely pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "Reported NIS variant with functional assessment in congenital hypothyroidism/iodide transport defect context.",
        "Frequency_Statement_From_Source": "",
        "Source": "Identification and characterization of novel mutations in the SLC5A5 gene in a cohort of 26 Italian patients",
        "PMID_or_DOI": "PMID:31596074",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/31596074/",
    },
    {
        "Variante": "p.Gly350Asp",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "Reported with congenital hypothyroidism; in vitro assays showed reduced/altered NIS function.",
        "Frequency_Statement_From_Source": "",
        "Source": "Novel compound heterozygous pathogenic variants in SLC5A5 causing dyshormonogenetic congenital hypothyroidism",
        "PMID_or_DOI": "DOI:10.1111/cen.15189",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/?term=10.1111%2Fcen.15189",
    },
    {
        "Variante": "p.Thr354Pro",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "natural variant; reviewed functional disease variant",
        "Functional_Summary": "TDH1/NIS disease variant reported in UniProt and reviews; associated with impaired iodide transport.",
        "Frequency_Statement_From_Source": "",
        "Source": "UniProt Q92911; NIS mutation reviews",
        "PMID_or_DOI": "PMID:9171822; PMID:9745458",
        "URL": "https://rest.uniprot.org/uniprotkb/Q92911",
    },
    {
        "Variante": "p.Ser356Phe",
        "Literature_Label": "Likely pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "S356F showed markedly reduced iodide uptake in COS-7 cells and loss of plasma membrane localization.",
        "Frequency_Statement_From_Source": "Reported absent from gnomAD in the paper abstract.",
        "Source": "Brief Report: Novel Sodium/Iodide Symporter Mutation S356F Causes Congenital Hypothyroidism",
        "PMID_or_DOI": "DOI:10.1089/thy.2021.0478",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/?term=10.1089%2Fthy.2021.0478",
    },
    {
        "Variante": "p.Gly395Arg",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "natural variant; reviewed functional disease variant",
        "Functional_Summary": "TDH1/NIS disease variant reported in UniProt and reviews; associated with impaired iodide transport.",
        "Frequency_Statement_From_Source": "",
        "Source": "UniProt Q92911; NIS mutation reviews",
        "PMID_or_DOI": "PMID:10487695",
        "URL": "https://rest.uniprot.org/uniprotkb/Q92911",
    },
    {
        "Variante": "p.Gly421Arg",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "G421R and G51fs were studied in vitro; G421R caused impaired iodide transport and altered expression/localization.",
        "Frequency_Statement_From_Source": "",
        "Source": "Compound heterozygous loss-of-function mutations in SLC5A5 causing thyroid dyshormonogenesis",
        "PMID_or_DOI": "DOI:10.3389/fendo.2021.755988",
        "URL": "https://www.frontiersin.org/articles/10.3389/fendo.2021.755988/full",
    },
    {
        "Variante": "p.Tyr531*",
        "Literature_Label": "Pathogenic truncating",
        "Evidence_Type": "reviewed disease variant",
        "Functional_Summary": "Premature stop/truncating NIS mutation reported in iodide transport defect literature.",
        "Frequency_Statement_From_Source": "",
        "Source": "NIS mutation reviews",
        "PMID_or_DOI": "",
        "URL": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1219868/",
    },
    {
        "Variante": "p.Gly543Glu",
        "Literature_Label": "Pathogenic/functional defect",
        "Evidence_Type": "natural variant; reviewed functional disease variant",
        "Functional_Summary": "TDH1/NIS disease variant reported in UniProt and reviews; associated with impaired iodide transport.",
        "Frequency_Statement_From_Source": "",
        "Source": "UniProt Q92911; NIS mutation reviews",
        "PMID_or_DOI": "PMID:9745458",
        "URL": "https://rest.uniprot.org/uniprotkb/Q92911",
    },
    {
        "Variante": "p.Ser547Arg",
        "Literature_Label": "Likely pathogenic/functional defect",
        "Evidence_Type": "experimental functional characterization",
        "Functional_Summary": "Reported NIS variant with functional assessment in congenital hypothyroidism/iodide transport defect context.",
        "Frequency_Statement_From_Source": "",
        "Source": "Identification and characterization of novel mutations in the SLC5A5 gene in a cohort of 26 Italian patients",
        "PMID_or_DOI": "PMID:31596074",
        "URL": "https://pubmed.ncbi.nlm.nih.gov/31596074/",
    },
]


def main():
    df = pd.read_csv(INPUT)
    lit = pd.DataFrame(LITERATURE)
    lit["Variant_Key"] = lit["Variante"].map(norm_variant)
    df["Variant_Key"] = df["Variante"].map(norm_variant)

    merged = df.merge(
        lit.drop(columns=["Variante"]),
        on="Variant_Key",
        how="left",
    )
    merged["Literature_Match"] = merged["Literature_Label"].notna()
    merged.to_csv(OUTPUT, index=False)
    lit.drop(columns=["Variant_Key"]).to_csv(LIT_CSV, index=False)

    simple_cols = [
        "Variante",
        "ClinVar_Final_Classification",
        "Literature_Label",
        "Evidence_Type",
        "Functional_Summary",
        "Frequency_Statement_From_Source",
        "Source",
        "PMID_or_DOI",
        "URL",
        "Allele_Frequency",
        "DDG_7UUY",
        "DDG_7UUZ",
        "DDG_7UV0",
        "DDG_AF",
    ]
    merged[simple_cols].to_csv(OUTPUT_SIMPLE, index=False)

    print(f"Wrote: {OUTPUT}")
    print(f"Wrote: {OUTPUT_SIMPLE}")
    print(f"Wrote: {LIT_CSV}")
    print(f"Literature-curated variants: {len(lit)}")
    print(f"Rows in FoldX table matched to literature: {merged['Literature_Match'].sum()}")
    print(merged.loc[merged["Literature_Match"], simple_cols].to_string(index=False))


if __name__ == "__main__":
    main()
