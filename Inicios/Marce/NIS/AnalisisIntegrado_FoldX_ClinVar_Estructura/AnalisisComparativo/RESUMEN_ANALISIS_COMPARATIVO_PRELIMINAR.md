# Analisis comparativo preliminar NIS / FoldX

## Conteos
| Metric | Value |
| --- | --- |
| Rows | 888 |
| Clinical group: VUS_or_unclassified | 771 |
| Clinical group: Uncertain_significance | 78 |
| Clinical group: Pathogenic_or_likely | 26 |
| Clinical group: Benign_or_likely | 9 |
| Clinical group: Conflicting | 4 |
| Structural consensus: core | 697 |
| Structural consensus: superficie | 174 |
| Structural consensus: sitio activo | 17 |
| Most destabilizing experimental state: 7UUZ_ReO4_Na | 236 |
| Most destabilizing experimental state: no_experimental_ddg | 235 |
| Most destabilizing experimental state: 7UUY_apo | 223 |
| Most destabilizing experimental state: 7UV0_I_Na | 194 |

## DDG por grupo clinico
| Clinical_Group | Structure | Column | N | Mean | Median | Q1 | Q3 | Min | Max | Pct_DDG_gt_1 | Pct_DDG_gt_2 | Pct_DDG_gt_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign_or_likely | 7UUY_apo | DDG_7UUY | 4 | 0.268 | -0.076 | -0.385 | 0.577 | -1.087 | 2.308 | 25.000 | 25.000 | 0.000 |
| Benign_or_likely | 7UUZ_ReO4_Na | DDG_7UUZ | 4 | 3.952 | 1.222 | -0.003 | 5.177 | -0.011 | 13.377 | 50.000 | 50.000 | 25.000 |
| Benign_or_likely | 7UV0_I_Na | DDG_7UV0 | 4 | 1.526 | 1.448 | 0.144 | 2.829 | 0.067 | 3.141 | 50.000 | 50.000 | 25.000 |
| Benign_or_likely | AF_humano | DDG_AF | 9 | 0.404 | 0.366 | 0.058 | 0.784 | -0.440 | 1.254 | 11.111 | 0.000 | 0.000 |
| Conflicting | 7UUY_apo | DDG_7UUY | 3 | 0.015 | 0.134 | -0.128 | 0.218 | -0.391 | 0.301 | 0.000 | 0.000 | 0.000 |
| Conflicting | 7UUZ_ReO4_Na | DDG_7UUZ | 3 | 1.772 | 1.089 | 0.878 | 2.324 | 0.668 | 3.559 | 66.667 | 33.333 | 33.333 |
| Conflicting | 7UV0_I_Na | DDG_7UV0 | 3 | 0.745 | 0.317 | 0.289 | 0.987 | 0.261 | 1.657 | 33.333 | 0.000 | 0.000 |
| Conflicting | AF_humano | DDG_AF | 4 | 0.833 | 0.560 | -0.122 | 1.515 | -0.131 | 2.340 | 50.000 | 25.000 | 0.000 |
| Pathogenic_or_likely | 7UUY_apo | DDG_7UUY | 21 | 2.245 | 1.885 | 0.306 | 3.395 | -0.672 | 5.925 | 61.905 | 42.857 | 33.333 |
| Pathogenic_or_likely | 7UUZ_ReO4_Na | DDG_7UUZ | 20 | 2.406 | 1.099 | 0.108 | 5.541 | -0.610 | 7.455 | 55.000 | 35.000 | 35.000 |
| Pathogenic_or_likely | 7UV0_I_Na | DDG_7UV0 | 21 | 1.632 | 1.097 | 0.465 | 2.229 | -1.558 | 8.664 | 52.381 | 28.571 | 19.048 |
| Pathogenic_or_likely | AF_humano | DDG_AF | 26 | 6.412 | 3.912 | 2.030 | 8.226 | -0.770 | 32.337 | 84.615 | 76.923 | 61.538 |
| Uncertain_significance | 7UUY_apo | DDG_7UUY | 55 | 1.955 | 1.133 | 0.071 | 2.519 | -1.512 | 17.649 | 54.545 | 27.273 | 21.818 |
| Uncertain_significance | 7UUZ_ReO4_Na | DDG_7UUZ | 57 | 1.963 | 0.793 | -0.140 | 2.163 | -1.955 | 28.564 | 43.860 | 28.070 | 15.789 |
| Uncertain_significance | 7UV0_I_Na | DDG_7UV0 | 54 | 1.126 | 0.401 | -0.266 | 1.751 | -1.935 | 14.054 | 40.741 | 24.074 | 12.963 |
| Uncertain_significance | AF_humano | DDG_AF | 78 | 1.881 | 1.085 | 0.155 | 2.979 | -1.619 | 20.287 | 51.282 | 38.462 | 25.641 |
| VUS_or_unclassified | 7UUY_apo | DDG_7UUY | 549 | 2.067 | 1.045 | 0.019 | 2.532 | -2.948 | 34.971 | 51.548 | 32.787 | 20.947 |
| VUS_or_unclassified | 7UUZ_ReO4_Na | DDG_7UUZ | 547 | 2.346 | 1.328 | 0.011 | 3.226 | -2.968 | 31.186 | 57.038 | 38.757 | 27.605 |
| VUS_or_unclassified | 7UV0_I_Na | DDG_7UV0 | 545 | 1.946 | 0.990 | -0.005 | 2.769 | -3.497 | 27.927 | 49.725 | 33.761 | 22.385 |
| VUS_or_unclassified | AF_humano | DDG_AF | 771 | 1.979 | 1.054 | 0.109 | 2.589 | -3.143 | 62.181 | 50.713 | 33.593 | 21.141 |

## DDG por categoria estructural
| Structural_Category_Consensus | Structure | Column | N | Mean | Median | Q1 | Q3 | Min | Max | Pct_DDG_gt_1 | Pct_DDG_gt_2 | Pct_DDG_gt_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core | 7UUY_apo | DDG_7UUY | 597 | 2.058 | 1.045 | 0.009 | 2.537 | -2.948 | 34.971 | 51.424 | 32.328 | 21.273 |
| core | 7UUZ_ReO4_Na | DDG_7UUZ | 596 | 2.297 | 1.211 | 0.003 | 3.139 | -2.968 | 31.186 | 55.201 | 37.081 | 26.678 |
| core | 7UV0_I_Na | DDG_7UV0 | 592 | 1.857 | 0.943 | -0.006 | 2.677 | -3.497 | 27.927 | 48.480 | 32.770 | 21.284 |
| core | AF_humano | DDG_AF | 697 | 2.439 | 1.382 | 0.252 | 3.124 | -3.143 | 62.181 | 56.385 | 40.459 | 26.399 |
| sitio activo | 7UUY_apo | DDG_7UUY | 17 | 2.276 | 1.889 | 0.353 | 2.853 | -0.820 | 6.985 | 64.706 | 41.176 | 23.529 |
| sitio activo | 7UUZ_ReO4_Na | DDG_7UUZ | 17 | 3.712 | 2.914 | 0.936 | 5.060 | -0.011 | 12.438 | 70.588 | 64.706 | 47.059 |
| sitio activo | 7UV0_I_Na | DDG_7UV0 | 17 | 2.937 | 1.958 | 0.767 | 4.123 | -1.275 | 16.041 | 70.588 | 47.059 | 41.176 |
| sitio activo | AF_humano | DDG_AF | 17 | 1.027 | 1.242 | -0.082 | 1.688 | -0.674 | 3.896 | 52.941 | 23.529 | 5.882 |
| superficie | 7UUY_apo | DDG_7UUY | 18 | 1.279 | 0.991 | 0.501 | 2.040 | -1.296 | 4.377 | 50.000 | 27.778 | 16.667 |
| superficie | 7UUZ_ReO4_Na | DDG_7UUZ | 18 | 1.793 | 1.277 | 0.369 | 2.577 | -1.244 | 7.751 | 61.111 | 33.333 | 11.111 |
| superficie | 7UV0_I_Na | DDG_7UV0 | 18 | 0.824 | 0.364 | -0.004 | 1.508 | -0.617 | 3.441 | 44.444 | 16.667 | 5.556 |
| superficie | AF_humano | DDG_AF | 174 | 0.738 | 0.457 | 0.009 | 1.296 | -2.507 | 5.105 | 31.034 | 13.793 | 8.046 |

## Top 20 por DDG experimental maximo
| Variante | Clinical_Group | ClinVar_Final_Classification | Structural_Category_Consensus | Structural_Categories_By_Structure | DDG_7UUY | DDG_7UUZ | DDG_7UV0 | DDG_AF | Experimental_Mean_DDG | Experimental_Max_DDG | Conformational_Range_Experimental | Most_Destabilizing_Experimental_State | Allele_Frequency | Literature_Label | Functional_Summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p.Tyr324His | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 34.971 | 1.247 | -0.891 | 3.247 | 11.775 | 34.971 | 35.862 | 7UUY_apo | 0.000 |  |  |
| p.Pro77Leu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 1.047 | 31.186 | -0.796 | 5.086 | 10.479 | 31.186 | 31.981 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Ile284Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 30.668 | 1.020 | 7.535 | 0.326 | 13.075 | 30.668 | 29.648 | 7UUY_apo | 0.000 |  |  |
| p.Arg474Lys | Uncertain_significance | Uncertain significance | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | -0.069 | 28.564 | 14.054 | 0.632 | 14.183 | 28.564 | 28.633 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Pro326Leu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=superficie; 7UV0=core; AF=core | -1.215 | 2.126 | 27.927 | 0.762 | 9.613 | 27.927 | 29.143 | 7UV0_I_Na | 0.000 |  |  |
| p.Val445Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 27.538 | 1.085 | 0.094 | 13.280 | 9.573 | 27.538 | 27.444 | 7UUY_apo | 0.000 |  |  |
| p.Leu168Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 5.734 | 22.578 | 5.436 | 5.881 | 11.249 | 22.578 | 17.142 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Ala518Val | Uncertain_significance | Uncertain significance | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 2.875 | 22.129 | 0.000 | 0.159 | 8.335 | 22.129 | 22.129 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Val367Ile | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 1.194 | 1.394 | 22.109 | 0.316 | 8.233 | 22.109 | 20.915 | 7UV0_I_Na | 0.000 |  |  |
| p.Thr274Arg | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 21.880 | -1.271 | 1.619 | 2.810 | 7.410 | 21.880 | 23.151 | 7UUY_apo | 0.000 |  |  |
| p.Leu100Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=superficie; 7UV0=core; AF=core | 4.232 | -0.901 | 21.383 | 0.240 | 8.238 | 21.383 | 22.284 | 7UV0_I_Na | 0.000 |  |  |
| p.Arg482His | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 0.896 | 0.356 | 21.256 | 1.082 | 7.503 | 21.256 | 20.901 | 7UV0_I_Na | 0.000 |  |  |
| p.Thr101Ile | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=core; 7UV0=superficie; AF=core | -0.719 | 20.811 | 1.448 | 2.149 | 7.180 | 20.811 | 21.530 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Asp322Val | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | -0.117 | 19.553 | 4.411 | -0.633 | 7.949 | 19.553 | 19.670 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Arg474Met | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | -0.263 | 19.550 | 8.119 | 1.895 | 9.136 | 19.550 | 19.813 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Glu470Lys | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 1.239 | 18.588 | 19.225 | 1.256 | 13.017 | 19.225 | 17.986 | 7UV0_I_Na | 0.000 |  |  |
| p.Ala451Val | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 2.543 | 18.945 | 9.786 | 0.536 | 10.425 | 18.945 | 16.402 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Asp163Glu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 1.775 | 18.573 | 1.960 | 1.803 | 7.436 | 18.573 | 16.798 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Val195Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 18.468 | -0.107 | 0.025 | 1.339 | 6.129 | 18.468 | 18.575 | 7UUY_apo | 0.000 |  |  |
| p.Tyr324Asn | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 18.321 | 1.467 | -0.767 | 4.480 | 6.340 | 18.321 | 19.089 | 7UUY_apo | 0.000 |  |  |

## Top 20 por sensibilidad conformacional experimental
| Variante | Clinical_Group | ClinVar_Final_Classification | Structural_Category_Consensus | Structural_Categories_By_Structure | DDG_7UUY | DDG_7UUZ | DDG_7UV0 | DDG_AF | Experimental_Mean_DDG | Experimental_Max_DDG | Conformational_Range_Experimental | Most_Destabilizing_Experimental_State | Allele_Frequency | Literature_Label | Functional_Summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p.Tyr324His | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 34.971 | 1.247 | -0.891 | 3.247 | 11.775 | 34.971 | 35.862 | 7UUY_apo | 0.000 |  |  |
| p.Pro77Leu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 1.047 | 31.186 | -0.796 | 5.086 | 10.479 | 31.186 | 31.981 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Ile284Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 30.668 | 1.020 | 7.535 | 0.326 | 13.075 | 30.668 | 29.648 | 7UUY_apo | 0.000 |  |  |
| p.Pro326Leu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=superficie; 7UV0=core; AF=core | -1.215 | 2.126 | 27.927 | 0.762 | 9.613 | 27.927 | 29.143 | 7UV0_I_Na | 0.000 |  |  |
| p.Arg474Lys | Uncertain_significance | Uncertain significance | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | -0.069 | 28.564 | 14.054 | 0.632 | 14.183 | 28.564 | 28.633 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Val445Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 27.538 | 1.085 | 0.094 | 13.280 | 9.573 | 27.538 | 27.444 | 7UUY_apo | 0.000 |  |  |
| p.Thr274Arg | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 21.880 | -1.271 | 1.619 | 2.810 | 7.410 | 21.880 | 23.151 | 7UUY_apo | 0.000 |  |  |
| p.Leu100Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=superficie; 7UV0=core; AF=core | 4.232 | -0.901 | 21.383 | 0.240 | 8.238 | 21.383 | 22.284 | 7UV0_I_Na | 0.000 |  |  |
| p.Ala518Val | Uncertain_significance | Uncertain significance | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 2.875 | 22.129 | 0.000 | 0.159 | 8.335 | 22.129 | 22.129 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Thr101Ile | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=superficie; 7UUZ=core; 7UV0=superficie; AF=core | -0.719 | 20.811 | 1.448 | 2.149 | 7.180 | 20.811 | 21.530 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Val367Ile | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 1.194 | 1.394 | 22.109 | 0.316 | 8.233 | 22.109 | 20.915 | 7UV0_I_Na | 0.000 |  |  |
| p.Arg482His | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 0.896 | 0.356 | 21.256 | 1.082 | 7.503 | 21.256 | 20.901 | 7UV0_I_Na | 0.000 |  |  |
| p.Arg474Met | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | -0.263 | 19.550 | 8.119 | 1.895 | 9.136 | 19.550 | 19.813 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Asp322Val | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | -0.117 | 19.553 | 4.411 | -0.633 | 7.949 | 19.553 | 19.670 | 7UUZ_ReO4_Na | 0.000 |  |  |
| p.Tyr324Asn | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 18.321 | 1.467 | -0.767 | 4.480 | 6.340 | 18.321 | 19.089 | 7UUY_apo | 0.000 |  |  |
| p.Val195Phe | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=core | 18.468 | -0.107 | 0.025 | 1.339 | 6.129 | 18.468 | 18.575 | 7UUY_apo | 0.000 |  |  |
| p.Pro468Leu | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 16.574 | -0.009 | -1.859 | 4.044 | 4.902 | 16.574 | 18.433 | 7UUY_apo | 0.000 |  |  |
| p.Arg376Trp | Uncertain_significance | Uncertain significance | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 17.649 | 3.544 | -0.602 | 0.038 | 6.864 | 17.649 | 18.251 | 7UUY_apo | 0.000 |  |  |
| p.Glu470Lys | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 1.239 | 18.588 | 19.225 | 1.256 | 13.017 | 19.225 | 17.986 | 7UV0_I_Na | 0.000 |  |  |
| p.Pro517Thr | VUS_or_unclassified | Uncertain significance / not classified in gnomAD extract | core | 7UUY=core; 7UUZ=core; 7UV0=core; AF=superficie | 18.317 | 0.347 | 2.248 | 1.217 | 6.970 | 18.317 | 17.970 | 7UUY_apo | 0.000 |  |  |