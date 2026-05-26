from pathlib import Path
import html

import pandas as pd


INPUT = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\TABLA_MAESTRA_FOLDX_NIS_RECONSTRUIDA.csv")
OUTPUT_SVG = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\histograma_frecuencia_alelica_NIS.svg")
OUTPUT_BINS = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\histograma_frecuencia_alelica_NIS_bins.csv")


df = pd.read_csv(INPUT)
af = pd.to_numeric(df["Allele_Frequency"], errors="coerce").dropna()

bins = [
    (0, 1e-6, "<1e-6"),
    (1e-6, 2e-6, "1e-6 a 2e-6"),
    (2e-6, 5e-6, "2e-6 a 5e-6"),
    (5e-6, 1e-5, "5e-6 a 1e-5"),
    (1e-5, 5e-5, "1e-5 a 5e-5"),
    (5e-5, 1e-4, "5e-5 a 1e-4"),
    (1e-4, 1e-3, "1e-4 a 1e-3"),
    (1e-3, float("inf"), ">=1e-3"),
]

rows = []
for low, high, label in bins:
    if high == float("inf"):
        count = int((af >= low).sum())
    else:
        count = int(((af >= low) & (af < high)).sum())
    rows.append({"AF_bin": label, "lower_bound": low, "upper_bound": high, "count": count})

bin_df = pd.DataFrame(rows)
bin_df.to_csv(OUTPUT_BINS, index=False)

labels = bin_df["AF_bin"].tolist()
values = bin_df["count"].tolist()

width = 1200
height = 720
margin_left = 95
margin_right = 45
margin_top = 90
margin_bottom = 185
plot_width = width - margin_left - margin_right
plot_height = height - margin_top - margin_bottom
max_value = max(values)
bar_gap = 18
bar_width = (plot_width - bar_gap * (len(values) - 1)) / len(values)
colors = ["#52796f", "#84a98c", "#cad2c5", "#f2cc8f", "#f4a261", "#e76f51", "#b56576", "#6d597a"]

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="600" y="38" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700">'
    "Distribucion de frecuencias alelicas de variantes NIS</text>",
    f'<text x="600" y="65" text-anchor="middle" font-family="Arial" font-size="14" fill="#555">'
    f"gnomAD AF, n = {len(af)} variantes con frecuencia numerica</text>",
]

tick_step = 50
upper_tick = ((max_value + tick_step - 1) // tick_step) * tick_step
for tick in range(0, upper_tick + 1, tick_step):
    y = margin_top + plot_height - (tick / upper_tick) * plot_height
    parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#e7e7e7"/>')
    parts.append(
        f'<text x="{margin_left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#555">{tick}</text>'
    )

parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_height}" stroke="#333"/>')
parts.append(
    f'<line x1="{margin_left}" y1="{margin_top+plot_height}" x2="{width-margin_right}" y2="{margin_top+plot_height}" stroke="#333"/>'
)

for i, (label, value) in enumerate(zip(labels, values)):
    x = margin_left + i * (bar_width + bar_gap)
    bar_height = (value / upper_tick) * plot_height
    y = margin_top + plot_height - bar_height
    cx = x + bar_width / 2
    label_y = margin_top + plot_height + 24
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
        f'fill="{colors[i % len(colors)]}" stroke="#333" stroke-width="0.7"/>'
    )
    parts.append(
        f'<text x="{cx:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{value}</text>'
    )
    parts.append(
        f'<text x="{cx:.1f}" y="{label_y}" text-anchor="end" transform="rotate(-35 {cx:.1f} {label_y})" '
        f'font-family="Arial" font-size="14">{html.escape(label)}</text>'
    )

y_mid = margin_top + plot_height / 2
parts.append(
    f'<text x="32" y="{y_mid}" text-anchor="middle" transform="rotate(-90 32 {y_mid})" '
    'font-family="Arial" font-size="16" font-weight="600">Numero de variantes</text>'
)
parts.append(
    f'<text x="{margin_left + plot_width / 2}" y="{height - 24}" text-anchor="middle" '
    'font-family="Arial" font-size="15" font-weight="600">Rango de frecuencia alelica (AF, escala logaritmica por bins)</text>'
)
parts.append("</svg>")

OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")

print(OUTPUT_SVG)
print(OUTPUT_BINS)
print(bin_df[["AF_bin", "count"]].to_string(index=False))
