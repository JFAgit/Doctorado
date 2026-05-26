from pathlib import Path
import html

import pandas as pd


INPUT = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\TABLA_FOLDX_NIS_SIMPLE.csv")
OUTPUT = Path(r"C:\Users\fran_\Documents\Doctorado\Inicios\Marce\histograma_frecuencias_ClinVar_NIS.svg")


df = pd.read_csv(INPUT)
order = [
    "VUS / Sin Clasificar",
    "Uncertain significance",
    "Likely Benign",
    "Benign",
    "Likely Pathogenic",
    "Pathogenic",
]

counts = df["ClinVar"].fillna("Sin dato").value_counts()
labels = [x for x in order if x in counts.index] + [x for x in counts.index if x not in order]
values = [int(counts[x]) for x in labels]

width = 1100
height = 720
margin_left = 95
margin_right = 45
margin_top = 80
margin_bottom = 190
plot_width = width - margin_left - margin_right
plot_height = height - margin_top - margin_bottom
max_value = max(values)
colors = ["#8f969e", "#f2c14e", "#6ab7e8", "#2f80ed", "#ef8354", "#d62828", "#777777"]
bar_gap = 22
bar_width = (plot_width - bar_gap * (len(values) - 1)) / len(values)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<text x="550" y="38" text-anchor="middle" font-family="Arial" font-size="25" font-weight="700">'
    "Frecuencia de variantes NIS por clasificacion ClinVar</text>",
    f'<text x="550" y="65" text-anchor="middle" font-family="Arial" font-size="14" fill="#555">n = {len(df)} variantes</text>',
]

tick_step = 100
for tick in range(0, max_value + tick_step, tick_step):
    y = margin_top + plot_height - (tick / max_value) * plot_height
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
    bar_height = (value / max_value) * plot_height
    y = margin_top + plot_height - bar_height
    cx = x + bar_width / 2
    label_y = margin_top + plot_height + 22
    color = colors[i % len(colors)]
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
        f'fill="{color}" stroke="#333" stroke-width="0.7"/>'
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
parts.append("</svg>")

OUTPUT.write_text("\n".join(parts), encoding="utf-8")

print(OUTPUT)
for label, value in zip(labels, values):
    print(f"{label}: {value}")
