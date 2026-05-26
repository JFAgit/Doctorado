from pathlib import Path
import re

import pandas as pd


BASE = Path(r"C:\Users\fran_\Documents\Doctorado")
WORK = BASE / "Inicios" / "Marce"
MARCE = BASE / "MarceNIS"
OLD_NIS_STRUCTURAL = WORK / "NIS" / "ClasificacionEstructural"
STRUCTURAL = MARCE / "AnalisisEstructural" / "residuos_clasificados.csv"

PDBS = {
    "AF": {
        "pdb": MARCE / "FoldX" / "EstructuraAlphaFold" / "AF-Q92911model_Repair.pdb",
        "classification": OLD_NIS_STRUCTURAL / "residuos_clasificados_AF_Human.csv",
        "has_real_membrane": False,
    },
    "7UUY": {
        "pdb": OLD_NIS_STRUCTURAL / "7uuyMembrane.pdb",
        "classification": OLD_NIS_STRUCTURAL / "residuos_clasificados_7uuy.csv",
        "has_real_membrane": True,
    },
    "7UUZ": {
        "pdb": OLD_NIS_STRUCTURAL / "7uuzMembrane.pdb",
        "classification": OLD_NIS_STRUCTURAL / "residuos_clasificados_7uuz.csv",
        "has_real_membrane": True,
    },
    "7UV0": {
        "pdb": OLD_NIS_STRUCTURAL / "7uv0Membrane.pdb",
        "classification": OLD_NIS_STRUCTURAL / "residuos_clasificados_7uv0.csv",
        "has_real_membrane": True,
    },
}

OUT_DIR = WORK / "PyMOL_coloreo_estructural"
OUT_DIR.mkdir(exist_ok=True)


def residue_pos(value):
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def norm_cat(value):
    text = str(value).strip().lower()
    if text in {"sitio activo", "active site", "sitio_activo"}:
        return "sitio_activo"
    if text == "core":
        return "core"
    if text in {"superficie", "surface"}:
        return "superficie"
    return text.replace(" ", "_")


def consensus(cats):
    cats = set(cats)
    if "sitio_activo" in cats:
        return "sitio_activo"
    if "core" in cats:
        return "core"
    if "superficie" in cats:
        return "superficie"
    return ""


def load_classification(path):
    local_df = pd.read_csv(path)
    cat_col = "Categoría" if "Categoría" in local_df.columns else "CategorÃ­a"
    local_df["pos"] = local_df["Residuo"].map(residue_pos)
    local_df["cat"] = local_df[cat_col].map(norm_cat)
    return local_df.dropna(subset=["pos"]).groupby("pos")["cat"].apply(consensus).reset_index()


def atom_bounds(path):
    xs, ys, zs = [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    pass
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def chunked(values, size=80):
    values = list(values)
    for i in range(0, len(values), size):
        yield values[i : i + size]


def write_pml(name, cfg):
    pdb_path = cfg["pdb"]
    class_path = cfg["classification"]
    grouped = load_classification(class_path)
    grouped["pos"] = grouped["pos"].astype(int)

    by_cat = {
        cat: sorted(grouped.loc[grouped["cat"].eq(cat), "pos"].tolist())
        for cat in ["superficie", "core", "sitio_activo"]
    }

    object_name = f"NIS_{name}"
    pml = []
    pml.append(f'load "{pdb_path.as_posix()}", {object_name}')
    pml.append("hide everything")
    pml.append(f"select protein_{name}, {object_name} and polymer.protein and chain A")
    pml.append(f"select membrane_{name}, {object_name} and not polymer.protein")
    pml.append(f"show cartoon, protein_{name}")
    pml.append(f"color gray80, protein_{name}")
    pml.append(f"show sticks, membrane_{name}")
    pml.append(f"color gray65, membrane_{name}")
    pml.append(f"set stick_radius, 0.08, membrane_{name}")
    pml.append(f"set transparency, 0.65, membrane_{name}")
    pml.append("set cartoon_transparency, 0.08")
    pml.append("set ray_opaque_background, off")
    pml.append("bg_color white")
    pml.append("set_color color_superficie, [0.25, 0.55, 0.95]")
    pml.append("set_color color_core, [0.95, 0.55, 0.10]")
    pml.append("set_color color_sitio_activo, [0.90, 0.05, 0.08]")
    pml.append("")

    if not cfg["has_real_membrane"]:
        xmin, xmax, ymin, ymax, _zmin, _zmax = atom_bounds(pdb_path)
        pad = 10
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad
        half_thickness = 15.2
        pml.append("from pymol.cgo import *")
        pml.append("from pymol import cmd")
        pml.append(
            "membrane_planes = ["
            "ALPHA, 0.18, COLOR, 0.55, 0.55, 0.55, BEGIN, TRIANGLES, "
            f"VERTEX, {xmin:.3f}, {ymin:.3f}, {half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymin:.3f}, {half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymax:.3f}, {half_thickness:.3f}, "
            f"VERTEX, {xmin:.3f}, {ymin:.3f}, {half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymax:.3f}, {half_thickness:.3f}, "
            f"VERTEX, {xmin:.3f}, {ymax:.3f}, {half_thickness:.3f}, END, "
            "ALPHA, 0.18, COLOR, 0.55, 0.55, 0.55, BEGIN, TRIANGLES, "
            f"VERTEX, {xmin:.3f}, {ymin:.3f}, {-half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymax:.3f}, {-half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymin:.3f}, {-half_thickness:.3f}, "
            f"VERTEX, {xmin:.3f}, {ymin:.3f}, {-half_thickness:.3f}, "
            f"VERTEX, {xmin:.3f}, {ymax:.3f}, {-half_thickness:.3f}, "
            f"VERTEX, {xmax:.3f}, {ymax:.3f}, {-half_thickness:.3f}, END]"
        )
        pml.append("cmd.load_cgo(membrane_planes, 'membrane_planes_AF')")
        pml.append("")

    labels = {
        "superficie": "Superficie",
        "core": "Core",
        "sitio_activo": "Sitio activo",
    }
    colors = {
        "superficie": "color_superficie",
        "core": "color_core",
        "sitio_activo": "color_sitio_activo",
    }
    for cat, residues in by_cat.items():
        if not residues:
            continue
        selections = []
        for n, chunk in enumerate(chunked(residues), start=1):
            sel_name = f"{cat}_{n}"
            resi_expr = "+".join(map(str, chunk))
            pml.append(f"select {sel_name}, protein_{name} and resi {resi_expr}")
            selections.append(sel_name)
        pml.append(f"select {cat}, " + " or ".join(selections))
        pml.append(f"color {colors[cat]}, {cat}")
        pml.append(f"show sticks, {cat}")
        if cat == "sitio_activo":
            pml.append(f"show spheres, {cat}")
            pml.append(f"set sphere_scale, 0.45, {cat}")
        pml.append(f"set stick_radius, 0.22, {cat}")
        pml.append(f"disable {' '.join(selections)}")
        pml.append("")

    pml.append("enable superficie")
    pml.append("enable core")
    pml.append("enable sitio_activo")
    pml.append("zoom all, 8")
    pml.append("orient")
    pml.append("")
    pml.append("# Legend:")
    pml.append("# superficie = blue")
    pml.append("# core = orange")
    pml.append("# sitio activo = red")
    pml.append("# membrane/lipids = translucent gray")
    pml.append(f"# classification source = {class_path}")
    pml.append(f"# structure source = {pdb_path}")
    pml.append(f'png "{(OUT_DIR / f"vista_coloreo_{name}.png").as_posix()}", width=1800, height=1400, dpi=300, ray=1')
    pml.append(f'save "{(OUT_DIR / f"sesion_coloreo_{name}.pse").as_posix()}"')

    out = OUT_DIR / f"colorear_estructural_{name}.pml"
    out.write_text("\n".join(pml) + "\n", encoding="utf-8")
    summary_rows = [
        {"Estructura": name, "Categoria": cat, "N_residuos": len(vals), "Residues": " ".join(map(str, vals))}
        for cat, vals in by_cat.items()
    ]
    return out, summary_rows


created = []
all_summary_rows = []
for name, cfg in PDBS.items():
    if cfg["pdb"].exists() and cfg["classification"].exists():
        out, rows = write_pml(name, cfg)
        created.append(out)
        all_summary_rows.extend(rows)

pd.DataFrame(all_summary_rows).to_csv(OUT_DIR / "resumen_residuos_pymol.csv", index=False)

print(f"Output dir: {OUT_DIR}")
for p in created:
    print(p)
print("Summary:")
print(pd.DataFrame(all_summary_rows).to_string(index=False))
