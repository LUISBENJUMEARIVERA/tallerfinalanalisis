import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
import prince
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig_dir = Path("figuras")
fig_dir.mkdir(exist_ok=True)
tab_dir = Path("resultados")
tab_dir.mkdir(exist_ok=True)

# 1. Load & prepare data
cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
df_raw = pd.read_csv("heart_cleveland_raw.csv", names=cols, na_values="?")
df = df_raw.dropna().copy().reset_index(drop=True)

df["sex"] = df["sex"].map({0.0: "Femenino", 1.0: "Masculino"}).astype("category")
df["cp"] = df["cp"].map({
    1.0: "Angina_tipica",
    2.0: "Angina_atipica",
    3.0: "Dolor_no_anginoso",
    4.0: "Asintomatico"
}).astype("category")
df["fbs"] = df["fbs"].map({0.0: "Gluc_<=120", 1.0: "Gluc_>120"}).astype("category")
df["restecg"] = df["restecg"].map({
    0.0: "ECG_Normal",
    1.0: "ECG_Anomalia_STT",
    2.0: "ECG_Hipertrofia_VI"
}).astype("category")
df["exang"] = df["exang"].map({0.0: "Exang_No", 1.0: "Exang_Si"}).astype("category")
df["slope"] = df["slope"].map({
    1.0: "Pend_Ascendente",
    2.0: "Pend_Plana",
    3.0: "Pend_Descendente"
}).astype("category")
df["ca"] = df["ca"].astype(int).astype(str).map({
    "0": "Vasos_0", "1": "Vasos_1", "2": "Vasos_2", "3": "Vasos_3"
}).astype("category")
df["thal"] = df["thal"].map({
    3.0: "Thal_Normal",
    6.0: "Thal_Defecto_Fijo",
    7.0: "Thal_Defecto_Rev"
}).astype("category")
df["diagnostico"] = np.where(df["num"] == 0, "Sano", "Enfermedad_Cardiaca")
df["diagnostico"] = df["diagnostico"].astype("category")

groups = {
    "G1_Basal": ["age", "trestbps", "chol"],
    "G2_Esfuerzo_Num": ["thalach", "oldpeak"],
    "G3_Clinica_Cat": ["sex", "cp", "fbs", "restecg"],
    "G4_Prueba_Cat": ["exang", "slope", "ca", "thal"],
    "G5_Diagnostico_Sup": ["diagnostico"],
}

group_labels_es = {
    "G1_Basal": "G1: Basal (Cuant.)",
    "G2_Esfuerzo_Num": "G2: Esfuerzo (Cuant.)",
    "G3_Clinica_Cat": "G3: Clínica (Cual.)",
    "G4_Prueba_Cat": "G4: Prueba/Angio (Cual.)",
    "G5_Diagnostico_Sup": "G5: Diagnóstico (Supl.)",
}

data_dict = {(grp, v): df[v] for grp, vs in groups.items() for v in vs}
X_mfa = pd.DataFrame(data_dict)
X_mfa.columns = pd.MultiIndex.from_tuples(X_mfa.columns, names=["group", "variable"])

mfa = prince.MFA(
    n_components=10,
    n_iter=10,
    rescale_with_mean=True,
    rescale_with_std=True,
    random_state=42,
).fit(
    X_mfa,
    groups=list(groups.keys()),
    supplementary_groups=["G5_Diagnostico_Sup"],
)

eig_df = mfa.eigenvalues_summary.copy()
eig_df.to_csv(tab_dir / "eigenvalues.csv")

row_coords = mfa.row_coordinates(X_mfa)
col_coords = mfa.column_coordinates_
col_contrib = mfa.column_contributions_
col_cos2 = mfa.column_cosine_similarities_
row_contrib = mfa.row_contributions_
row_cos2 = mfa.row_cosine_similarities(X_mfa)
partial_coords = mfa.partial_row_coordinates(X_mfa)

# Save quantitative variable correlations with Dim 1-3
col_corr = mfa.column_correlations
col_corr[[0, 1, 2]].to_csv(tab_dir / "quant_correlations.csv")

# Save group summary
grp_summary = pd.DataFrame({
    "Dim1_coord": mfa.group_coordinates_[0],
    "Dim2_coord": mfa.group_coordinates_[1],
    "Dim3_coord": mfa.group_coordinates_[2],
    "Dim1_contrib_pct": mfa.group_contributions_[0] * 100,
    "Dim2_contrib_pct": mfa.group_contributions_[1] * 100,
    "Dim3_contrib_pct": mfa.group_contributions_[2] * 100,
    "Dim1_cos2": mfa.group_cosine_similarities_[0],
    "Dim2_cos2": mfa.group_cosine_similarities_[1],
    "Dim3_cos2": mfa.group_cosine_similarities_[2],
})
grp_summary.to_csv(tab_dir / "group_summary.csv")

# Save full column table
col_table = pd.DataFrame({
    "Dim1_coord": col_coords[0],
    "Dim2_coord": col_coords[1],
    "Dim3_coord": col_coords[2],
    "Dim1_contrib_pct": col_contrib[0] * 100,
    "Dim2_contrib_pct": col_contrib[1] * 100,
    "Dim3_contrib_pct": col_contrib[2] * 100,
    "Dim1_cos2": col_cos2[0],
    "Dim2_cos2": col_cos2[1],
    "Dim3_cos2": col_cos2[2],
    "Cos2_sum_3D": col_cos2[[0, 1, 2]].sum(axis=1),
})
col_table.to_csv(tab_dir / "column_summary.csv")

# Descriptive stats for numerical vars
num_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]
num_desc = df[num_vars].agg(["count", "mean", "std", "min", "median", "max"]).T
num_desc.to_csv(tab_dir / "numeric_descriptives.csv")

# ---- FIGURE 1: Scree Plot ----
fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
dims = np.arange(1, len(mfa.eigenvalues_) + 1)
eigs = mfa.eigenvalues_
var_pct = mfa.percentage_of_variance_
cum_pct = mfa.cumulative_percentage_of_variance_

colors_bar = ["#1f4e78" if e >= 1.0 else "#9bb0c1" for e in eigs]
bars = ax1.bar(dims, eigs, color=colors_bar, edgecolor="#16324f", linewidth=0.7, alpha=0.9, label="Autovalor (Inercia)")
ax1.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="Criterio de Kaiser (λ = 1.0)")
for b, e, vp in zip(bars, eigs, var_pct):
    ax1.text(b.get_x() + b.get_width()/2, e + 0.04, f"{e:.2f}\n({vp:.1f}%)", ha="center", va="bottom", fontsize=7.5)

ax1.set_xlabel("Dimensión factorial")
ax1.set_ylabel("Autovalor (λ)", color="#1f4e78")
ax1.set_xticks(dims)
ax1.set_ylim(0, max(eigs) * 1.22)

ax2 = ax1.twinx()
ax2.plot(dims, cum_pct, color="#d35400", marker="o", linewidth=1.8, markersize=5, label="Varianza acumulada (%)")
ax2.set_ylabel("Varianza explicada acumulada (%)", color="#d35400")
ax2.set_ylim(0, 85)
ax2.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.9)
ax1.set_title("Gráfico de sedimentación (Scree Plot) del AFM — Selección de 3 dimensiones (32.74%)")
plt.tight_layout()
plt.savefig(fig_dir / "fig1_scree_plot.png")
plt.close()

# ---- FIGURE 2: Group Cloud ----
fig, ax = plt.subplots(figsize=(6.5, 5.2))
gc = mfa.group_coordinates_
grp_colors = {
    "G1_Basal": "#1b9e77",
    "G2_Esfuerzo_Num": "#d95f02",
    "G3_Clinica_Cat": "#7570b3",
    "G4_Prueba_Cat": "#e7298a",
    "G5_Diagnostico_Sup": "#555555",
}
for grp in gc.index:
    x, y = gc.loc[grp, 0], gc.loc[grp, 1]
    is_sup = (grp == "G5_Diagnostico_Sup")
    marker = "^" if is_sup else "o"
    ax.scatter(x, y, s=130, color=grp_colors[grp], marker=marker, edgecolor="black", linewidth=0.8, zorder=4)
    ax.annotate(
        group_labels_es[grp],
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold" if not is_sup else "normal",
        style="italic" if is_sup else "normal",
        color=grp_colors[grp],
    )
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlim(-0.05, 0.90)
ax.set_ylim(-0.05, 0.72)
ax.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax.set_title("Nube de Grupos (Representación de bloques activos y suplementario)")
ax.grid(True, linestyle="--", alpha=0.25)
plt.tight_layout()
plt.savefig(fig_dir / "fig2_nube_grupos.png")
plt.close()

# ---- FIGURE 3: Quantitative Variables Correlation Circle ----
fig, ax = plt.subplots(figsize=(6.0, 6.0))
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color="#7f8c8d", linewidth=1.0, linestyle="--")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

var_names_es = {
    "age": "Edad (age)",
    "trestbps": "Presión arterial (trestbps)",
    "chol": "Colesterol (chol)",
    "thalach": "FC máxima (thalach)",
    "oldpeak": "Depresión ST (oldpeak)",
}
for idx in col_corr.index:
    grp, v = idx
    x, y = col_corr.loc[idx, 0], col_corr.loc[idx, 1]
    c = grp_colors[grp]
    ax.arrow(0, 0, x, y, head_width=0.035, head_length=0.04, fc=c, ec=c, linewidth=1.8, length_includes_head=True, zorder=4)
    offset_x = 0.04 if x >= 0 else -0.04
    offset_y = 0.04 if y >= 0 else -0.06
    ha = "left" if x >= 0 else "right"
    ax.text(x + offset_x, y + offset_y, var_names_es.get(v, v), fontsize=8.5, fontweight="bold", color=c, ha=ha)

ax.set_xlim(-1.18, 1.18)
ax.set_ylim(-1.18, 1.18)
ax.set_aspect("equal")
ax.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax.set_title("Círculo de Correlaciones — Variables Cuantitativas (Dim 1 vs Dim 2)")
legend_handles = [
    mpatches.Patch(color=grp_colors["G1_Basal"], label="G1: Basal"),
    mpatches.Patch(color=grp_colors["G2_Esfuerzo_Num"], label="G2: Esfuerzo"),
]
ax.legend(handles=legend_handles, loc="lower left", frameon=True)
plt.tight_layout()
plt.savefig(fig_dir / "fig3_nube_variables_cuant.png")
plt.close()

# ---- FIGURE 4: Categorical Modalities Cloud (Dim 1 vs Dim 2) ----
fig, ax = plt.subplots(figsize=(8.6, 6.4))
cat_mask = [idx[0] in ["G3_Clinica_Cat", "G4_Prueba_Cat", "G5_Diagnostico_Sup"] for idx in col_coords.index]
cat_df = col_coords.loc[cat_mask].copy()
cat_cos2_2d = col_cos2.loc[cat_mask, [0, 1]].sum(axis=1)

custom_offsets = {
    "Femenino": (8, 0),
    "Masculino": (8, 0),
    "Vasos_0": (-42, 8),
    "Angina_atipica": (8, -12),
    "Vasos_2": (8, 8),
    "Vasos_3": (8, -10),
    "Exang_Si": (-12, 10),
    "Asintomatico": (8, -6),
    "Thal_Defecto_Rev": (-48, -14),
    "Enfermedad_Cardiaca": (-35, 12),
    "Sano": (8, 6),
    "Thal_Normal": (8, 5),
    "Exang_No": (8, -6),
}

for idx, row in cat_df.iterrows():
    grp, var_mod = idx
    x, y = row[0], row[1]
    c = grp_colors[grp]
    is_sup = (grp == "G5_Diagnostico_Sup")
    cos_val = cat_cos2_2d.loc[idx]
    ax.scatter(x, y, s=95 if is_sup else 55 + 140 * cos_val, color=c,
               marker="^" if is_sup else "o", edgecolor="black", linewidth=0.7, alpha=0.9, zorder=4)
    if cos_val >= 0.06 or is_sup:
        label = var_mod.split("__")[-1]
        dx, dy = custom_offsets.get(label, (6, 4))
        ax.annotate(
            label, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=8, fontweight="bold" if is_sup else "normal",
            color=c,
        )

ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlim(-0.76, 0.82)
ax.set_ylim(-0.68, 0.72)
ax.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax.set_title("Nube de Modalidades Categóricas y Variable Ilustrativa Diagnóstico (Dim 1 vs Dim 2)")
legend_handles = [
    mpatches.Patch(color=grp_colors["G3_Clinica_Cat"], label="G3: Clínica (Cual.)"),
    mpatches.Patch(color=grp_colors["G4_Prueba_Cat"], label="G4: Prueba/Angio (Cual.)"),
    mpatches.Patch(color=grp_colors["G5_Diagnostico_Sup"], label="G5: Diagnóstico (Suplementario)"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=True)
ax.grid(True, linestyle="--", alpha=0.2)
plt.tight_layout()
plt.savefig(fig_dir / "fig4_nube_modalidades.png")
plt.close()

# ---- FIGURE 5: Individuals Cloud & Partial Clouds ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4))

# Panel A: Individuals colored by supplementary diagnosis
diag_colors = {"Sano": "#27ae60", "Enfermedad_Cardiaca": "#c0392b"}
for label_d in ["Sano", "Enfermedad_Cardiaca"]:
    m = df["diagnostico"] == label_d
    pts = row_coords.loc[m, [0, 1]].to_numpy()
    ax1.scatter(pts[:, 0], pts[:, 1], s=26, alpha=0.65, color=diag_colors[label_d],
                label=f"{label_d.replace('_', ' ')} (n={m.sum()})", edgecolor="white", linewidth=0.3)
    # Centroid
    cx, cy = pts.mean(axis=0)
    ax1.scatter(cx, cy, s=160, color=diag_colors[label_d], marker="X", edgecolor="black", linewidth=1.0, zorder=5)

ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax1.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax1.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax1.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax1.set_title("A. Nube de Individuos (n=297) según Diagnóstico")
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.2)

# Panel B: Partial clouds for 3 representative individuals (max Dim1+, max Dim1-, max Dim2+)
idx_pos1 = row_coords[0].idxmax()
idx_neg1 = row_coords[0].idxmin()
idx_pos2 = row_coords[1].idxmax()
rep_ids = [idx_neg1, idx_pos1, idx_pos2]
rep_names = {
    idx_neg1: f"Pac. #{idx_neg1} (Sano extremo)",
    idx_pos1: f"Pac. #{idx_pos1} (Isquemia severa)",
    idx_pos2: f"Pac. #{idx_pos2} (Perfil Dim 2+)",
}
active_grps = ["G1_Basal", "G2_Esfuerzo_Num", "G3_Clinica_Cat", "G4_Prueba_Cat"]
for pid in rep_ids:
    gx, gy = row_coords.loc[pid, 0], row_coords.loc[pid, 1]
    ax2.scatter(gx, gy, s=110, color="black", marker="o", zorder=5)
    ax2.annotate(rep_names[pid], (gx, gy), xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold")
    for g in active_grps:
        px = partial_coords.loc[pid, (g, 0)]
        py = partial_coords.loc[pid, (g, 1)]
        ax2.plot([gx, px], [gy, py], color=grp_colors[g], linewidth=1.3, alpha=0.8)
        ax2.scatter(px, py, s=45, color=grp_colors[g], edgecolor="black", linewidth=0.5, zorder=4)

ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax2.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax2.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax2.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax2.set_title("B. Nube de Individuos Parciales (3 pacientes representativos)")
legend_p = [mpatches.Patch(color=grp_colors[g], label=group_labels_es[g]) for g in active_grps]
ax2.legend(handles=legend_p, loc="lower left", fontsize=7.5, frameon=True)
ax2.grid(True, linestyle="--", alpha=0.2)

plt.tight_layout()
plt.savefig(fig_dir / "fig5_nube_individuos_parciales.png")
plt.close()

# ---- FIGURE 6: Top Contributions & Cos2 for Dim 1 & Dim 2 ----
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
for d, ax in enumerate(axes):
    top = (col_contrib[d] * 100).sort_values(ascending=False).head(10).iloc[::-1]
    cos_vals = col_cos2.loc[top.index, d]
    labels = [f"{idx[1].split('__')[-1]}" for idx in top.index]
    bars = ax.barh(labels, top.values, color="#2c5f8a", alpha=0.88, edgecolor="#16324f", linewidth=0.6)
    avg_contrib = 100.0 / len(col_contrib)
    ax.axvline(avg_contrib, color="#c0392b", linestyle="--", linewidth=1.1, label=f"Contribución media ({avg_contrib:.2f}%)")
    for b, c2 in zip(bars, cos_vals):
        ax.text(b.get_width() + 0.25, b.get_y() + b.get_height()/2, f"cos²={c2:.2f}", va="center", fontsize=7.5)
    ax.set_xlabel("Contribución (%)")
    ax.set_xlim(0, top.max() * 1.25)
    ax.set_title(f"Top 10 Contribuciones y cos² — Dimensión {d+1} ({var_pct[d]:.2f}%)")
    ax.legend(loc="lower right", fontsize=7.5)
plt.tight_layout()
plt.savefig(fig_dir / "fig6_contrib_cos2.png")
plt.close()

# ---- CLUSTERING (HCPC on 3D MFA coords) ----
coords_3d = row_coords[[0, 1, 2]].to_numpy()
Z_link = linkage(coords_3d, method="ward")

cluster_metrics = []
for k in range(2, 7):
    w_lab = fcluster(Z_link, t=k, criterion="maxclust") - 1
    cents = np.array([coords_3d[w_lab == c].mean(axis=0) for c in range(k)])
    hc = KMeans(n_clusters=k, init=cents, n_init=1, random_state=42).fit(coords_3d)
    cluster_metrics.append({
        "k": k,
        "silhouette": silhouette_score(coords_3d, hc.labels_),
        "calinski_harabasz": calinski_harabasz_score(coords_3d, hc.labels_),
        "inertia": hc.inertia_,
    })
pd.DataFrame(cluster_metrics).to_csv(tab_dir / "clustering_metrics.csv", index=False)

k_sel = 4
ward_labels = fcluster(Z_link, t=k_sel, criterion="maxclust") - 1
centroids = np.array([coords_3d[ward_labels == c].mean(axis=0) for c in range(k_sel)])
hcpc = KMeans(n_clusters=k_sel, init=centroids, n_init=1, random_state=42).fit(coords_3d)
df["cluster"] = hcpc.labels_ + 1  # 1..4
df.to_csv("heart_cleveland_clustered.csv", index=False)

# ---- FIGURE 7: Dendrogram + HCPC Clusters ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
dendrogram(Z_link, truncate_mode="lastp", p=24, leaf_rotation=90, leaf_font_size=7,
           color_threshold=Z_link[-(k_sel-1), 2] - 0.05, ax=ax1)
cut_height = (Z_link[-(k_sel-1), 2] + Z_link[-k_sel, 2]) / 2
ax1.axhline(cut_height, color="#c0392b", linestyle="--", linewidth=1.3, label=f"Corte HCPC (k={k_sel})")
ax1.set_title("A. Dendrograma de Ward (truncado a 24 ramas)")
ax1.set_xlabel("Nodos / Individuos agrupados")
ax1.set_ylabel("Distancia de Ward")
ax1.legend(loc="upper right")

cl_colors = {1: "#2980b9", 2: "#27ae60", 3: "#e67e22", 4: "#c0392b"}
cl_names = {
    1: "C1: Femenino / Bajo riesgo",
    2: "C2: Joven / Alta capacidad",
    3: "C3: Metabólico / Angina típica",
    4: "C4: Alto riesgo isquémico",
}
for c in range(1, k_sel + 1):
    m = df["cluster"] == c
    pts = row_coords.loc[m, [0, 1]].to_numpy()
    ax2.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.7, color=cl_colors[c],
                label=f"{cl_names[c]} (n={m.sum()})", edgecolor="white", linewidth=0.3)
    cx, cy = pts.mean(axis=0)
    ax2.scatter(cx, cy, s=170, color=cl_colors[c], marker="D", edgecolor="black", linewidth=1.0, zorder=5)

ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax2.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax2.set_xlabel(f"Dimensión 1 ({var_pct[0]:.2f}%)")
ax2.set_ylabel(f"Dimensión 2 ({var_pct[1]:.2f}%)")
ax2.set_title("B. Clústeres HCPC (k=4) en el Plano Factorial (Dim 1 vs Dim 2)")
ax2.legend(loc="upper left", fontsize=7.5, frameon=True)
ax2.grid(True, linestyle="--", alpha=0.2)
plt.tight_layout()
plt.savefig(fig_dir / "fig7_clustering_hcpc.png")
plt.close()

# ---- COMPUTE & SAVE ALL V-TESTS ----
N = len(df)
cat_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "diagnostico"]
vtest_quant_rows = []
vtest_cat_rows = []

for c in range(1, k_sel + 1):
    mask = (df["cluster"] == c)
    n_k = int(mask.sum())
    for v in num_vars:
        mk = df.loc[mask, v].mean()
        mg = df[v].mean()
        sdg = df[v].std(ddof=0)
        denom = sdg * np.sqrt((N - n_k) / (n_k * (N - 1)))
        vt = (mk - mg) / denom
        pv = 2 * (1 - stats.norm.cdf(abs(vt)))
        vtest_quant_rows.append({
            "cluster": c, "variable": v, "mean_cluster": mk, "mean_global": mg,
            "v_test": vt, "p_value": pv
        })
    for v in cat_vars:
        for mod in df[v].cat.categories:
            N_c = int((df[v] == mod).sum())
            n_kc = int(((df[v] == mod) & mask).sum())
            exp_kc = n_k * N_c / N
            if n_kc >= exp_kc:
                pv = stats.hypergeom.sf(n_kc - 1, N, N_c, n_k)
                sign = 1.0
            else:
                pv = stats.hypergeom.cdf(n_kc, N, N_c, n_k)
                sign = -1.0
            pv2 = min(1.0, 2.0 * pv)
            vt = sign * stats.norm.isf(max(pv2, 1e-300) / 2.0)
            vtest_cat_rows.append({
                "cluster": c, "variable": v, "modalidad": mod,
                "pct_cluster": n_kc / n_k * 100, "pct_global": N_c / N * 100,
                "v_test": vt, "p_value": pv2
            })

vt_q_df = pd.DataFrame(vtest_quant_rows)
vt_c_df = pd.DataFrame(vtest_cat_rows)
vt_q_df.to_csv(tab_dir / "vtest_quant.csv", index=False)
vt_c_df.to_csv(tab_dir / "vtest_cat.csv", index=False)

# ---- FIGURE 8: v-test Heatmap for Quantitative Variables + Top Qualitative ----
pivot_q = vt_q_df.pivot(index="variable", columns="cluster", values="v_test")
pivot_q.columns = [f"Clúster {c}" for c in pivot_q.columns]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
sns.heatmap(pivot_q, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, vmin=-10, vmax=10,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Valor test (v-test)"})
ax.set_title("Valores test (v-test) de variables cuantitativas por Clúster HCPC")
ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(fig_dir / "fig8_vtest_clusters.png")
plt.close()

print("All tables and figures generated successfully!")
