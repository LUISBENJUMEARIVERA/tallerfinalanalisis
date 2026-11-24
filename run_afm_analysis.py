import json
import numpy as np
import pandas as pd
import prince
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 1. Load & clean dataset
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

data_dict = {}
for grp, var_list in groups.items():
    for v in var_list:
        data_dict[(grp, v)] = df[v]
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

row_coords = mfa.row_coordinates(X_mfa)
col_coords = mfa.column_coordinates_
col_contrib = mfa.column_contributions_
col_cos2 = mfa.column_cosine_similarities_
row_contrib = mfa.row_contributions_
row_cos2 = mfa.row_cosine_similarities(X_mfa)

print("=== GROUP COORDINATES (Dim 1-3) ===")
print(mfa.group_coordinates_[[0, 1, 2]].round(4))

print("\n=== GROUP CONTRIBUTIONS % (Dim 1-3) ===")
print((mfa.group_contributions_[[0, 1, 2]] * 100).round(2))

print("\n=== GROUP COS2 (Dim 1-3) ===")
print(mfa.group_cosine_similarities_[[0, 1, 2]].round(4))

print("\n=== COL CONTRIB SUM CHECK ===")
print(col_contrib[[0, 1, 2]].sum().round(4))

print("\n=== TOP 10 COLUMN CONTRIBUTIONS % (Dim 1, 2, 3) ===")
for d in [0, 1, 2]:
    print(f"\n-- Dim {d+1} --")
    s = (col_contrib[d] * 100).sort_values(ascending=False).head(10)
    for idx, val in s.items():
        coord = col_coords.loc[idx, d]
        c2 = col_cos2.loc[idx, d]
        print(f"  {idx[0]:16s} | {idx[1]:25s} | coord={coord:+.3f} | contrib={val:5.2f}% | cos2={c2:.3f}")

print("\n=== SUPPLEMENTARY GROUP (G5_Diagnostico_Sup) COORDS & COS2 ===")
sup_idx = [idx for idx in col_coords.index if idx[0] == "G5_Diagnostico_Sup"]
for idx in sup_idx:
    print(f"  {idx[1]:30s} | Dim1={col_coords.loc[idx,0]:+.3f} (cos2={col_cos2.loc[idx,0]:.3f}) | Dim2={col_coords.loc[idx,1]:+.3f} (cos2={col_cos2.loc[idx,1]:.3f}) | Dim3={col_coords.loc[idx,2]:+.3f} (cos2={col_cos2.loc[idx,2]:.3f})")

print("\n=== CLUSTERING METRICS (on 3D MFA coords) ===")
coords_3d = row_coords[[0, 1, 2]].to_numpy()
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(coords_3d)
    sil = silhouette_score(coords_3d, km.labels_)
    ch = calinski_harabasz_score(coords_3d, km.labels_)
    Z_link = linkage(coords_3d, method="ward")
    ward_labels = fcluster(Z_link, t=k, criterion="maxclust") - 1
    sil_w = silhouette_score(coords_3d, ward_labels)
    ch_w = calinski_harabasz_score(coords_3d, ward_labels)
    # HCPC: Ward centroids -> KMeans consolidation
    centroids = np.array([coords_3d[ward_labels == c].mean(axis=0) for c in range(k)])
    hcpc = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=42).fit(coords_3d)
    sil_h = silhouette_score(coords_3d, hcpc.labels_)
    ch_h = calinski_harabasz_score(coords_3d, hcpc.labels_)
    print(f"k={k}: KMeans(sil={sil:.4f}, CH={ch:.1f}) | Ward(sil={sil_w:.4f}, CH={ch_w:.1f}) | HCPC(sil={sil_h:.4f}, CH={ch_h:.1f})")

# Use HCPC with k=3 (let's also inspect k=3 vs k=4)
for k_sel in [3, 4]:
    print(f"\n================ HCPC k={k_sel} ================")
    Z_link = linkage(coords_3d, method="ward")
    ward_labels = fcluster(Z_link, t=k_sel, criterion="maxclust") - 1
    centroids = np.array([coords_3d[ward_labels == c].mean(axis=0) for c in range(k_sel)])
    hcpc = KMeans(n_clusters=k_sel, init=centroids, n_init=1, random_state=42).fit(coords_3d)
    labels = hcpc.labels_
    print("Cluster sizes:", pd.Series(labels).value_counts().sort_index().to_dict())
    print("Cross-tab with diagnostico:")
    print(pd.crosstab(pd.Series(labels, name="Cluster"), df["diagnostico"], normalize="index").round(3))

    # Compute v-tests
    N = len(df)
    num_vars = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_vars = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "diagnostico"]

    for c in range(k_sel):
        mask = (labels == c)
        n_k = mask.sum()
        print(f"\n--- Cluster {c+1} (n={n_k}, {n_k/N*100:.1f}%) ---")
        # Quantitative v-tests
        q_rows = []
        for v in num_vars:
            mean_k = df.loc[mask, v].mean()
            mean_g = df[v].mean()
            sd_g = df[v].std(ddof=0)
            denom = sd_g * np.sqrt((N - n_k) / (n_k * (N - 1)))
            vtest = (mean_k - mean_g) / denom
            pval = 2 * (1 - stats.norm.cdf(abs(vtest)))
            q_rows.append((v, mean_k, mean_g, vtest, pval))
        q_rows.sort(key=lambda x: abs(x[3]), reverse=True)
        print("  [Cuantitativas]")
        for v, mk, mg, vt, pv in q_rows:
            if abs(vt) >= 1.96:
                print(f"    {v:10s}: media_cluster={mk:6.2f} vs global={mg:6.2f} | v-test={vt:+6.2f} (p={pv:.2e})")

        # Qualitative v-tests (hypergeometric -> normal quantile)
        c_rows = []
        for v in cat_vars:
            for mod in df[v].cat.categories:
                N_c = (df[v] == mod).sum()
                n_kc = ((df[v] == mod) & mask).sum()
                exp_kc = n_k * N_c / N
                if n_kc >= exp_kc:
                    pval = stats.hypergeom.sf(n_kc - 1, N, N_c, n_k)
                    sign = 1.0
                else:
                    pval = stats.hypergeom.cdf(n_kc, N, N_c, n_k)
                    sign = -1.0
                pval_two = min(1.0, 2.0 * pval)
                # Convert to v-test
                pval_clip = max(pval_two, 1e-300)
                vt = sign * stats.norm.isf(pval_clip / 2.0)
                c_rows.append((f"{v}={mod}", n_kc / n_k * 100, N_c / N * 100, vt, pval_two))
        c_rows.sort(key=lambda x: abs(x[3]), reverse=True)
        print("  [Cualitativas |v-test| >= 2.0, top 8]")
        shown = 0
        for name, pct_k, pct_g, vt, pv in c_rows:
            if abs(vt) >= 1.96 and shown < 8:
                print(f"    {name:30s}: {pct_k:5.1f}% en cluster vs {pct_g:5.1f}% global | v-test={vt:+6.2f} (p={pv:.2e})")
                shown += 1
