import pandas as pd
import numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

tab_dir = Path("resultados")
fig_dir = Path("figuras")

eig_df = pd.read_csv(tab_dir / "eigenvalues.csv")
grp_df = pd.read_csv(tab_dir / "group_summary.csv")
col_df = pd.read_csv(tab_dir / "column_summary.csv")
corr_df = pd.read_csv(tab_dir / "quant_correlations.csv")
desc_df = pd.read_csv(tab_dir / "numeric_descriptives.csv")
cl_metrics = pd.read_csv(tab_dir / "clustering_metrics.csv")
vt_q = pd.read_csv(tab_dir / "vtest_quant.csv")
vt_c = pd.read_csv(tab_dir / "vtest_cat.csv")
df_cl = pd.read_csv("heart_cleveland_clustered.csv")

doc = Document()

# Page margins (compact academic layout so report stays ~11-13 pages <= 15)
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

# Base Normal style
style_normal = doc.styles["Normal"]
style_normal.font.name = "Calibri"
style_normal.font.size = Pt(10.5)
style_normal.paragraph_format.space_after = Pt(5)
style_normal.paragraph_format.line_spacing = 1.12

NAVY = RGBColor(0x1F, 0x4E, 0x78)
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x55, 0x55, 0x55)

def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = NAVY
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x2C, 0x5F, 0x8A)
    return p

def add_styled_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(str(h))
        r.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(hdr_cells[i], "1F4E78")
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        bg = "F4F7FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cells[c_idx].text = ""
            p = cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(1.5)
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            r.font.color.rgb = DARK
            set_cell_bg(cells[c_idx], bg)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(3)
    return table

def add_figure(img_path, caption, width_in=6.2):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(6)
    rc = p_cap.add_run(caption)
    rc.italic = True
    rc.font.size = Pt(8.5)
    rc.font.color.rgb = MUTED

# ================= HEADER / TITLE BLOCK =================
p_uni = doc.add_paragraph()
p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_uni.paragraph_format.space_after = Pt(2)
r = p_uni.add_run("UNIVERSIDAD AUTÓNOMA DE OCCIDENTE\n")
r.bold = True; r.font.size = Pt(13); r.font.color.rgb = NAVY
r2 = p_uni.add_run("MAESTRÍA EN INTELIGENCIA ARTIFICIAL Y CIENCIA DE DATOS\nANÁLISIS MULTIVARIADO — Profesora: Claudia Lorena Montes\n")
r2.bold = True; r2.font.size = Pt(10); r2.font.color.rgb = DARK
r3 = p_uni.add_run("Fecha de entrega: 28 de noviembre de 2026")
r3.font.size = Pt(9.5); r3.font.color.rgb = MUTED

p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(8)
p_title.paragraph_format.space_after = Pt(8)
rt = p_title.add_run("Tarea 4: Aplicación de la Técnica Multivariada Análisis Factorial Múltiple (AFM) y Clustering HCPC sobre Datos Clínicos Cardiovasculares")
rt.bold = True; rt.font.size = Pt(14); rt.font.color.rgb = NAVY

# ================= SECTION 1 =================
add_heading_1("1. Conjunto de datos, fuente, tamaño de muestra y número de variables")
doc.add_paragraph(
    "Se seleccionó el conjunto de datos reales Heart Disease — Cohorte Cleveland (processed.cleveland.data), "
    "proveniente del UCI Machine Learning Repository (fuente original: V.A. Medical Center, Long Beach y Cleveland Clinic Foundation; "
    "investigador principal Robert Detrano, M.D., Ph.D.; URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/). "
    "El archivo original contiene 303 registros de pacientes evaluados por sospecha de enfermedad arterial coronaria y 14 variables clínicas. "
    "Tras depurar 6 registros con valores faltantes (4 en la variable ca y 2 en thal), se trabajó con una muestra completa de n = 297 individuos "
    "(160 pacientes sanos [53.9%] y 137 con enfermedad cardíaca [46.1%]) y p = 14 variables."
)
doc.add_paragraph(
    "Para aplicar el Análisis Factorial Múltiple (AFM) bajo el enfoque de Escofier y Pagès (implementación prince 0.23.1, equivalente a FactoMineR), "
    "las 13 variables explicativas se organizaron en 4 grupos temáticos activos (2 cuantitativos estandarizados y 2 cualitativos vía ACM) "
    "y 1 grupo suplementario (ilustrativo) para proyectar el diagnóstico clínico sin influir en la construcción de los ejes factoriales:"
)
bullets_g = [
    "G1_Basal (Cuantitativo, 3 variables): age, trestbps, chol — perfil demográfico y hemodinámico en reposo.",
    "G2_Esfuerzo_Num (Cuantitativo, 2 variables): thalach, oldpeak — respuesta cuantitativa a la prueba de esfuerzo.",
    "G3_Clinica_Cat (Cualitativo, 4 variables): sex, cp, fbs, restecg — sintomatología y hallazgos clínicos en reposo.",
    "G4_Prueba_Cat (Cualitativo, 4 variables): exang, slope, ca, thal — hallazgos categóricos de esfuerzo y angiografía.",
    "G5_Diagnostico_Sup (Cualitativo Suplementario, 1 variable): diagnostico (derivada de num: Sano [num=0] vs. Enfermedad_Cardiaca [num>0]).",
]
for b in bullets_g:
    p = doc.add_paragraph(b, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)

# ================= SECTION 2 =================
add_heading_1("2. Descripción de variables, tipo, escala y unidades de medición")
doc.add_paragraph(
    "La Tabla 1 detalla las 14 variables del conjunto de datos, su clasificación estadística, escala de medición, unidades/categorías "
    "y el grupo asignado dentro de la estructura del AFM."
)
var_rows = [
    ["age", "Cuantitativa discreta", "Razón", "Años (29 – 77; media 54.5 ± 9.0)", "G1_Basal (Activo)"],
    ["trestbps", "Cuantitativa continua", "Razón", "mm Hg (94 – 200; media 131.7 ± 17.8)", "G1_Basal (Activo)"],
    ["chol", "Cuantitativa continua", "Razón", "mg/dl (126 – 564; media 247.4 ± 52.0)", "G1_Basal (Activo)"],
    ["thalach", "Cuantitativa discreta", "Razón", "lpm / bpm (71 – 202; media 149.6 ± 22.9)", "G2_Esfuerzo_Num (Activo)"],
    ["oldpeak", "Cuantitativa continua", "Razón", "mm depresión ST (0.0 – 6.2; media 1.06 ± 1.17)", "G2_Esfuerzo_Num (Activo)"],
    ["sex", "Cualitativa dicotómica", "Nominal", "Femenino (32.3%), Masculino (67.7%)", "G3_Clinica_Cat (Activo)"],
    ["cp", "Cualitativa politómica", "Nominal", "Angina típica (7.7%), atípica (16.5%), no anginoso (27.9%), asintomático (47.8%)", "G3_Clinica_Cat (Activo)"],
    ["fbs", "Cualitativa dicotómica", "Nominal", "Glucemia ayunas <=120 mg/dl (85.5%), >120 mg/dl (14.5%)", "G3_Clinica_Cat (Activo)"],
    ["restecg", "Cualitativa politómica", "Nominal", "Normal (49.5%), Anomalía ST-T (1.3%), Hipertrofia VI (49.2%)", "G3_Clinica_Cat (Activo)"],
    ["exang", "Cualitativa dicotómica", "Nominal", "Angina inducida por ejercicio: No (67.3%), Sí (32.7%)", "G4_Prueba_Cat (Activo)"],
    ["slope", "Cualitativa politómica", "Ordinal", "Pendiente ST: Ascendente (46.8%), Plana (46.1%), Descendente (7.1%)", "G4_Prueba_Cat (Activo)"],
    ["ca", "Cualitativa politómica", "Ordinal", "Vasos coloreados por fluoroscopia: 0 (58.6%), 1 (21.9%), 2 (12.8%), 3 (6.7%)", "G4_Prueba_Cat (Activo)"],
    ["thal", "Cualitativa politómica", "Nominal", "Normal (55.2%), Defecto fijo (6.1%), Defecto reversible (38.7%)", "G4_Prueba_Cat (Activo)"],
    ["diagnostico (num)", "Cualitativa dicotómica", "Nominal", "Sano [num=0] (53.9%), Enfermedad Cardíaca [num>0] (46.1%)", "G5_Diagnostico_Sup (Ilust.)"],
]
add_styled_table(["Variable", "Tipo", "Escala", "Unidades / Categorías (n=297)", "Grupo AFM"], var_rows, [0.95, 1.15, 0.7, 2.55, 1.35])

# ================= SECTION 3 =================
add_heading_1("3. Número de dimensiones a trabajar y porcentaje de varianza explicado")
doc.add_paragraph(
    "La Tabla 2 presenta los autovalores (inercias), el porcentaje de varianza explicada y el porcentaje acumulado para las primeras 6 dimensiones del AFM. "
    "Aplicando el criterio de Kaiser (retener dimensiones con autovalor λ ≥ 1.0, dado que el AFM pondera cada grupo para que su primer autovalor sea igual a 1) "
    "en conjunto con el gráfico de sedimentación (Figura 1), se seleccionan exactamente 3 dimensiones factoriales:"
)
eig_rows = []
for i in range(6):
    r = eig_df.iloc[i]
    eig_rows.append([f"Dimensión {i+1}", f"{r['eigenvalue']:.3f}", r["% of variance"], r["% of variance (cumulative)"]])
add_styled_table(["Componente", "Autovalor (λ)", "Varianza explicada (%)", "Varianza acumulada (%)"], eig_rows, [1.5, 1.4, 1.8, 1.9])

doc.add_paragraph(
    "Las 3 dimensiones retenidas (λ₁ = 2.221, λ₂ = 1.188, λ₃ = 1.076) explican en conjunto el 32.74% de la varianza total "
    "(16.21% en Dim 1, 8.67% en Dim 2 y 7.85% en Dim 3). A partir de la Dimensión 4, el autovalor cae por debajo de la unidad (λ₄ = 0.921). "
    "En un AFM mixto con bloques categóricos codificados mediante Análisis de Correspondencias Múltiples (ACM), la inercia total se distribuye entre múltiples modalidades, "
    "por lo que un 32.74% en 3 ejes representa una estructura factorial sólida y claramente interpretable."
)
add_figure(fig_dir / "fig1_scree_plot.png", "Figura 1. Gráfico de sedimentación del AFM con umbral de Kaiser (λ = 1.0) y varianza acumulada.", 5.8)

# ================= SECTION 4 =================
add_heading_1("4. Análisis de las nubes del AFM: grupos, variables, modalidades e individuos")

add_heading_2("4.1 Nube de Grupos")
doc.add_paragraph(
    "La Figura 2 muestra la representación de los 4 grupos activos y del grupo suplementario G5_Diagnostico_Sup en el primer plano factorial. "
    "La Dimensión 1 está fuertemente estructurada por G4_Prueba_Cat (coord. 0.742, contribución 33.43%) y G2_Esfuerzo_Num (coord. 0.721, contribución 32.49%), "
    "seguidos por G3_Clinica_Cat (coord. 0.494, 22.24%); es decir, el primer eje sintetiza de forma consensual la respuesta fisiológica y clínica a la isquemia por esfuerzo. "
    "La Dimensión 2 está dominada por G3_Clinica_Cat (coord. 0.575, contribución 48.36%) y G1_Basal (coord. 0.479, contribución 40.31%), capturando el perfil basal demográfico-lipídico. "
    "La Dimensión 3 responde principalmente a G3_Clinica_Cat (coord. 0.784, contribución 72.88%)."
)
add_figure(fig_dir / "fig2_nube_grupos.png", "Figura 2. Nube de grupos activos y grupo ilustrativo Diagnóstico en el plano Dim 1 vs Dim 2.", 5.2)

add_heading_2("4.2 Nube de Variables Cuantitativas y Nube de Modalidades Categóricas")
doc.add_paragraph(
    "En el círculo de correlaciones (Figura 3), la Dimensión 1 opone de manera nítida la frecuencia cardíaca máxima alcanzada thalach (r = -0.702) "
    "contra la depresión del segmento ST oldpeak (r = +0.693) y la edad age (r = +0.512). La Dimensión 2 agrupa positivamente las variables basales: "
    "presión arterial en reposo trestbps (r = +0.497), colesterol sérico chol (r = +0.478) y edad (r = +0.454), las cuales son ortogonales a las variables de esfuerzo."
)
add_figure(fig_dir / "fig3_nube_variables_cuant.png", "Figura 3. Círculo de correlaciones de las variables cuantitativas (G1_Basal y G2_Esfuerzo_Num).", 4.8)

doc.add_paragraph(
    "La nube de modalidades categóricas (Figura 4) confirma una clara polarización clínica sobre la Dimensión 1: en el semieje negativo (-0.65 a -0.45) "
    "se concentran los marcadores de normalidad funcional (Pend_Ascendente, Thal_Normal, Exang_No, Vasos_0, Angina_atipica) junto con la modalidad ilustrativa Sano (-0.670). "
    "En el semieje positivo (+0.48 a +0.67) se agrupan los marcadores de enfermedad coronaria (Asintomatico, Exang_Si, Pend_Plana, Thal_Defecto_Rev) junto con Enfermedad_Cardiaca (+0.670). "
    "Sobre la Dimensión 2 se observa la separación entre sexo Femenino (+0.598) y Masculino (-0.598), asociándose el polo positivo con glucemia >120 mg/dl (+0.357) e hipertrofia VI (+0.285)."
)
add_figure(fig_dir / "fig4_nube_modalidades.png", "Figura 4. Nube de modalidades categóricas y variable ilustrativa Diagnóstico (Dim 1 vs Dim 2).", 5.8)

add_heading_2("4.3 Nube de Individuos e Individuos Parciales")
doc.add_paragraph(
    "La Figura 5A proyecta los n = 297 pacientes coloreados según su diagnóstico clínico real (variable suplementaria). Se evidencia una separación marcada a lo largo de la Dimensión 1 "
    "entre los centroides de pacientes Sanos y con Enfermedad Cardíaca. La Figura 5B ilustra las nubes de individuos parciales para tres pacientes extremos: "
    "el Paciente #222 (sano extremo, Dim 1 < -3) muestra alta coherencia entre sus 4 grupos (todos en el semieje izquierdo), mientras que el Paciente #125 (isquemia severa, Dim 1 > +3) "
    "presenta sus puntos parciales desplazados hacia el semieje derecho, aunque con mayor dispersión de G1_Basal y G3_Clinica_Cat hacia Dim 2 positivo."
)
add_figure(fig_dir / "fig5_nube_individuos_parciales.png", "Figura 5. A) Nube de individuos según diagnóstico suplementario; B) Nube de individuos parciales.", 6.4)

# ================= SECTION 5 =================
add_heading_1("5. Contribuciones y cosenos cuadrados (cos²)")
doc.add_paragraph(
    "La Tabla 3 resume las contribuciones (%) y cosenos cuadrados (cos²) de los grupos y de las variables/modalidades más influyentes en las 3 dimensiones retenidas "
    "(la contribución promedio esperada por columna activa es 100% / 29 = 3.45%). La Figura 6 visualiza el top 10 de Dim 1 y Dim 2:"
)
top_rows = []
for d_idx, d_name in [(0, "Dim 1"), (1, "Dim 2"), (2, "Dim 3")]:
    sub = col_df[col_df["group"] != "G5_Diagnostico_Sup"].sort_values(f"Dim{d_idx+1}_contrib_pct", ascending=False).head(5)
    for _, r in sub.iterrows():
        v_clean = r["variable"].split("__")[-1]
        top_rows.append([
            d_name, r["group"], v_clean,
            f"{r[f'Dim{d_idx+1}_coord']:+.3f}",
            f"{r[f'Dim{d_idx+1}_contrib_pct']:.2f}%",
            f"{r[f'Dim{d_idx+1}_cos2']:.3f}"
        ])
add_styled_table(["Eje", "Grupo", "Variable / Modalidad", "Coord.", "Contrib. (%)", "cos²"], top_rows, [0.65, 1.25, 1.85, 0.85, 1.0, 0.8])

doc.add_paragraph(
    "Interpretación conjunta: En Dim 1, thalach (contrib. 16.45%, cos² = 0.492) y oldpeak (16.04%, cos² = 0.480) explican casi un tercio del eje, "
    "respaldados por age (8.28%), cp=Asintomatico (6.87%, cos² = 0.366) y slope=Pend_Ascendente (5.20%, cos² = 0.385). "
    "La variable suplementaria diagnostico alcanza un cos² = 0.449 en Dim 1, confirmando que el primer eje factorial es un indicador directo de severidad isquémica. "
    "En Dim 2, las mayores contribuciones corresponden a sex=Femenino (16.32%, cos² = 0.358), trestbps (14.62%, cos² = 0.247), chol (13.52%, cos² = 0.229) y age (12.17%, cos² = 0.206). "
    "En Dim 3, destacan cp=Angina_tipica (24.26%, cos² = 0.354), fbs=Gluc_>120 (15.51%, cos² = 0.244) y sex=Femenino (12.62%, cos² = 0.251)."
)
add_figure(fig_dir / "fig6_contrib_cos2.png", "Figura 6. Top 10 contribuciones (%) y cosenos cuadrados (cos²) en las Dimensiones 1 y 2.", 6.4)

# ================= SECTION 6 =================
add_heading_1("6. Técnica de clustering, selección de k e interpretación")
doc.add_paragraph(
    "Se aplicó la metodología HCPC (Hierarchical Clustering on Principal Components): agrupamiento jerárquico aglomerativo mediante el método de Ward "
    "(distancia euclidiana sobre las coordenadas de las 3 dimensiones retenidas del AFM), seguido de una consolidación mediante K-Means usando los centroides de Ward como semillas. "
    "Esta técnica es el estándar natural tras un AFM porque elimina el ruido de las dimensiones residuales y combina la estabilidad estructural de Ward con la optimalidad local de K-Means."
)
cm_rows = []
for _, r in cl_metrics.iterrows():
    cm_rows.append([f"k = {int(r['k'])}", f"{r['silhouette']:.4f}", f"{r['calinski_harabasz']:.1f}", f"{r['inertia']:.2f}"])
add_styled_table(["Número de clústeres", "Índice Silhouette", "Calinski-Harabasz", "Inercia intra-clúster"], cm_rows, [1.6, 1.6, 1.7, 1.7])

doc.add_paragraph(
    "Argumentación de k = 4: Aunque k = 2 obtiene el mayor Silhouette (0.3279), solo reproduce la dicotomía trivial sano/enfermo. "
    "Entre las particiones más granulares, k = 4 alcanza un máximo local en el índice de Calinski-Harabasz (145.4 frente a 140.1 en k = 3 y 135.8 en k = 5), "
    "mantiene un Silhouette competitivo (0.3183) y coincide con el salto de altura en el dendrograma de Ward (Figura 7A), revelando cuatro perfiles clínicos claramente diferenciados (Figura 7B):"
)
cl_interp = [
    "Clúster 1 (n = 62, 20.9%) — Perfil Femenino Mayor con Colesterol Elevado y Bajo Riesgo: 95.2% mujeres, edad media 57.6 años, colesterol elevado (272.7 mg/dl), pero sin isquemia de esfuerzo (oldpeak 0.43 mm, 96.8% talasemia normal) y 87.1% pacientes sanos.",
    "Clúster 2 (n = 88, 29.6%) — Perfil Joven con Alta Capacidad Funcional y Bajo Riesgo: pacientes más jóvenes (47.0 años), mayor frecuencia cardíaca máxima (165.5 lpm), presión arterial baja (123.6 mm Hg), pendiente ST ascendente (79.5%), sin angina de esfuerzo (90.9%) y 76.1% sanos.",
    "Clúster 3 (n = 55, 18.5%) — Perfil Metabólico-Hipertensivo con Angina Típica (Riesgo Intermedio): mayoría masculina (85.5%), presión arterial elevada (143.8 mm Hg), glucemia >120 mg/dl (52.7%) y alta prevalencia de angina típica (40.0%), con 47.3% de enfermedad cardíaca.",
    "Clúster 4 (n = 92, 31.0%) — Perfil de Alto Riesgo Isquémico Severo: pacientes mayores (58.0 años), menor FC máxima (129.6 lpm), mayor depresión ST (oldpeak 1.88 mm), dolor asintomático (88.0%), angina inducida por ejercicio (70.7%), pendiente ST plana (79.3%) y 89.1% de enfermedad cardíaca.",
]
for item in cl_interp:
    p = doc.add_paragraph(item, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)

add_figure(fig_dir / "fig7_clustering_hcpc.png", "Figura 7. A) Dendrograma de Ward con corte en k=4; B) Clústeres HCPC proyectados en el plano factorial Dim 1 vs Dim 2.", 6.4)

# ================= SECTION 7 =================
add_heading_1("7. Variables de mayor relevancia en cada clúster mediante valores test (v-test)")
doc.add_paragraph(
    "Para identificar formalmente las variables que caracterizan cada clúster, se calcularon los valores test (v-test) bajo la formulación de FactoMineR: "
    "comparación de medias (con corrección por población finita) para variables cuantitativas y prueba hipergeométrica transformada a cuantil normal estándar para modalidades categóricas. "
    "Un valor |v-test| ≥ 1.96 indica significancia estadística al 5% (p < 0.05). La Figura 8 muestra el mapa de calor cuantitativo y la Tabla 4 resume las variables más relevantes por clúster."
)
vt_rows = []
for c in [1, 2, 3, 4]:
    q_sub = vt_q[(vt_q["cluster"] == c) & (vt_q["v_test"].abs() >= 1.96)].sort_values("v_test", key=abs, ascending=False).head(3)
    c_sub = vt_c[(vt_c["cluster"] == c) & (vt_c["v_test"].abs() >= 1.96)].sort_values("v_test", key=abs, ascending=False).head(4)
    q_str = "; ".join([f"{r['variable']} ({r['v_test']:+.2f})" for _, r in q_sub.iterrows()])
    c_str = "; ".join([f"{r['variable']}={r['modalidad']} ({r['v_test']:+.2f})" for _, r in c_sub.iterrows()])
    n_k = int((df_cl["cluster"] == c).sum())
    vt_rows.append([f"Clúster {c} (n={n_k})", q_str, c_str])
add_styled_table(["Clúster", "Top Cuantitativas (v-test)", "Top Modalidades Categóricas (v-test)"], vt_rows, [1.2, 2.3, 3.1])

add_figure(fig_dir / "fig8_vtest_clusters.png", "Figura 8. Mapa de calor de valores test (v-test) para las variables cuantitativas en los 4 clústeres HCPC.", 5.6)

# ================= SECTION 8 =================
add_heading_1("8. Conclusiones y hallazgos más importantes")
conclusions = [
    "El AFM permitió integrar de forma balanceada 13 variables clínicas mixtas (2 bloques cuantitativos y 2 cualitativos) sobre 297 pacientes, capturando el 32.74% de la inercia total en 3 dimensiones significativas (λ > 1).",
    "La Dimensión 1 (16.21%) constituye un eje clínico robusto de severidad isquémica (thalach r = -0.702, oldpeak r = +0.693, cos² de diagnóstico = 0.449), validado sin haber incluido el diagnóstico en la construcción factorial.",
    "Las Dimensiones 2 (8.67%) y 3 (7.85%) revelan heterogeneidad clínica independiente de la isquemia: perfil demográfico-lipídico femenino vs. masculino (Dim 2) y síndrome metabólico con glucemia elevada y angina típica (Dim 3).",
    "El clustering HCPC (k = 4) segmentó la cohorte en cuatro fenotipos accionables, destacando un grupo de alto riesgo isquémico (Clúster 4, 89.1% enfermedad cardíaca, v-test = +10.25) y un grupo metabólico-hipertensivo de riesgo intermedio (Clúster 3, 47.3% enfermedad), claramente diferenciados de los dos perfiles de bajo riesgo (Clúster 1 femenino y Clúster 2 joven).",
]
for c_txt in conclusions:
    p = doc.add_paragraph(c_txt, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)

out_docx = "Informe_Tarea4_AFM_UAO.docx"
doc.save(out_docx)
print(f"Saved {out_docx} successfully!")
