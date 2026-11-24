# Tarea 4: Análisis Factorial Múltiple (AFM) y Clustering HCPC

**Universidad Autónoma de Occidente** — Maestría en Inteligencia Artificial y Ciencia de Datos  
**Asignatura:** Análisis Multivariado | **Profesora:** Claudia Lorena Montes  
**Fecha de entrega:** 28 de noviembre de 2026

## Descripción
Aplicación de la técnica multivariada **Análisis Factorial Múltiple (AFM)** mixto y **clustering HCPC** (Ward + consolidación K-Means) sobre el conjunto de datos reales **Heart Disease — Cohorte Cleveland** (UCI Machine Learning Repository, `n = 297` pacientes completos, `p = 14` variables).

## Estructura del repositorio
- `Informe_Tarea4_AFM_UAO.pdf` — Informe final (9 páginas, ≤ 15 páginas).
- `Informe_Tarea4_AFM_UAO.docx` — Informe editable en Word.
- `generate_outputs.py` — Script principal de limpieza, ajuste del AFM (`prince 0.23.1`), clustering HCPC, cálculo de contribuciones, `cos²`, valores test (`v-test`) y generación de gráficas.
- `build_report.py` — Generación automatizada del informe en `.docx`.
- `run_afm_analysis.py` — Script exploratorio de validación inicial.
- `heart_cleveland_raw.csv` / `heart_cleveland_clean.csv` / `heart_cleveland_clustered.csv` — Datos crudos, depurados y con asignación de clústeres.
- `figuras/` — Gráficas generadas (scree plot, nubes de grupos/variables/modalidades/individuos, contribuciones, dendrograma HCPC y mapa de calor `v-test`).
- `resultados/` — Tablas exactas en CSV (autovalores, correlaciones, contribuciones, `cos²`, métricas de clustering y `v-test`).

## Reproducibilidad
```bash
pip install prince pandas numpy scipy scikit-learn matplotlib seaborn python-docx
python generate_outputs.py
python build_report.py
```
