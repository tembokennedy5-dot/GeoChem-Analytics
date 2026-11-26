# GeoChem-Analytics

![Tests](https://github.com/tembokennedy5-dot/GeoChem-Analytics/actions/workflows/tests.yml/badge.svg)

**An interactive Streamlit toolkit for exploration geochemistry and compositional data analysis (CoDa).**

Multi-element assay data is compositional: the values are parts of a whole, so raw correlations and PCA can show patterns that come from the constant sum rather than from geology. GeoChem-Analytics applies log-ratio transforms first, then finds multivariate anomalies and element associations and maps them in space, so you can separate real mineralisation signals from closure artefacts.

## Features

| Tab | What it does |
|---|---|
| **The data** | Loads CSV/XLSX assays, handles below-detection values (e.g. `<0.5`), and previews the dataset |
| **Raw vs CoDa** | Compares correlation heatmaps side by side for raw data and ALR / CLR / ILR transformed data |
| **Biplot** | CLR compositional biplot for spotting pathfinder element suites |
| **Ternary** | Ternary explorer for any three elements, with optional centring |
| **Outliers** | Robust Mahalanobis distance (Minimum Covariance Determinant) on ILR coordinates, a χ² 97.5% cut-off, a distance–distance plot, and an anomaly map |
| **Clustering / MDS** | K-means clustering and classical multidimensional scaling |
| **PCA** | Loadings, score plots, and PC score maps in space |
| **Factor analysis** | Varimax-rotated factors, factor score maps, and an FA vs PCA comparison |

Each tab also includes interpretation notes and a guiding question that link the statistics back to geological process. Results can be exported as a scored CSV, with each sample classed as *anomalous target* or *background*.

## Quick start

```bash
git clone https://github.com/tembokennedy5-dot/GeoChem-Analytics.git
cd GeoChem-Analytics
pip install -r requirements.txt
streamlit run geochem_app.py
```

The app opens at `http://localhost:8501` with a built-in **synthetic demo dataset**: 180 soil samples with Cu, Pb, Zn, Au, As, Ag, Fe, S, Sb and Cd, plus 12 planted Cu–Au–As–Sb anomalies. To use your own data, untick the demo box and upload a spreadsheet with sample coordinates and assay columns.

You can also open the repo in **GitHub Codespaces**. The dev container installs everything and launches the app automatically.

## Tests

```bash
pip install pytest
pytest -q
```

The 12 unit tests check the core maths (CLR rows sum to zero, ILR preserves Aitchison distance, below-detection handling, varimax orthogonality, MDS distance recovery). They also include a smoke test that renders the full app. A GitHub Actions workflow runs them on every push.

## Windows executable

A GitHub Actions workflow (`.github/workflows/build.yml`) builds a standalone `.exe` with PyInstaller on every push to `main`. You can download it from the **Actions** tab under the *GeoChem-EXE* artifact.

## Methods

- Log-ratio transforms: ALR, CLR, ILR (Aitchison geometry)
- Robust multivariate outlier detection: MCD covariance with a χ² threshold
- PCA, factor analysis with varimax rotation, K-means, classical MDS

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · SciPy · Plotly · Matplotlib · Seaborn

## Author

**Tembo Kennedy**, MSc Data Science & Exploration Geology
