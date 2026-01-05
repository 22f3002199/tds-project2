# Autolysis

Autolysis is a command-line Python tool for **automated exploratory data analysis (EDA)** and **AI-generated dataset narratives**.
Given a CSV file, it analyzes the data, generates visualizations, detects outliers, and writes an LLM-generated analysis to a README.

---

## Features

* Automatic dataset profiling (summary stats, missing values, skewness)
* Correlation heatmap
* PCA visualization
* Outlier detection (Isolation Forest)
* AI-generated narrative report
* Robust CSV loading with multiple encodings

---

## Requirements

* **Python ≥ 3.11**
* **uv** (recommended)

> This script follows the `uv` inline-dependency format.
> When run with `uv`, all dependencies are resolved automatically.

---

## Setup

Set the API token used for narrative generation:

```bash
export AIPROXY_TOKEN="your_api_token_here"
```

---

## Usage (Recommended)

Run directly with **uv** — no manual dependency installation required:

```bash
uv autolysis.py dataset.csv
```

`uv` will:

* Create an isolated environment
* Install all required dependencies
* Execute the script

---

## Alternative Usage (Without uv)

If you choose **not** to use `uv`, you must install dependencies manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy requests
```

Then run:

```bash
python autolysis.py dataset.csv
```

---

## Output

Creates a folder named:

```
analysis_<dataset_name>/
```

Contents:

```
README.md               # AI-generated analysis narrative
correlation_heatmap.png # Feature correlations
pca_analysis.png        # PCA plot
outliers.png            # Outlier detection
```

---

## Notes

* PCA and outlier plots use the **first two numerical columns**, change it to fit your data
* Missing numerical values are mean-imputed
* LLM output depends on API availability and token limits

---
