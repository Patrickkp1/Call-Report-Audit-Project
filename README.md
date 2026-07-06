# FDIC Bank Failure Prediction — Dodd-Frank Value Test

A reproducible research pipeline that uses FDIC Call Report data (1995Q3–2025Q4) to test whether the post-2010 Dodd-Frank expansion of regulatory reporting actually improved our ability to predict bank failures from public data.

**TL;DR**: It didn't. On a fair head-to-head test with 61 pre-Dodd-Frank quarters vs. 61 post-Dodd-Frank quarters, a model using only the 41 ratios available before Dodd-Frank achieves out-of-sample ROC-AUC = 0.8929 vs. 0.8850 for the full 48-ratio modern feature set. The pre-DF model is statistically significantly better (Diebold-Mariano p < 0.001).

## What this project answers

> Does the post-Dodd-Frank expansion of Call Report disclosures (uninsured/insured deposit estimates, brokered-deposit breakdown, Basel III RWA) carry independent predictive signal for bank failure that wasn't already in the pre-2010 schedules?

Three converging pieces of evidence say **no**:

1. **The Dodd-Frank Value Test**: Same XGBoost architecture, same 2018–2025 test period. Pre-DF (41 features) beats Modern (48 features) by 0.8 AUC points, p < 0.001.
2. **Era-stratified analysis**: Pre-DF features achieve AUC = 0.984 in the calm 1995–2007 period — the signal isn't crisis-dependent.
3. **SHAP importance**: 17 of the top 20 predictors in the modern model are pre-DF features (capital, asset quality, and earnings ratios).

## Headline results

| Stage | Result |
|---|---|
| **Headline model** (LightGBM, 48-feature spec, 70/30 split) | AUC = **0.9960** OOS |
| **Dodd-Frank Value Test** (XGBoost, test 2018–2025) | Pre-DF 0.893 > Modern 0.885, p < 0.001 |
| **Era-stratified AUC** | Pre-GFC 1995–2007: 0.984 · GFC: 0.986 · DF: 0.996 · Basel III: 0.998 · Post-SVB: 0.962 |
| **Unsupervised ensemble** | 75.7% recall on failures at 3.94% flag rate |
| **Lead time** (best supervised) | Detected 574/574 failed banks · median lead 16Q (4 years) |

## Data

- **Source**: [FDIC BankFind public API](https://api.fdic.gov/banks/financials) (Call Report) + [FDIC failed-bank list](https://www.fdic.gov/bank-failures/download-data.csv)
- **Period**: 1995Q3 – 2025Q4 (122 quarters)
- **Panel**: 943,615 bank-quarters across 14,753 unique banks
- **Target**: `failure_t4` = 1 if the bank fails within the next 4 quarters
- **Features**: 48 financial ratios in 7 buckets (capital, asset quality, earnings, liquidity, concentration, off-balance-sheet, growth)

The 1995Q3 start was deliberately chosen so the **pre-Dodd-Frank window (1995Q3–2010Q3)** matches the **post-Dodd-Frank window (2010Q4–2025Q4)** in length exactly: 61 quarters each. This avoids the v2 weakness where the pre-DF baseline was only 11 GFC-era quarters.

| Era | Quarters | Bank-quarters | Failures |
|---|---|---|---|
| Pre-DF (1995Q3–2010Q3) | **61** | 590,023 | 1,676 |
| Post-DF (2010Q4–2025Q4) | **61** | 353,592 | 789 |

## Repository structure

```
fdic_anomaly_v3/
├── src/
│   ├── 01_load_data.py          # Pulls Call Report + failed-bank list from FDIC API
│   ├── 02_build_ratios_stats.py # Builds 48 ratios, winsorizes, computes peer z-scores
│   ├── 03_run_models.py         # All 5 stages: headline, Dodd-Frank test, eras, ensemble, lead-time
│   ├── 04_make_figures.py       # Generates all 8 figures for the paper
│   └── feature_eras.py          # Hand-classifies each ratio by regulatory era
├── data/
│   ├── raw/                     # One parquet per quarter (auto-cached, ~168 MB)
│   ├── processed/               # Final analysis panel + trained models
│   └── outcomes/                # Failed-bank list
├── tables/                      # CSV outputs for every result reported in the paper
├── figures/                     # PNG + PDF for each paper figure
├── docs/
│   ├── fdic_anomaly_paper.tex   # 27-page working paper
│   ├── fdic_anomaly_paper.pdf
│   ├── interview_cheat_sheet.md # One-pager for Schwab MV interview prep
│   └── resume_bullets.md
└── README.md
```

## How to reproduce

### Prerequisites

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

(See `requirements.txt` for exact versions. Core deps: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, prince, kmodes, requests, joblib, matplotlib.)

### Pipeline

Run scripts top to bottom. The full pipeline takes about 20–25 minutes on a 2-core / 8 GB machine; the heaviest steps are the API pull (~10 min, mostly latency-bound) and the Random Forest fit on 660K rows (~6 min).

```bash
# 1. Pull raw Call Reports from FDIC API (122 quarters)
python3 src/01_load_data.py

# 2. Build the 48-ratio analysis panel (1 panel.parquet)
python3 src/02_build_ratios_stats.py

# 3. Run all 5 model stages (supervised + unsupervised ensemble + lead-time)
python3 src/03_run_models.py

# 4. Generate all paper figures
python3 src/04_make_figures.py

# 5. Compile the paper (requires pdflatex)
cd docs && pdflatex -interaction=nonstopmode fdic_anomaly_paper.tex
pdflatex -interaction=nonstopmode fdic_anomaly_paper.tex  # 2nd pass for refs
```

All scripts use only relative paths and write into the project directory — no config files, no environment variables, no manual setup beyond `pip install`.

## Methodology choices and the literature

This project follows the consensus in the modern bank-failure literature. Key choices, with the papers we're following:

**Tree-based models as primary classifiers**.
Petropoulos, Siakoulis, Stavroulakis & Vlachogiannakis (2020, *International Journal of Forecasting*), Lang, Peltonen & Sarlin (2018, ECB working paper), and Beutel, List & von Schweinitz (2019, *Journal of Financial Stability*) all use tree-based methods as primary classifiers and treat plain logit as a benchmark. We use XGBoost, LightGBM, and Random Forest, with logit as the linear benchmark. LightGBM is the champion at AUC = 0.9960.

**LASSO as a diagnostic, not a classifier**.
In every modern bank-failure ML survey (e.g., Tannous, Wang & Anuar 2024), LASSO appears as a *variable-selection tool*, not a primary model. We fit LASSO once on a 50K stratified subsample purely to identify which ratios carry independent signal. Result: at optimal regularization (C=1.0), all 48 features are retained — there is no sparse subset that would be sufficient.

**Median imputation for missing fields**.
One ratio (`past_due_30_89`) was not separately reported on Schedule RC-N until 2001Q1. For pre-2001 quarters we median-impute per the standard approach in Kolari, Glennon, Shin & Caputo (2002), Cole & White (2012), and DeYoung & Torna (2013). Row-deletion would have thrown out 575K observations from the pre-2001 era; tree models also handle NaN natively, providing a robustness check.

**Hand-classified era mapping**.
Each of the 48 ratios is hand-tagged in `src/feature_eras.py` as `pre_df` (available since at least 2001 from Schedule RC/RI/RC-C/RC-N), `dodd_frank` (added/expanded 2010Q3–2014Q4 via RC-O Memo item 2 and brokered-deposit breakdowns), `basel_iii` (revised RC-R Part II, 2015Q1+), or `post_svb` (2023Q2+ enhancements). Sources are documented in the docstrings of `feature_eras.py`.

**Unsupervised ensemble (MCA + Isolation Forest + Autoencoder)**.
This is the SR 11-7 "independent challenger model" angle. Each method captures a different anomaly type (cluster distance, density isolation, reconstruction error). Trained on subsamples of 50K (MCA, K-Modes) and 200K (autoencoder), then scored on the full 943K panel via chunked projection. This subsampling pattern matches Demyanyk & Hasan (2010) and Gogas, Papadimitriou & Agrapetidou (2018).

## Key results in detail

### Stage 1: Headline model comparison

70/30 stratified random split on the full 943K-row panel.

| Model | OOS ROC-AUC | Avg Precision | Time (s) |
|---|---|---|---|
| Logistic Regression | 0.9808 | 0.5200 | 6 |
| Random Forest | 0.9897 | 0.4978 | 348 |
| XGBoost | 0.9959 | 0.6387 | 13 |
| **LightGBM (champion)** | **0.9960** | 0.6260 | 15 |

LASSO diagnostic (50K subsample, Cs=3, cv=3): 48/48 features selected at C=1.0. Top 10 by |coefficient| are all capital, earnings, and concentration ratios — consistent with the SHAP analysis.

### Stage 2: The Dodd-Frank Value Test

Train: 1995Q3–2017Q4 (785,153 obs, 2,386 failures). Test: 2018Q1–2025Q4 (158,462 obs, 79 failures).

| Feature Set | # Features | OOS AUC | 95% CI | AP |
|---|---|---|---|---|
| **Pre-Dodd-Frank** | 41 | **0.8929** | [0.843, 0.940] | 0.0737 |
| Modern (full) | 48 | 0.8850 | [0.836, 0.934] | 0.0752 |

**ΔAUC = −0.0079** (pre-DF outperforms). Diebold-Mariano test: DM = −21.91, p < 0.001 ***.
Logit robustness: pre-DF AUC = 0.8818, modern = 0.8785, ΔAUC = −0.0033 (same direction).

### Stage 3: Era-stratified XGBoost

| Era | Quarters | Failures | AUC | AP |
|---|---|---|---|---|
| Pre-GFC pre-DF (1995Q3–2007Q4) | 50 | 159 | 0.9838 | 0.2432 |
| GFC pre-DF (2008Q1–2010Q3) | 11 | 1,517 | 0.9864 | 0.7001 |
| Dodd-Frank era (2010Q4–2014Q4) | 17 | 647 | 0.9959 | 0.7356 |
| Basel III era (2015Q1–2022Q4) | 32 | 114 | 0.9982 | 0.6879 |
| Post-SVB era (2023Q1–2025Q4) | 12 | 28 | 0.9616 | 0.3932 |

The pre-GFC AUC of 0.984 is the key new finding — the failure-prediction signal works in calm periods, not just crises.

### Stage 4: Unsupervised ensemble

Three independent methods, each flagging the top 5% of bank-quarters; ensemble flags a row if at least 2 of 3 agree.

| Method | Trained on | Top 5% flagged |
|---|---|---|
| MCA + K-Modes | 50K subsample | 47,181 |
| Isolation Forest | Full 943K | 28,309 |
| Autoencoder (MLPRegressor, 24-12-24) | 200K subsample | 47,181 |
| **Ensemble (≥2 of 3)** | — | **37,198 (3.94%)** |

Ensemble caught 1,866 of 2,465 actual failures = **75.7% recall** at a 3.94% panel flag rate.

### Stage 5: Lead-time analysis (574 failed banks)

| Method | Avg Lead | Median Lead | Detection Rate |
|---|---|---|---|
| **Best supervised (LightGBM)** | 23.2Q | 16Q | **100.0%** |
| Ensemble | 22.6Q | 18Q | 94.9% |
| MCA+K-Modes | 25.2Q | 21Q | 96.7% |
| Isolation Forest | 24.8Q | 21Q | 95.1% |
| Autoencoder | 26.8Q | 24Q | 96.5% |


## License

Code: MIT.
Data: FDIC Call Report and failed-bank list are public-domain federal records.

## Author

Patrick Poleshuk (UT Austin Economics MS, Senior Auditor in financial services).
This is a personal research project; views expressed do not represent any employer.
