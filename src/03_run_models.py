# 03_run_models.py
# Runs every model for the paper and answers the central question:
#   "Did the Dodd-Frank expansion of Call Report fields actually improve
#    our ability to predict bank failures from public regulatory data?"
#
# Five stages, top to bottom:
#   1. Headline:           four model classes on the full feature set
#   2. Dodd-Frank value:   pre-DF vs modern feature sets, same model
#   3. Era stratified:     a model per regulatory era
#   4. Robustness:         unsupervised anomaly ensemble (MCA + IForest + AE)
#   5. Lead time:          how early does each method flag failed banks

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

# Models from sklearn / xgboost / lightgbm
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble       import IsolationForest, RandomForestClassifier
from sklearn.linear_model   import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics        import roc_auc_score, roc_curve, average_precision_score, confusion_matrix
from scipy                  import stats

import xgboost  as xgb
import lightgbm as lgb

# MCA (multiple correspondence analysis) + K-Modes for the unsupervised piece
import prince
from kmodes.kmodes import KModes
from collections import Counter

# Era classification lives in feature_eras.py next to this script
sys.path.insert(0, os.path.dirname(__file__))
from feature_eras import FEATURE_ERAS, pre_df_features, all_features

warnings.filterwarnings("ignore")

PROC = "data/processed"
TBL  = "tables"
os.makedirs(TBL, exist_ok=True)


# Load the analysis panel from script 02
df = pd.read_parquet(f"{PROC}/panel.parquet")
print(f"Loaded panel: {df.shape}")

target = "failure_t4"   # actual bank failure within 4 quarters
print(f"Target: {target} (positive rate = {df[target].mean()*100:.2f}%)")


# Helper - pull (X, y, mask, kept_features) for a feature list.
#
# Two strategies for missing values:
#   strategy="impute" (default) - median-impute NaN/inf per feature and KEEP
#       the row. This is the consensus approach in the bank-failure literature
#       (Kolari et al. 2002, Cole & White 2012, DeYoung & Torna 2013). Critical
#       for the extended 1995-onward panel because past_due_30_89 wasn't
#       reported until 2001Q1 - row-deletion would throw away ~575K obs.
#   strategy="drop" - strict: drop any row with NaN/inf in the feature columns.
#       Kept for robustness checks.
def clean_xy(df, features, target_col=target, strategy="impute"):
    avail = [f for f in features if f in df.columns]

    if strategy == "drop":
        mask = (df[avail + [target_col]].notna().all(axis=1)
                & np.isfinite(df[avail]).all(axis=1))
        X = df.loc[mask, avail].values
        y = df.loc[mask, target_col].values
        return X, y, mask, avail

    # impute: require non-null target, then median-fill NaN/inf per feature
    mask = df[target_col].notna()
    sub  = df.loc[mask, avail].copy()
    sub  = sub.replace([np.inf, -np.inf], np.nan)
    medians = sub.median(numeric_only=True)
    sub  = sub.fillna(medians).fillna(0)   # safety net for all-NaN columns
    X = sub.values
    y = df.loc[mask, target_col].values
    return X, y, mask, avail


# =====================================================================
# STAGE 1 - Headline: how predictive are Call Report ratios?
# =====================================================================
print("\n" + "=" * 68)
print("STAGE 1 - Headline: Predictive value of full Call Report feature set")
print("=" * 68)

X, y, mask, used_feats = clean_xy(df, all_features())
print(f"Sample: {len(y):,} obs · {y.sum():,} positives "
      f"({y.mean()*100:.3f}% rate) · {len(used_feats)} features")

# Standardize once — linear models need it, tree models don't
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Class-balance ratio for XGBoost
pos_rate = y.mean()
spw_stage1 = (1 - pos_rate) / pos_rate

# Four model classes — sizes tuned to keep total runtime under ~10 min on a laptop
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42),

    # NOTE: LASSO is run separately below as a feature-selection diagnostic,
    # not as a headline classifier. This follows the bank-failure literature
    # (Petropoulos et al. 2020, Lang et al. 2018, Beutel et al. 2019):
    # tree-based models dominate the leaderboard; LASSO is used only to
    # identify which ratios carry independent signal.

    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1),

    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw_stage1, eval_metric="auc",
        random_state=42, n_jobs=-1),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
}

# Only plain logit needs standardized inputs; everything else is tree-based
LINEAR_MODELS = {"Logistic Regression"}

# 70/30 random split — much faster than 5-fold CV on 550K rows and still honest
X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te = train_test_split(
    X, y, X_s, test_size=0.3, random_state=42, stratify=y)

# Loop the models and collect a flat list of results (one row per model)
results = []
print(f"\n{'Model':<22s} {'OOS ROC-AUC':>12s} {'OOS Avg Prec':>12s} {'Time (s)':>10s}")
print("-" * 60)

for name, m in models.items():
    t0 = time.time()
    if name in LINEAR_MODELS:
        X_train, X_test = Xs_tr, Xs_te
    else:
        X_train, X_test = X_tr, X_te

    m.fit(X_train, y_tr)
    y_prob = m.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)

    results.append({
        "model":  name,
        "auc":    auc,
        "ap":     ap,
        "y_prob": y_prob,
        "y_test": y_te,
    })
    print(f"{name:<22s} {auc:>12.4f} {ap:>12.4f} {time.time()-t0:>10.1f}")


# LASSO variable selection (diagnostic only, not in the leaderboard above).
# Fit on a 50K stratified subsample of the training data - standard practice
# for L1 selection at this sample size. Tells us which ratios carry
# independent predictive signal after shrinkage.
print("\nFitting LASSO for variable selection (50K subsample)...")
rng_l = np.random.default_rng(42)
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
n_pos   = min(int(50_000 * y_tr.mean()), len(pos_idx))
n_neg   = 50_000 - n_pos
lasso_idx = np.concatenate([
    rng_l.choice(pos_idx, n_pos, replace=False),
    rng_l.choice(neg_idx, n_neg, replace=False),
])
lasso = LogisticRegressionCV(
    Cs=3, cv=3, penalty="l1", solver="liblinear",
    class_weight="balanced", scoring="roc_auc",
    max_iter=200, random_state=42, n_jobs=1, tol=1e-3,
)
t_lasso = time.time()
lasso.fit(Xs_tr[lasso_idx], y_tr[lasso_idx])
print(f"LASSO fit in {time.time()-t_lasso:.1f}s")
coefs = lasso.coef_[0]
n_nonzero = int((coefs != 0).sum())
print(f"\nLASSO selected {n_nonzero}/{len(used_feats)} features (rest shrunk to zero)")
print(f"Optimal C from CV: {lasso.C_[0]:.4f}")

lasso_coefs = pd.DataFrame({
    "feature":         used_feats,
    "coefficient":     coefs,
    "abs_coefficient": np.abs(coefs),
    "selected":        (coefs != 0).astype(int),
}).sort_values("abs_coefficient", ascending=False)
lasso_coefs.to_csv(f"{TBL}/lasso_coefficients.csv", index=False)
print("\nTop 10 LASSO-selected features (by |coefficient|):")
print(lasso_coefs[lasso_coefs["selected"] == 1].head(10).to_string(index=False))


# Pick the winner (max AUC), save model + scaler + feature list
best_row = max(results, key=lambda r: r["auc"])
best     = best_row["model"]
print(f"\nBest model: {best} (AUC = {best_row['auc']:.4f})")

best_model = models[best]
X_use = X_s if best in LINEAR_MODELS else X
best_model.fit(X_use, y)
joblib.dump(best_model, f"{PROC}/best_model.joblib")
joblib.dump(scaler,     f"{PROC}/scaler.joblib")
joblib.dump(used_feats, f"{PROC}/feature_names.joblib")

# ROC curve for the best model on the held-out split
fpr, tpr, thr = roc_curve(best_row["y_test"], best_row["y_prob"])
n_pts = min(len(fpr), len(tpr), len(thr))
pd.DataFrame({
    "fpr":       fpr[:n_pts],
    "tpr":       tpr[:n_pts],
    "threshold": thr[:n_pts],
}).to_csv(f"{TBL}/roc_curve.csv", index=False)

# Stage 1 leaderboard
pd.DataFrame([
    {"model": r["model"], "roc_auc": r["auc"], "avg_precision": r["ap"]}
    for r in results
]).to_csv(f"{TBL}/stage1_model_comparison.csv", index=False)

# Confusion matrix at Youden's-J optimal threshold
opt_thr = thr[np.argmax(tpr - fpr)]
y_pred  = (best_row["y_prob"] >= opt_thr).astype(int)
cm = confusion_matrix(best_row["y_test"], y_pred)
print(f"\n{best} confusion matrix at threshold={opt_thr:.3f}:")
print(f"   TN={cm[0,0]:>8,d}  FP={cm[0,1]:>8,d}")
print(f"   FN={cm[1,0]:>8,d}  TP={cm[1,1]:>8,d}")
print(f"   Sensitivity (Recall) = {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
print(f"   Specificity          = {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%")
print(f"   Precision            = {cm[1,1]/(cm[0,1]+cm[1,1])*100:.1f}%")


# =====================================================================
# STAGE 2 - The Dodd-Frank Value Test
# Same model class, two feature sets:
#   (A) Modern: all 48 ratios (includes 7 post-2010 fields)
#   (B) Pre-DF: 41 ratios derivable from pre-2010 Call Report schedules
# AUC delta = predictive value of the post-2010 reporting expansion.
# Test period = 2018-2025 (modern era when both sets actually exist).
# =====================================================================
print("\n" + "=" * 68)
print("STAGE 2 - The Dodd-Frank Value Test")
print("=" * 68)
print("Question: did the post-2010 expansion of regulatory reporting add")
print("          predictive value for bank failure prediction?")

TEST_START = "20180101"
train_mask = df["REPDTE"] <  TEST_START
test_mask  = df["REPDTE"] >= TEST_START

print(f"\nTraining set: REPDTE <  {TEST_START}  "
      f"({train_mask.sum():,} obs, {df.loc[train_mask, target].sum():,} positives)")
print(f"Test set:     REPDTE >= {TEST_START}  "
      f"({test_mask.sum():,} obs, {df.loc[test_mask, target].sum():,} positives)")


# Train on the train mask, evaluate on the test mask; return AUC + 95% CI.
# `model_cls` is the constructor and `model_kw` are kwargs for it.
def train_test_auc(features, model_cls, model_kw, label):
    avail = [f for f in features if f in df.columns]

    # Keep every row with a non-null target; median-impute NaN/inf in features.
    # Same imputation logic as clean_xy() above - consistent with the literature
    # standard for handling pre-2001 missing fields.
    tr_m = train_mask & df[target].notna()
    te_m = test_mask  & df[target].notna()

    def _impute(sub):
        sub = sub.replace([np.inf, -np.inf], np.nan)
        return sub.fillna(sub.median(numeric_only=True)).fillna(0)

    X_tr = _impute(df.loc[tr_m, avail].copy()).values
    X_te = _impute(df.loc[te_m, avail].copy()).values
    y_tr = df.loc[tr_m, target].values
    y_te = df.loc[te_m, target].values

    # Standardize for linear / LASSO models
    if model_cls in (LogisticRegression, LogisticRegressionCV):
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    m = model_cls(**model_kw)
    m.fit(X_tr, y_tr)
    p_te = m.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, p_te)
    ap  = average_precision_score(y_te, p_te)

    # 95% CI via 500 bootstrap resamples of the test set
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(500):
        idx = rng.choice(len(y_te), len(y_te), replace=True)
        # Skip a resample if it accidentally landed all-one-class
        if y_te[idx].sum() > 0 and (1 - y_te[idx]).sum() > 0:
            boots.append(roc_auc_score(y_te[idx], p_te[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "label":      label,
        "auc":        auc,
        "ap":         ap,
        "ci_lo":      lo,
        "ci_hi":      hi,
        "n_features": len(avail),
        "y_true":     y_te,
        "y_prob":     p_te,
    }


# XGBoost is the primary model. Class-balance ratio computed on the train mask.
spw = (1 - df.loc[train_mask, target].mean()) / df.loc[train_mask, target].mean()
xgb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              scale_pos_weight=spw, eval_metric="auc",
              random_state=42, n_jobs=-1, verbosity=0)

print("\nTesting both feature sets with the same model architecture (XGBoost):")
modern_xgb = train_test_auc(all_features(),    xgb.XGBClassifier, xgb_kw, "Modern (all 48 features)")
predf_xgb  = train_test_auc(pre_df_features(), xgb.XGBClassifier, xgb_kw, "Pre-Dodd-Frank only (41 features)")

# Robustness check #1: plain logit
print("\n(Also testing with logistic regression for robustness)")
lr_kw = dict(max_iter=500, class_weight="balanced", random_state=42)
modern_lr = train_test_auc(all_features(),    LogisticRegression, lr_kw, "Modern (logit)")
predf_lr  = train_test_auc(pre_df_features(), LogisticRegression, lr_kw, "Pre-DF (logit)")

# NOTE: LASSO is NOT included in the Stage 2 Dodd-Frank Value Test.
# Following the consensus in the bank-failure literature (Petropoulos et al.
# 2020, Lang et al. 2018), Stage 2 uses XGBoost as the primary model and
# plain logit as the robustness check. LASSO's role in this paper is
# variable selection only - see the diagnostic above in Stage 1.

# Diebold-Mariano test on the squared-error series between the two XGBoost models
d_t = ((modern_xgb["y_true"] - predf_xgb["y_prob"]) ** 2
       - (modern_xgb["y_true"] - modern_xgb["y_prob"]) ** 2)
dm_stat = d_t.mean() / np.sqrt(d_t.var(ddof=1) / len(d_t))
dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

# Print results
print("\n" + "-" * 78)
print(f"{'Feature Set':<32s} {'# Feats':>8s} {'OOS AUC':>10s} {'95% CI':>16s} {'Avg Prec':>10s}")
print("-" * 78)
for r in [predf_xgb, modern_xgb]:
    print(f"{r['label']:<32s} {r['n_features']:>8d} {r['auc']:>10.4f} "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  {r['ap']:>10.4f}")
print("-" * 78)

delta = modern_xgb["auc"] - predf_xgb["auc"]
print(f"\nDODD-FRANK VALUE (AUC delta): {delta:+.4f}")
print(f"Diebold-Mariano test:  DM={dm_stat:>+.3f}  p={dm_pval:.4f}")

if   dm_pval < 0.01: star = "***"
elif dm_pval < 0.05: star = "**"
elif dm_pval < 0.10: star = "*"
else:                star = "(n.s.)"

if   delta > 0 and dm_pval < 0.05: verdict = "Modern feature set significantly better"
elif delta < 0 and dm_pval < 0.05: verdict = "Pre-DF set significantly better"
else:                              verdict = "No statistically significant difference"

print(f"Significance: {star}")
print(f"Verdict: {verdict}")

delta_lr = modern_lr["auc"] - predf_lr["auc"]
print(f"\nRobustness check (logit): pre-DF AUC={predf_lr['auc']:.4f}, "
      f"modern AUC={modern_lr['auc']:.4f}, delta={delta_lr:+.4f}")

# Save the comparison table (one row per model x feature-set)
rows = []
for model_name, pre, mod in [
    ("XGBoost",     predf_xgb,   modern_xgb),
    ("Logit",       predf_lr,    modern_lr),
]:
    for setname, r in [("Pre-Dodd-Frank", pre), ("Modern (full)", mod)]:
        rows.append({
            "model":       model_name,
            "feature_set": setname,
            "n_features":  r["n_features"],
            "auc":         r["auc"],
            "ci_lo":       r["ci_lo"],
            "ci_hi":       r["ci_hi"],
            "avg_prec":    r["ap"],
        })

pd.DataFrame(rows).to_csv(f"{TBL}/dodd_frank_value_test.csv", index=False)


# =====================================================================
# STAGE 3 - Era-stratified analysis
# Train a separate model on each regulatory era to see how performance
# and feature drivers shift over time.
# =====================================================================
print("\n" + "=" * 68)
print("STAGE 3 - Era-stratified analysis")
print("=" * 68)

# Now that we have 1995Q3 data, split the pre-DF era into pre-GFC and GFC.
# This separates calm-period predictive performance from crisis-period.
eras = {
    "Pre-GFC pre-DF (1995Q3-2007Q4)": ("19950701", "20080101"),
    "GFC pre-DF     (2008Q1-2010Q3)": ("20080101", "20101001"),
    "Dodd-Frank era (2010Q4-2014Q4)": ("20101001", "20150101"),
    "Basel III era  (2015Q1-2022Q4)": ("20150101", "20230101"),
    "Post-SVB era   (2023Q1-2025Q4)": ("20230101", "20260101"),
}

era_results = []
for era_label, (start, end) in eras.items():
    em    = (df["REPDTE"] >= start) & (df["REPDTE"] < end)
    n     = em.sum()
    n_pos = df.loc[em, target].sum()

    if n_pos < 5:
        print(f"  {era_label}: too few positives ({n_pos}) - skipping")
        continue

    # Pre-DF era only has pre-DF features available; later eras get the full set
    # Both pre-DF eras only get the pre-DF feature set
    if era_label.startswith("Pre-GFC") or era_label.startswith("GFC pre-DF"):
        feats = pre_df_features()
    else:
        feats = all_features()
    avail = [f for f in feats if f in df.columns]

    # Same impute-don't-drop policy as Stages 1-2
    sub_mask = em & df[target].notna()
    sub      = df.loc[sub_mask, avail].copy().replace([np.inf, -np.inf], np.nan)
    sub      = sub.fillna(sub.median(numeric_only=True)).fillna(0)
    X_sub    = sub.values
    y_sub    = df.loc[sub_mask, target].values

    m = xgb.XGBClassifier(**xgb_kw)
    if y_sub.sum() >= 10:
        skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_prob = cross_val_predict(m, X_sub, y_sub, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        auc    = roc_auc_score(y_sub, y_prob)
        ap     = average_precision_score(y_sub, y_prob)
    else:
        auc, ap = np.nan, np.nan

    era_results.append({
        "era":         era_label,
        "n_obs":       int(n),
        "n_failures":  int(n_pos),
        "n_features":  len(avail),
        "roc_auc":     auc,
        "avg_prec":    ap,
    })
    print(f"  {era_label:<35s} n={n:>7,}  fail={n_pos:>4}  AUC={auc:.4f}  AP={ap:.4f}")

pd.DataFrame(era_results).to_csv(f"{TBL}/era_stratified_models.csv", index=False)


# =====================================================================
# STAGE 4 - Unsupervised anomaly ensemble (chunked for memory safety)
# =====================================================================
print("\n" + "=" * 68)
print("STAGE 4 - Robustness: unsupervised anomaly ensemble")
print("=" * 68)


# --- (1) MCA + K-Modes ---
print("\n[1/3] MCA + K-Modes...")

# Fit on a 50K-row subsample (literature standard for unsupervised methods)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_all), min(50_000, len(X_all)), replace=False)
X_samp = X_all.iloc[sample_idx]

# Discretize each column into 5 quantile bins (categorical for MCA)
disc_samp = pd.DataFrame(index=X_samp.index)
discs = {}
for c in X_samp.columns:
    if X_samp[c].nunique() < 5:
        disc_samp[c] = "0"
        continue
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    disc_samp[c] = kbd.fit_transform(X_samp[[c]]).flatten().astype(int).astype(str)
    discs[c] = kbd

# Fit MCA (10 dims) and K-Modes (5 clusters) on the subsample
print("   fitting MCA + K-Modes on 50K subsample...")
mca = prince.MCA(n_components=10, random_state=42).fit(disc_samp)
mca_samp = mca.row_coordinates(disc_samp)

km = KModes(n_clusters=5, init="Huang", n_init=3, random_state=42, verbose=0)
labels_samp = km.fit_predict(disc_samp.values)

# Cluster centers in MCA space
centers = np.zeros((5, 10))
for k in range(5):
    if (labels_samp == k).any():
        centers[k] = mca_samp.loc[labels_samp == k].mean().values

# Project ALL rows into the same MCA space, in chunks to keep memory low.
# At 943K x 5 x 10 the full distance tensor is ~1.5GB - chunking fixes that.
print("   projecting full panel in chunks...")
CHUNK = 50_000
n_rows = len(X_all)
distances = np.empty(n_rows, dtype=np.float32)
clusters  = np.empty(n_rows, dtype=np.int32)
all_coords_list = []

for start in range(0, n_rows, CHUNK):
    end = min(start + CHUNK, n_rows)
    chunk_X = X_all.iloc[start:end]

    # Discretize this chunk using the fitted bin edges
    disc_chunk = pd.DataFrame(index=chunk_X.index)
    for c in chunk_X.columns:
        if c in discs:
            try:
                disc_chunk[c] = discs[c].transform(chunk_X[[c]]).flatten().astype(int).astype(str)
            except ValueError:
                disc_chunk[c] = "0"
        else:
            disc_chunk[c] = "0"

    # Project into MCA space, compute distance to nearest cluster center
    coords_chunk = mca.row_coordinates(disc_chunk).values.astype(np.float32)
    all_coords_list.append(coords_chunk)

    diffs = coords_chunk[:, None, :] - centers[None, :, :]
    d_per_k = np.sqrt((diffs ** 2).sum(axis=2))
    distances[start:end] = d_per_k.min(axis=1)
    clusters[start:end]  = d_per_k.argmin(axis=1)

    if start % 200_000 == 0:
        print(f"     chunked {end:,}/{n_rows:,} rows")

mca_all_coords = pd.DataFrame(np.vstack(all_coords_list), index=X_all.index)
del all_coords_list

# Bonus penalty if the row is in a small/sparse cluster (likely outlier cluster)
counts = Counter(clusters)
sparse = [k for k, v in counts.items() if v < len(clusters) * 0.02]
bonus  = np.where(np.isin(clusters, sparse), np.percentile(distances, 90), 0)
mca_raw   = distances + bonus
mca_score = (mca_raw - mca_raw.min()) / (mca_raw.max() - mca_raw.min() + 1e-10)

joblib.dump({
    "row_coords_full":   mca_all_coords,
    "col_coords":        mca.column_coordinates(disc_samp),
    "explained_inertia": mca.percentage_of_variance_,
}, f"{PROC}/mca_results.joblib")
print(f"   top 5%: {(mca_score >= np.percentile(mca_score, 95)).sum():,} flagged")


# --- (2) Isolation Forest ---
print("\n[2/3] Isolation Forest...")
iso_scaler = StandardScaler()
X_sc_full = iso_scaler.fit_transform(X_all)
iforest = IsolationForest(n_estimators=200, contamination=0.03,
                          random_state=42, n_jobs=-1)
iforest.fit(X_sc_full)
if_raw   = -iforest.decision_function(X_sc_full)
if_score = (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-10)
joblib.dump(iforest, f"{PROC}/iforest_model.joblib")
print(f"   flagged: {(iforest.predict(X_sc_full) == -1).sum():,}")


# --- (3) Autoencoder (sklearn MLPRegressor) ---
# Trained on a 200K subsample to keep runtime/memory bounded, then scores all rows.
# This is standard practice for autoencoder-based anomaly detection.
print("\n[3/3] Autoencoder (sklearn MLPRegressor)...")
ae_sample_n = min(200_000, len(X_sc_full))
ae_idx = rng.choice(len(X_sc_full), ae_sample_n, replace=False)
X_ae_train = X_sc_full[ae_idx]

input_dim   = X_sc_full.shape[1]
hidden      = max(5, input_dim // 2)
bottleneck  = max(3, input_dim // 4)

ae = MLPRegressor(
    hidden_layer_sizes=(hidden, bottleneck, hidden),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    batch_size=512,
    max_iter=60,
    early_stopping=True,
    n_iter_no_change=8,
    validation_fraction=0.15,
    random_state=42,
    verbose=False,
)
print(f"   fitting AE on {ae_sample_n:,} rows, scoring all {len(X_sc_full):,}...")
ae.fit(X_ae_train, X_ae_train)

# Score all rows in chunks (predict is memory-hungry)
recon = np.empty(len(X_sc_full), dtype=np.float32)
for start in range(0, len(X_sc_full), CHUNK):
    end = min(start + CHUNK, len(X_sc_full))
    pred = ae.predict(X_sc_full[start:end])
    recon[start:end] = ((pred - X_sc_full[start:end]) ** 2).mean(axis=1)

ae_score = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
joblib.dump(ae, f"{PROC}/autoencoder.joblib")
print(f"   top 5%: {(ae_score >= np.percentile(ae_score, 95)).sum():,} flagged")


# --- Ensemble: row flagged if in top 5% of at least 2 of 3 methods ---
print("\nEnsemble (top 5% on >= 2 of 3):")
df["mca_score"] = mca_score
df["if_score"]  = if_score
df["ae_score"]  = ae_score
df["mca_flag"]  = (mca_score >= np.percentile(mca_score, 95)).astype(int)
df["if_flag"]   = (if_score  >= np.percentile(if_score,  95)).astype(int)
df["ae_flag"]   = (ae_score  >= np.percentile(ae_score,  95)).astype(int)
df["n_flags"]   = df["mca_flag"] + df["if_flag"] + df["ae_flag"]
df["ensemble_flag"]   = (df["n_flags"] >= 2).astype(int)
df["composite_score"] = (mca_score + if_score + ae_score) / 3

print(f"   ensemble flagged: {df['ensemble_flag'].sum():,} "
      f"({df['ensemble_flag'].mean()*100:.2f}%)")
print(pd.crosstab(df["ensemble_flag"], df[target], margins=True))


# =====================================================================
# STAGE 5 - Lead time: how early does each method flag failed banks?
# =====================================================================
print("\n" + "=" * 68)
print("STAGE 5 - Lead time: how early does each method flag failed banks?")
print("=" * 68)

# Score every bank-quarter under the best supervised model
LINEAR_MODELS = {"LogisticRegression"}
X_full_clean = df[used_feats].replace([np.inf, -np.inf], np.nan)
X_full_clean = X_full_clean.fillna(X_full_clean.median()).fillna(0).values
if best in LINEAR_MODELS:
    X_full_clean = scaler.transform(X_full_clean)

df["best_model_prob"] = best_model.predict_proba(X_full_clean)[:, 1]
df["best_model_flag"] = (df["best_model_prob"]
                         >= np.percentile(df["best_model_prob"], 95)).astype(int)

failed_banks = df[df[target] == 1]["CERT"].unique()
print(f"   {len(failed_banks):,} unique distressed banks")

methods = {
    "Best supervised":  "best_model_flag",
    "Ensemble":         "ensemble_flag",
    "MCA+K-Modes":      "mca_flag",
    "Isolation Forest": "if_flag",
    "Autoencoder":      "ae_flag",
}

lead_results = []
for method, col in methods.items():
    leads = []
    for cert in failed_banks:
        bd  = df[df["CERT"] == cert].sort_values("REPDTE")
        d_q = bd[bd[target] == 1]["REPDTE"]
        if len(d_q) == 0:
            continue
        last_d  = d_q.max()
        flagged = bd[(bd[col] == 1) & (bd["REPDTE"] <= last_d)]
        if len(flagged) == 0:
            continue
        ql = list(bd["REPDTE"])
        if last_d in ql and flagged["REPDTE"].min() in ql:
            leads.append(ql.index(last_d) - ql.index(flagged["REPDTE"].min()))

    if leads:
        lead_results.append({
            "method":               method,
            "avg_lead_quarters":    np.mean(leads),
            "median_lead_quarters": np.median(leads),
            "n_detected":           len(leads),
            "n_total":              len(failed_banks),
            "detection_rate":       len(leads) / len(failed_banks),
        })
        print(f"   {method:<20s} avg={np.mean(leads):>4.1f}Q  "
              f"median={np.median(leads):>3.0f}Q  "
              f"detected {len(leads):>4}/{len(failed_banks)} "
              f"({len(leads)/len(failed_banks)*100:.1f}%)")

pd.DataFrame(lead_results).to_csv(f"{TBL}/lead_time.csv", index=False)


# Save the scored panel for figures
df.to_parquet(f"{PROC}/panel_with_scores.parquet", index=False)
print(f"\nSaved scored panel ({df.shape[0]:,} x {df.shape[1]}) "
      f"to {PROC}/panel_with_scores.parquet")
print("\n" + "=" * 68)
print("ALL STAGES COMPLETE")
print("=" * 68)
