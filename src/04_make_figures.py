# 04_make_figures.py
# All paper figures, top to bottom:
#   1. Stage 1 model comparison (4-model bar chart)
#   2. ROC curve for the best model
#   3. Dodd-Frank Value Test bar chart (the money chart)
#   4. Era-stratified performance line chart
#   5. SHAP feature importance, color-coded by regulatory era
#   6. MCA biplot
#   7. Case study trajectories for SVB, Signature, Republic First

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

warnings.filterwarnings("ignore")

PROC = "data/processed"
FIG  = "figures"
TBL  = "tables"
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from feature_eras import FEATURE_ERAS

# Global plot style
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi":     200,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})

# Era color map (used across multiple figures)
ERA_COLORS = {
    "pre_df":     "#1976D2",   # blue
    "dodd_frank": "#F57C00",   # orange
    "basel_iii":  "#7B1FA2",   # purple
    "post_svb":   "#C62828",   # red
}
ERA_LABELS = {
    "pre_df":     "Pre-Dodd-Frank",
    "dodd_frank": "Dodd-Frank (2010Q4+)",
    "basel_iii":  "Basel III (2015Q1+)",
    "post_svb":   "Post-SVB (2023Q3+)",
}

# Case studies for the trajectory plots
case_studies = {
    "SVB":            {"cert": 24735, "name": "Silicon Valley Bank", "fail": "2023-03-10"},
    "Signature":      {"cert": 57053, "name": "Signature Bank",      "fail": "2023-03-12"},
    "Republic_First": {"cert": 27332, "name": "Republic First Bank", "fail": "2024-04-26"},
}


# Load the scored panel from script 03
df = pd.read_parquet(f"{PROC}/panel_with_scores.parquet")
print(f"Loaded scored panel: {df.shape}")


# =====================================================================
# 1. Stage 1 - Four-model comparison bar chart
# =====================================================================
print("\n[1] Stage 1 model comparison...")
m_cmp = pd.read_csv(f"{TBL}/stage1_model_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AUC bars
ax = axes[0]
bars = ax.barh(m_cmp["model"], m_cmp["roc_auc"], color="#1976D2", alpha=0.85, edgecolor="black")
for bar, v in zip(bars, m_cmp["roc_auc"]):
    ax.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(min(m_cmp["roc_auc"]) - 0.005, 1.001)
ax.set_xlabel("Out-of-Sample ROC-AUC")
ax.set_title("Bank Failure Prediction - 4 Model Classes", fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Average Precision bars
ax = axes[1]
bars = ax.barh(m_cmp["model"], m_cmp["avg_precision"], color="#F57C00", alpha=0.85, edgecolor="black")
for bar, v in zip(bars, m_cmp["avg_precision"]):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Average Precision (PR-AUC)")
ax.set_title("Average Precision (Class Imbalance Adjusted)", fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Stage 1: How Predictive Are Call Report Ratios of Bank Failure?",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{FIG}/stage1_model_comparison.png")
fig.savefig(f"{FIG}/stage1_model_comparison.pdf")
plt.close()
print("   saved")


# =====================================================================
# 2. ROC curve for the best model
# =====================================================================
print("\n[2] ROC curve...")
roc      = pd.read_csv(f"{TBL}/roc_curve.csv")
best_ix  = m_cmp["roc_auc"].idxmax()
best     = m_cmp.loc[best_ix, "model"]
auc_best = m_cmp.loc[best_ix, "roc_auc"]

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(roc["fpr"], roc["tpr"], color="#1976D2", linewidth=2,
        label=f"{best} (AUC={auc_best:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.1, color="#1976D2")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {best}\nPredicting Bank Failure (t+4)", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
fig.savefig(f"{FIG}/roc_curve.png")
fig.savefig(f"{FIG}/roc_curve.pdf")
plt.close()
print("   saved")


# =====================================================================
# 3. THE MONEY CHART - Dodd-Frank Value Test
# =====================================================================
print("\n[3] Dodd-Frank Value Test (the money chart)...")
df_v = pd.read_csv(f"{TBL}/dodd_frank_value_test.csv")

fig, ax = plt.subplots(figsize=(11, 6))

models_present = df_v["model"].unique()
bar_width      = 0.35
group_centers  = np.arange(len(models_present))

# One paired (pre-DF, modern) group per model
for i, model in enumerate(models_present):
    sub = df_v[df_v["model"] == model].sort_values("n_features")
    pre = sub[sub["feature_set"] == "Pre-Dodd-Frank"].iloc[0]
    mod = sub[sub["feature_set"] == "Modern (full)"].iloc[0]

    # Pre-DF bar
    ax.bar(i - bar_width/2, pre["auc"], bar_width,
           yerr=[[pre["auc"] - pre["ci_lo"]], [pre["ci_hi"] - pre["auc"]]],
           capsize=5, color=ERA_COLORS["pre_df"], alpha=0.85, edgecolor="black",
           label="Pre-DF (41 features)" if i == 0 else "")
    ax.text(i - bar_width/2, pre["auc"] + 0.005, f"{pre['auc']:.3f}",
            ha="center", fontsize=9, fontweight="bold")

    # Modern bar
    ax.bar(i + bar_width/2, mod["auc"], bar_width,
           yerr=[[mod["auc"] - mod["ci_lo"]], [mod["ci_hi"] - mod["auc"]]],
           capsize=5, color=ERA_COLORS["dodd_frank"], alpha=0.85, edgecolor="black",
           label="Modern (48 features)" if i == 0 else "")
    ax.text(i + bar_width/2, mod["auc"] + 0.005, f"{mod['auc']:.3f}",
            ha="center", fontsize=9, fontweight="bold")

    # Delta annotation above the pair
    delta = mod["auc"] - pre["auc"]
    if   delta >  0.005: delta_color = "green"
    elif delta < -0.005: delta_color = "red"
    else:                delta_color = "gray"
    ax.annotate(f"D = {delta:+.4f}",
                xy=(i, max(pre["auc"], mod["auc"]) + 0.025),
                ha="center", fontsize=10, fontweight="bold", color=delta_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=delta_color))

ax.set_xticks(group_centers)
ax.set_xticklabels(models_present, fontsize=11)
ax.set_ylabel("Out-of-Sample ROC-AUC (test: 2018-2025)", fontsize=11)
ax.set_title("The Dodd-Frank Value Test: Did the Post-2010 Reporting Expansion\n"
             "Improve Bank Failure Prediction From Public Data?",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(min(df_v["ci_lo"]) - 0.05, 1.0)
ax.grid(True, alpha=0.3, axis="y")

caption = ("Pre-Dodd-Frank = 41 ratios derivable from pre-2010 Call Report schedules.\n"
           "Modern = 48 ratios including 7 fields added by Dodd-Frank (deposit/insurance) "
           "and Basel III (risk-weighted capital).\n"
           "Both feature sets trained on 2008-2017, evaluated on 2018-2025. "
           "Error bars: 95% bootstrap CI on AUC.")
fig.text(0.5, -0.03, caption, ha="center", fontsize=9, style="italic", color="dimgray")

plt.tight_layout()
fig.savefig(f"{FIG}/dodd_frank_value_test.png")
fig.savefig(f"{FIG}/dodd_frank_value_test.pdf")
plt.close()
print("   saved (the money chart)")


# =====================================================================
# 4. Era-stratified performance line chart
# =====================================================================
print("\n[4] Era-stratified performance...")
era_df = pd.read_csv(f"{TBL}/era_stratified_models.csv")

# Wider figure with extra room at top so 5 eras + labels don't collide
fig, ax1 = plt.subplots(figsize=(13, 6.5))
xs = np.arange(len(era_df))

# AUC line on primary axis
ax1.plot(xs, era_df["roc_auc"], "o-", color="#1976D2", linewidth=2.5, markersize=10, label="ROC-AUC")
# Place AUC labels BELOW each point (avoids colliding with the title)
for i, row in era_df.iterrows():
    ax1.text(i, row["roc_auc"] - 0.008, f"{row['roc_auc']:.4f}",
             ha="center", va="top", fontsize=9, fontweight="bold", color="#1976D2")
ax1.set_ylabel("ROC-AUC", color="#1976D2", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#1976D2")
# Add headroom at the top so the title doesn't touch the data
ax1.set_ylim(0.92, 1.015)
ax1.set_xticks(xs)
ax1.set_xticklabels([e.split("(")[0].strip() for e in era_df["era"]],
                    rotation=15, ha="right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Failure counts on a secondary axis - use darker red text for contrast
ax2 = ax1.twinx()
bar_max = era_df["n_failures"].max()
ax2.bar(xs, era_df["n_failures"], width=0.4, alpha=0.25, color="#C62828", label="# Failures")
for i, row in era_df.iterrows():
    ax2.text(i, row["n_failures"] + bar_max * 0.02, f"{int(row['n_failures'])}",
             ha="center", fontsize=9, color="#7B0000", fontweight="bold")
ax2.set_ylabel("# Failed Banks (in era)", color="#C62828", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#C62828")
ax2.set_ylim(0, bar_max * 1.15)

# Single-line title with padding to keep clear of data labels
ax1.set_title("Predictive Performance by Regulatory Era (XGBoost, era-specific feature set)",
              fontweight="bold", pad=18)
plt.tight_layout()
fig.savefig(f"{FIG}/era_stratified_performance.png")
fig.savefig(f"{FIG}/era_stratified_performance.pdf")
plt.close()
print("   saved")


# =====================================================================
# 5. SHAP feature importance, color-coded by regulatory era
# =====================================================================
print("\n[5] SHAP importance (era-coded)...")

best_model = joblib.load(f"{PROC}/best_model.joblib")
scaler     = joblib.load(f"{PROC}/scaler.joblib")
feat_names = joblib.load(f"{PROC}/feature_names.joblib")

mask    = df[feat_names + ["failure_t4"]].notna().all(axis=1) & np.isfinite(df[feat_names]).all(axis=1)
X_clean = df.loc[mask, feat_names].values

# Random 5K-row sample for SHAP (otherwise it's slow)
n_sample = min(5000, len(X_clean))
idx          = np.random.RandomState(42).choice(len(X_clean), n_sample, replace=False)
X_sample     = X_clean[idx]
X_sample_df  = pd.DataFrame(X_sample, columns=feat_names)

# TreeExplainer for the tree models; fall back to LinearExplainer if not a tree
try:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    # Tree models can return [neg_shap, pos_shap] or a 3-D array — pick positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
except Exception as e:
    print(f"   TreeExplainer failed ({e}); falling back to LinearExplainer")
    explainer   = shap.LinearExplainer(best_model, X_sample[:200])
    shap_values = explainer.shap_values(X_sample)

# Standard SHAP beeswarm
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample_df, show=False, max_display=20)
plt.title(f"SHAP Feature Importance - {type(best_model).__name__}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/shap_beeswarm.png")
plt.savefig(f"{FIG}/shap_beeswarm.pdf")
plt.close()

# Custom era-colored bar chart of mean |SHAP|
mean_shap = np.abs(shap_values).mean(axis=0)
imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_shap})
imp["era"] = imp["feature"].map(FEATURE_ERAS).fillna("pre_df")
imp = imp.sort_values("mean_abs_shap", ascending=True).tail(20)
imp.to_csv(f"{TBL}/shap_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = [ERA_COLORS[e] for e in imp["era"]]
ax.barh(imp["feature"], imp["mean_abs_shap"],
        color=colors_bar, edgecolor="black", alpha=0.85)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Top 20 Predictors of Bank Failure - Color-Coded by Regulatory Era",
             fontweight="bold")

era_handles = [mpatches.Patch(color=ERA_COLORS[e], label=ERA_LABELS[e])
               for e in ["pre_df", "dodd_frank", "basel_iii"] if e in imp["era"].values]
ax.legend(handles=era_handles, loc="lower right", fontsize=10,
          title="Feature available since:")
ax.grid(True, alpha=0.3, axis="x")

n_pre  = (imp["era"] == "pre_df").sum()
n_post = (imp["era"] != "pre_df").sum()
caption = (f"Of the top 20 predictors, {n_pre} are derivable from pre-Dodd-Frank "
           f"Call Report data and only {n_post} require post-2010 fields.")
fig.text(0.5, -0.02, caption, ha="center", fontsize=9, style="italic", color="dimgray")

plt.tight_layout()
fig.savefig(f"{FIG}/shap_by_era.png")
fig.savefig(f"{FIG}/shap_by_era.pdf")
plt.close()
print("   saved")


# =====================================================================
# 6. MCA biplot
# =====================================================================
print("\n[6] MCA biplot...")
mca_results = joblib.load(f"{PROC}/mca_results.joblib")
coords  = mca_results["row_coords_full"]
inertia = mca_results["explained_inertia"]

n = min(len(coords), len(df))
fig, ax = plt.subplots(figsize=(10, 8))

# Healthy banks: scatter a sample (otherwise the plot is a giant blob)
normal     = (df.iloc[:n]["failure_t4"] == 0).values
healthy_ix = np.random.RandomState(42).choice(np.where(normal)[0],
                                              min(8000, normal.sum()), replace=False)
ax.scatter(coords.iloc[healthy_ix, 0], coords.iloc[healthy_ix, 1],
           c="#1976D2", alpha=0.08, s=2, label="Healthy banks", rasterized=True)

# Failed banks (red)
fail_m = (df.iloc[:n]["failure_t4"] == 1).values
if fail_m.any():
    ax.scatter(coords.iloc[np.where(fail_m)[0], 0],
               coords.iloc[np.where(fail_m)[0], 1],
               c="#C62828", alpha=0.5, s=15, label="Failed within 4Q",
               edgecolors="black", linewidth=0.3)

# Anomaly ensemble flags that AREN'T failures (purple x)
ens_m = (df.iloc[:n]["ensemble_flag"] == 1).values & ~fail_m
if ens_m.any():
    ens_ix = np.where(ens_m)[0][:1500]
    ax.scatter(coords.iloc[ens_ix, 0], coords.iloc[ens_ix, 1],
               c="#9C27B0", alpha=0.25, s=6, label="Anomaly ensemble flag", marker="x")

# Crop to the central 99% to avoid extreme outliers blowing up the axes
ql = coords.iloc[:n].quantile(0.005)
qh = coords.iloc[:n].quantile(0.995)
ax.set_xlim(ql.iloc[0] - 0.1, qh.iloc[0] + 0.1)
ax.set_ylim(ql.iloc[1] - 0.1, qh.iloc[1] + 0.1)
ax.set_xlabel(f"Dimension 1 ({inertia[0]:.1f}% inertia)")
ax.set_ylabel(f"Dimension 2 ({inertia[1]:.1f}% inertia)")
ax.set_title("MCA Biplot - FDIC-Insured Banks\n"
             "Failed Banks Cluster Away From the Healthy Mass", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(f"{FIG}/mca_biplot.png")
fig.savefig(f"{FIG}/mca_biplot.pdf")
plt.close()
print("   saved")


# =====================================================================
# 7. Case study trajectories (SVB, Signature, Republic First)
# =====================================================================
print("\n[7] Case study trajectories...")

key_features = [
    "tier1_rwa", "npl_total_loans", "roa", "nim",
    "loans_to_deposits", "brokered_dep_ratio", "cash_sec_to_assets",
    "cre_total_capital", "qoq_deposit_growth", "qoq_asset_growth",
]

qmap = {"0331": "Q1", "0630": "Q2", "0930": "Q3", "1231": "Q4"}

for tag, info in case_studies.items():
    bank = df[df["CERT"] == info["cert"]].sort_values("REPDTE").copy()
    if len(bank) == 0:
        print(f"   {info['name']}: no data")
        continue

    bank["dt"] = pd.to_datetime(bank["REPDTE"], format="%Y%m%d")
    # Last 8 quarters before the failure date
    pre = bank[bank["dt"] <= pd.Timestamp(info["fail"])].tail(8)
    if len(pre) < 2:
        continue

    # X-axis labels like "20Q1", "20Q2", ...
    labels = [f"{q[2:4]}{qmap.get(q[4:], '')}" for q in pre["REPDTE"]]

    n_feat = len(key_features)
    n_rows = (n_feat + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten()

    xs = range(len(labels))
    for i, feat in enumerate(key_features):
        if feat not in pre.columns:
            continue
        era = FEATURE_ERAS.get(feat, "pre_df")
        axes[i].plot(xs, pre[feat].values, "o-", color=ERA_COLORS[era], linewidth=2, markersize=4)
        axes[i].set_title(feat.replace("_", " ").title(), fontsize=9, fontweight="bold")
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[i].grid(alpha=0.3)
        axes[i].tick_params(axis="y", labelsize=7)

    # Hide any leftover subplot slots
    for j in range(len(key_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{info['name']} - Pre-Failure Trajectory "
                 f"(color: regulatory era of feature)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG}/case_{tag}_metrics.png")
    fig.savefig(f"{FIG}/case_{tag}_metrics.pdf")
    plt.close()

print("\nAll figures saved to figures/")
