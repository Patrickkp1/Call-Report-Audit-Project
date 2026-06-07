# 02_build_ratios_stats.py
# Load every raw quarterly file, build 48 financial ratios, winsorize them
# at the 1st/99th percentile, compute peer-group z-scores, and tag each
# bank-quarter with failure / pre-failure indicators.
# Output: a single analysis panel in data/processed/panel.parquet

import os
import glob
import numpy as np
import pandas as pd

RAW  = "data/raw"
PROC = "data/processed"
OUT  = "data/outcomes"
os.makedirs(PROC, exist_ok=True)


# 1. Load every quarterly parquet into one big panel
files = sorted(glob.glob(f"{RAW}/call_report_*.parquet"))
print(f"Loading {len(files)} quarterly files...")
panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Cast dtypes — IDs / text stay as strings, everything else is numeric
panel["CERT"]   = pd.to_numeric(panel["CERT"], errors="coerce").astype("Int64")
panel["REPDTE"] = panel["REPDTE"].astype(str)

text_cols = {"CERT", "REPDTE", "NAMEFULL", "STALP", "BKCLASS", "SPECGRP", "ACTIVE", "ESTYMD"}
for c in panel.columns:
    if c not in text_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

panel = panel.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
print(f"Panel: {len(panel):,} bank-quarters, {panel['CERT'].nunique():,} unique banks")


# 2. Safe divide — returns NaN where denominator is zero or missing
def div(num, den):
    return np.where((den != 0) & den.notna() & num.notna(), num / den, np.nan)


# 3. Annualize income items
# Call Report income is year-to-date, so Q1 needs ×4, Q2 ×2, Q3 ×4/3, Q4 ×1
qmult = panel["REPDTE"].str[-4:].map({"0331": 4, "0630": 2, "0930": 4/3, "1231": 1})
income_items = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NCLNLS"]
for c in income_items:
    panel[c + "_A"] = panel[c] * qmult

# Lagged values for QoQ and YoY growth, and for averaging
lag_items = ["ASSET", "LNLSGR", "DEP", "EQ"]
for c in lag_items:
    panel[c + "_L1"] = panel.groupby("CERT")[c].shift(1)
    panel[c + "_L4"] = panel.groupby("CERT")[c].shift(4)

# Average of current + prior quarter — used as the denominator for flow ratios
panel["AVG_ASSET"] = (panel["ASSET"]  + panel["ASSET_L1"])  / 2
panel["AVG_LOAN"]  = (panel["LNLSGR"] + panel["LNLSGR_L1"]) / 2
panel["AVG_EQ"]    = (panel["EQ"]     + panel["EQ_L1"])     / 2

# Combined / derived items
panel["NPL"]         = panel[["P9ASSET"]].fillna(0).sum(axis=1)
panel["CRE"]         = panel[["LNRENRES", "LNREMULT", "LNRECONS"]].fillna(0).sum(axis=1)
panel["DERIV"]       = panel[["RTNVS", "FXNVS", "EDCM", "OTHNVS"]].fillna(0).sum(axis=1)
panel["EARN_ASSETS"] = panel[["LNLSGR", "SC", "TRADE", "FREPO"]].fillna(0).sum(axis=1)
panel["AVG_EARN"]    = (panel["EARN_ASSETS"]
                        + panel.groupby("CERT")["EARN_ASSETS"].shift(1)) / 2


# 4. Build the 48 ratio features (grouped by CAMELS-ish bucket)
f = pd.DataFrame({"CERT": panel["CERT"], "REPDTE": panel["REPDTE"]})
f["log_assets"] = np.log(panel["ASSET"].clip(lower=1))
f["bkclass"]    = panel["BKCLASS"]

# Capital (6) — RBCRWAJ is reported as a percent; divide by 100 if > 1
f["tier1_rwa"]          = np.where(panel["RBCRWAJ"] > 1, panel["RBCRWAJ"] / 100, panel["RBCRWAJ"])
f["leverage_ratio"]     = np.clip(div(panel["RBCT1J"], panel["ASSET"]), 0, 1)
f["tangible_eq_assets"] = div(panel["EQ"] - panel["INTAN"].fillna(0), panel["ASSET"])
f["total_capital_rwa"]  = (f["tier1_rwa"] * 1.1).clip(0, 1.5)
f["equity_to_assets"]   = div(panel["EQ"], panel["ASSET"])
f["tier1_lev_gap"]      = f["tier1_rwa"] - f["leverage_ratio"]

# Asset quality (8)
f["npl_total_loans"]     = div(panel["NPL"], panel["LNLSGR"])
f["past_due_30_89"]      = div(panel["P3ASSET"], panel["LNLSGR"])
f["past_due_90plus"]     = div(panel["P9ASSET"], panel["LNLSGR"])
f["nonaccrual_ratio"]    = div(panel["P9ASSET"], panel["LNLSGR"])   # P9 as proxy
f["oreo_assets"]         = div(panel["ORE"], panel["ASSET"])
f["coverage_ratio"]      = div(panel["LNATRES"], panel["NPL"])
f["nco_avg_loans"]       = div(panel["NCLNLS_A"], panel["AVG_LOAN"])
f["provision_avg_loans"] = div(panel["ELNATR_A"], panel["AVG_LOAN"])

# Earnings (8)
nii   = panel["INTINC"]   - panel["EINTEXP"]
nii_a = panel["INTINC_A"] - panel["EINTEXP_A"]
f["roa"]                = div(panel["NETINC_A"], panel["AVG_ASSET"])
f["roe"]                = div(panel["NETINC_A"], panel["AVG_EQ"])
f["nim"]                = div(nii_a, panel["AVG_EARN"])
f["efficiency_ratio"]   = div(panel["NONIX"], (nii + panel["NONII"].fillna(0)).clip(lower=0.001))
f["nonii_share"]        = div(panel["NONII"], (nii + panel["NONII"].fillna(0)).clip(lower=0.001))
f["provision_to_nii"]   = div(panel["ELNATR_A"], nii_a)
f["interest_exp_ratio"] = div(panel["EINTEXP_A"], panel["AVG_ASSET"])
f["op_income_assets"]   = div(panel["NETINC_A"], panel["AVG_ASSET"])

# Liquidity (8)
f["loans_to_deposits"]      = div(panel["LNLSGR"], panel["DEP"])
f["cash_sec_to_assets"]     = div(panel["CHBAL"].fillna(0) + panel["SC"].fillna(0), panel["ASSET"])
f["brokered_dep_ratio"]     = div(panel["BRO"], panel["DEP"])
f["core_dep_to_assets"]     = div(panel["COREDEP"], panel["ASSET"])
f["fed_funds_purch_assets"] = div(panel["FREPP"], panel["ASSET"])
f["liquid_assets_ratio"]    = div(panel["CHBAL"].fillna(0) + panel["SC"].fillna(0) + panel["FREPO"].fillna(0), panel["ASSET"])
f["insured_dep_ratio"]      = div(panel["ESTINS"], panel["DEP"])
f["volatile_liab_ratio"]    = div(panel["DEP"] - panel["COREDEP"].fillna(0) + panel["FREPP"].fillna(0), panel["ASSET"])

# Concentration (6)
f["cre_total_capital"]        = div(panel["CRE"], panel["RBCT1J"] + panel["RBCT2"].fillna(0))
f["ci_total_loans"]           = div(panel["LNCI"], panel["LNLSGR"])
f["cre_total_loans"]          = div(panel["CRE"], panel["LNLSGR"])
f["resi_re_total_loans"]      = div(panel["LNRERES"], panel["LNLSGR"])
f["construction_total_loans"] = div(panel["LNRECONS"], panel["LNLSGR"])
f["consumer_total_loans"]     = div(panel["LNCON"], panel["LNLSGR"])

# Off-balance sheet (4)
f["unused_commit_assets"]  = div(panel["UC"], panel["ASSET"])
f["deriv_notional_assets"] = div(panel["DERIV"], panel["ASSET"])
f["cc_unused_commit"]      = div(panel["UCCRCD"], panel["ASSET"])
f["cre_unused_commit"]     = div(panel["UCCOMRE"], panel["ASSET"])

# Growth (8) — both quarter-over-quarter and year-over-year
f["qoq_asset_growth"]     = div(panel["ASSET"]  - panel["ASSET_L1"],  panel["ASSET_L1"].abs().clip(lower=1))
f["qoq_loan_growth"]      = div(panel["LNLSGR"] - panel["LNLSGR_L1"], panel["LNLSGR_L1"].abs().clip(lower=1))
f["qoq_deposit_growth"]   = div(panel["DEP"]    - panel["DEP_L1"],    panel["DEP_L1"].abs().clip(lower=1))
f["qoq_equity_growth"]    = div(panel["EQ"]     - panel["EQ_L1"],     panel["EQ_L1"].abs().clip(lower=1))
f["yoy_asset_growth"]     = div(panel["ASSET"]  - panel["ASSET_L4"],  panel["ASSET_L4"].abs().clip(lower=1))
f["yoy_loan_growth"]      = div(panel["LNLSGR"] - panel["LNLSGR_L4"], panel["LNLSGR_L4"].abs().clip(lower=1))
f["qoq_npl_change"]       = f["npl_total_loans"]     - f.groupby("CERT")["npl_total_loans"].shift(1)
f["qoq_provision_change"] = f["provision_avg_loans"] - f.groupby("CERT")["provision_avg_loans"].shift(1)

# Count of real ratios: total cols minus (CERT, REPDTE, log_assets, bkclass)
print(f"Built {f.shape[1] - 4} ratio features")


# 5. Winsorize each ratio at 1st/99th percentile, separately per quarter
# (handles outliers without warping the cross-section)
ratio_cols = []
for c in f.columns:
    if c in ("CERT", "REPDTE", "log_assets", "bkclass"):
        continue
    if f[c].dtype in ("float64", "float32"):
        ratio_cols.append(c)

for q in f["REPDTE"].unique():
    mask = f["REPDTE"] == q
    for c in ratio_cols:
        vals = f.loc[mask, c].dropna()
        if len(vals) >= 10:
            lo, hi = np.nanpercentile(vals, [1, 99])
            f.loc[mask, c] = f.loc[mask, c].clip(lo, hi)

print(f"Winsorized {len(ratio_cols)} features")


# 6. Peer-group z-scores: group by charter type x asset-size decile, per quarter
charter_map = {
    "N":  "National",
    "NM": "State_NM",
    "SM": "State_SM",
    "SB": "Savings",
    "SA": "Savings",
}
f["charter_group"] = f["bkclass"].map(charter_map).fillna("Other")
f["asset_decile"]  = f.groupby("REPDTE")["log_assets"].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
)

# For each ratio, subtract peer mean and divide by peer std
for c in ratio_cols:
    g = f.groupby(["REPDTE", "charter_group", "asset_decile"])[c]
    f["z_" + c] = (f[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

print(f"Computed {len(ratio_cols)} peer-group z-scores")


# 7. Distress labels: did the bank fail in the next 4 quarters?
fails = pd.read_csv(f"{OUT}/failed_banks.csv")
fails["closing_date"] = pd.to_datetime(fails["closing_date"])
fails["CERT"] = fails["cert"].astype("Int64")
fail_map = fails[["CERT", "closing_date"]].dropna()

f["repdte_dt"] = pd.to_datetime(f["REPDTE"], format="%Y%m%d")
f = f.merge(fail_map, on="CERT", how="left")

# Months between this quarter and the bank's failure date (NaN if never failed)
months = (f["closing_date"] - f["repdte_dt"]).dt.days / 30

f["failure_t4"]     = ((months > 0)  & (months <= 13)).astype(int)   # fails in next 4Q
f["enforcement_t4"] = ((months >= 3) & (months <= 24)).astype(int)   # 3-24 months out
f["distress_t4"]    = ((f["failure_t4"] == 1) | (f["enforcement_t4"] == 1)).astype(int)
f["amendment_flag"] = (f["qoq_asset_growth"].abs() > 0.5).astype(int)

f = f.drop(columns=["closing_date", "repdte_dt"])

print("\nDistress labels:")
print(f"  failure_t4:      {f['failure_t4'].sum():,} ({f['failure_t4'].mean()*100:.2f}%)")
print(f"  enforcement_t4:  {f['enforcement_t4'].sum():,} ({f['enforcement_t4'].mean()*100:.2f}%)")
print(f"  distress_t4:     {f['distress_t4'].sum():,} ({f['distress_t4'].mean()*100:.2f}%)")


# 8. Save the panel + summary stats
os.makedirs("tables", exist_ok=True)
f.to_parquet(f"{PROC}/panel.parquet", index=False)
f[ratio_cols].describe().T.to_csv("tables/feature_summary_stats.csv")
print(f"\nSaved {f.shape[0]:,} x {f.shape[1]} panel to {PROC}/panel.parquet")
