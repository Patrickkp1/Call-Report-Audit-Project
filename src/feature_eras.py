# feature_eras.py
# Hand-classifies each of the 48 Call Report ratios into a regulatory era
# based on when the underlying FFIEC schedule fields became available.
#
# This is the central object for the Dodd-Frank value test: we ask whether
# the post-2010 expansion of regulatory reporting actually improved our
# ability to predict bank failures from public Call Report data.
#
# Era timeline:
#   "pre_df"     -> Available since at least 2001 (basic RC/RI/RC-C/RC-N)
#   "dodd_frank" -> Added/expanded 2010Q3 - 2014Q4 (Schedule RC-O Memo item 2,
#                   brokered-deposit breakdowns, Reg D rewrite, FDIC assessment changes)
#   "basel_iii"  -> Required starting 2015Q1 with the revised Schedule RC-R Part II
#                   (standardized RWA approach, CET1, advanced capital ratios)
#   "post_svb"   -> Reporting enhancements being phased in 2023Q2+ following the
#                   March 2023 failures (still partially implemented)
#
# Sources:
#   - FFIEC Call Report instruction history (1996-2025)
#   - FFIEC 031/041/051 schedule revisions, particularly the June 2014 NPR
#     (https://www.ffiec.gov) for Schedule RC-R Basel III implementation
#   - Reginfo.gov Dodd-Frank Schedule RC-O instructions (effective 2010-12-31)
#   - 2023-2024 Federal Register RFIs on deposit reporting enhancements


FEATURE_ERAS = {
    # ── Capital Ratios (6) ─────────────────────────────────────────
    # The basic Tier 1 ratio existed under Basel I from 1989, but the
    # specific MDRM fields (RBCT1J, RBCRWAJ, IDT1LEV) became mandatory
    # under their current definitions in the 2015 Basel III rewrite of
    # Schedule RC-R Part II.
    "tier1_rwa":           "basel_iii",
    "leverage_ratio":      "pre_df",
    "tangible_eq_assets":  "pre_df",
    "total_capital_rwa":   "basel_iii",
    "equity_to_assets":    "pre_df",
    "tier1_lev_gap":       "basel_iii",

    # ── Asset Quality (8) ──────────────────────────────────────────
    # Most from Schedule RC-N which has existed since at least the 1990s.
    # NOTE: P3ASSET (past-due 30-89) is the exception — it was added to
    # Schedule RC-N starting 2001Q1. For pre-2001 quarters it's NaN.
    "npl_total_loans":     "pre_df",
    "past_due_30_89":      "pre_df",   # available 2001Q1+, NaN before
    "past_due_90plus":     "pre_df",
    "nonaccrual_ratio":    "pre_df",
    "oreo_assets":         "pre_df",
    "coverage_ratio":      "pre_df",
    "nco_avg_loans":       "pre_df",
    "provision_avg_loans": "pre_df",

    # ── Earnings (8) ───────────────────────────────────────────────
    # Schedule RI - basic income statement stable since the 1990s
    "roa":                "pre_df",
    "roe":                "pre_df",
    "nim":                "pre_df",
    "efficiency_ratio":   "pre_df",
    "nonii_share":        "pre_df",
    "provision_to_nii":   "pre_df",
    "interest_exp_ratio": "pre_df",
    "op_income_assets":   "pre_df",

    # ── Liquidity (8) ──────────────────────────────────────────────
    # Most liquidity ratios derive from Schedule RC items that pre-date
    # Dodd-Frank. The DF additions are on Schedule RC-O:
    #   - Memo item 2 (estimated uninsured deposits) -> Dec 2010
    #   - Brokered deposit reciprocal/sweep breakdown -> June 2012
    #   - Core deposits redefinition under Section 165 -> 2010Q3
    "loans_to_deposits":      "pre_df",
    "cash_sec_to_assets":     "pre_df",
    "brokered_dep_ratio":     "dodd_frank",
    "core_dep_to_assets":     "dodd_frank",
    "fed_funds_purch_assets": "pre_df",
    "liquid_assets_ratio":    "pre_df",
    "insured_dep_ratio":      "dodd_frank",
    "volatile_liab_ratio":    "dodd_frank",

    # ── Concentration (6) ──────────────────────────────────────────
    # CRE concentration measures derive from Schedule RC-C, which has had
    # the residential / multifamily / nonfarm-nonresidential split since 2001.
    # The CRE concentration guidance (FIL-104-2006) pre-dates Dodd-Frank.
    "cre_total_capital":        "pre_df",
    "ci_total_loans":           "pre_df",
    "cre_total_loans":          "pre_df",
    "resi_re_total_loans":      "pre_df",
    "construction_total_loans": "pre_df",
    "consumer_total_loans":     "pre_df",

    # ── Off-Balance Sheet (4) ──────────────────────────────────────
    # Schedule RC-L. Unused commitments and basic derivative notionals are pre-DF.
    "unused_commit_assets":  "pre_df",
    "deriv_notional_assets": "pre_df",
    "cc_unused_commit":      "pre_df",
    "cre_unused_commit":     "pre_df",

    # ── Growth Signals (8) ─────────────────────────────────────────
    # Mechanically derived from balance-sheet items that pre-date Dodd-Frank
    "qoq_asset_growth":     "pre_df",
    "qoq_loan_growth":      "pre_df",
    "qoq_deposit_growth":   "pre_df",
    "qoq_equity_growth":    "pre_df",
    "yoy_asset_growth":     "pre_df",
    "yoy_loan_growth":      "pre_df",
    "qoq_npl_change":       "pre_df",
    "qoq_provision_change": "pre_df",
}


# Effective dates for each regulatory era
ERA_START_DATE = {
    "pre_df":     "20010101",   # all quarters in our sample
    "dodd_frank": "20101001",   # 2010Q4 - Memo item 2 took effect
    "basel_iii":  "20150101",   # 2015Q1 - revised RC-R Part II
    "post_svb":   "20230701",   # 2023Q3 - first quarter post-SVB enhancements
}

# Ranking used to filter "features available as of era X"
ERA_RANK = {"pre_df": 0, "dodd_frank": 1, "basel_iii": 2, "post_svb": 3}


# Convenience accessors
def features_in_era(era):
    """Return the list of features available as of (and including) `era`."""
    threshold = ERA_RANK[era]
    return [f for f, e in FEATURE_ERAS.items() if ERA_RANK[e] <= threshold]


def pre_df_features():
    """Features available before Dodd-Frank (the counterfactual feature set)."""
    return features_in_era("pre_df")


def all_features():
    """All 48 features (the full modern feature set)."""
    return list(FEATURE_ERAS.keys())


def era_summary():
    """Print summary of how features distribute across eras."""
    from collections import Counter
    c = Counter(FEATURE_ERAS.values())
    total = sum(c.values())
    print(f"{'Era':<15s} {'# Features':>12s} {'%':>8s}")
    print("-" * 38)
    for era in ["pre_df", "dodd_frank", "basel_iii", "post_svb"]:
        n = c.get(era, 0)
        print(f"{era:<15s} {n:>12d} {100*n/total:>7.1f}%")
    print("-" * 38)
    print(f"{'TOTAL':<15s} {total:>12d} {'100.0%':>8s}")


if __name__ == "__main__":
    era_summary()
    print(f"\nPre-Dodd-Frank only:  {len(pre_df_features())} features")
    print(f"Modern (all):         {len(all_features())} features")

    df_adds = [f for f, e in FEATURE_ERAS.items() if e == "dodd_frank"]
    print(f"\nDodd-Frank additions ({len(df_adds)}):")
    for f in df_adds:
        print(f"  {f}")

    basel_adds = [f for f, e in FEATURE_ERAS.items() if e == "basel_iii"]
    print(f"\nBasel III additions ({len(basel_adds)}):")
    for f in basel_adds:
        print(f"  {f}")
