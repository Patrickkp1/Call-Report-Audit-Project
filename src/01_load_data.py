# 01_load_data.py
# Pulls Call Report data from the FDIC public API (one parquet per quarter)
# plus the failed-bank list. Outputs go to data/raw/ and data/outcomes/.
#
# Two-pass design:
#   1. load_modern_era():  2008Q1 - 2025Q4 (the original sample window)
#   2. load_historical():  1995Q3 - 2007Q4 (backfill for a balanced pre-DF baseline)
#
# Why two functions? The historical backfill is experimental — some Call Report
# fields (e.g. ESTINS, past-due 30-89) don't exist pre-2001, and several
# ratio definitions changed at the Basel I -> Basel III transition. Keeping
# the historical pull separate makes it easy to skip if it causes issues
# downstream.

import os
import io
import time
import requests
import pandas as pd

# Folders
RAW = "data/raw"
OUT = "data/outcomes"
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# Endpoints
URL      = "https://api.fdic.gov/banks/financials"
FAIL_URL = "https://www.fdic.gov/bank-failures/download-data.csv"


# FDIC MDRM fields I want from the Call Report (IDs + main schedules)
fields = [
    # IDs
    "CERT", "REPDTE", "NAMEFULL", "STALP", "BKCLASS", "ASSET",
    # RC - Balance sheet
    "LNLSGR", "LNATRES", "SC", "DEP", "BRO", "CHBAL", "FREPO", "FREPP",
    "TRADE", "ORE", "INTAN", "EQ", "COREDEP", "ESTINS",
    # RC-R - Capital
    "RBCT1J", "RBCT2", "RBCRWAJ", "IDT1LEV",
    # RC-N - Past due / nonaccrual
    "P3ASSET", "P9ASSET", "NCLNLS",
    # RI - Income statement
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    # RC-C - Loans by type
    "LNRECONS", "LNRERES", "LNREMULT", "LNRENRES", "LNCI", "LNCON", "LNCRCD",
    # RC-L - Off-balance sheet
    "UC", "UCCOMRE", "UCCRCD", "RTNVS", "FXNVS", "EDCM", "OTHNVS",
]


def build_quarter_list(start_year, start_q, end_year, end_q):
    """Build a list of YYYYMMDD quarter-end strings between two (year, quarter) points.

    start_q / end_q are quarter numbers 1-4 (inclusive on both ends).
    """
    q_to_end = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    quarters = []
    for y in range(start_year, end_year + 1):
        # Which quarters to include this year
        q_lo = start_q if y == start_year else 1
        q_hi = end_q   if y == end_year   else 4
        for q in range(q_lo, q_hi + 1):
            quarters.append(f"{y}{q_to_end[q]}")
    return quarters


# Pull every bank for one quarter. The API returns at most 10,000 rows per
# call, so we page through with a fixed-size for loop. There are ~5k-14k
# banks per quarter historically (consolidation has shrunk the count over time),
# so 3 pages is more than enough.
def get_quarter(repdte):
    rows = []
    for offset in range(0, 30000, 10000):   # offsets 0, 10000, 20000
        params = {
            "filters": f"REPDTE:{repdte}",
            "fields":  ",".join(fields),
            "limit":   10000,
            "offset":  offset,
            "output":  "json",
        }
        r = requests.get(URL, params=params, timeout=120)
        page = r.json().get("data", [])
        # FDIC wraps each row in {"data": {...}} - unwrap it
        for row in page:
            rows.append(row.get("data", row))
        # Last page if we got back fewer than 10K rows
        if len(page) < 10000:
            break
        time.sleep(0.4)   # be polite to the API

    df = pd.DataFrame(rows)
    if not df.empty:
        df["REPDTE"] = repdte
    return df


# Generic "pull a list of quarters and save each one as parquet" runner.
# Skips quarters that are already cached on disk.
def pull_quarters(quarters, label):
    print(f"\n=== {label}: {len(quarters)} quarters ===")
    for i, q in enumerate(quarters, 1):
        path = f"{RAW}/call_report_{q}.parquet"
        if os.path.exists(path):
            print(f"[{i}/{len(quarters)}] {q}: cached")
            continue

        print(f"[{i}/{len(quarters)}] {q}: downloading...")
        df = get_quarter(q)
        if df.empty:
            print(f"   (no data returned)")
            continue
        df.to_parquet(path, index=False)
        print(f"   {len(df):,} rows")


# Pass 1 - modern era (the original window)
def load_modern_era():
    quarters = build_quarter_list(2008, 1, 2025, 4)
    pull_quarters(quarters, "Modern era (2008Q1 - 2025Q4)")


# Pass 2 - historical backfill (matched in length to the post-DF window)
# Post-DF window is 2010Q4 - 2025Q4 = 61 quarters.
# So pre-DF window should be 61 quarters ending 2010Q3 -> starts 1995Q3.
def load_historical():
    quarters = build_quarter_list(1995, 3, 2007, 4)
    pull_quarters(quarters, "Historical backfill (1995Q3 - 2007Q4)")


# Failed-bank list - small CSV, latin-1 encoding because of accented names.
# This is the source of truth for our distress labels.
def load_failed_banks():
    print("\n=== Failed-bank list ===")
    r = requests.get(FAIL_URL, timeout=60)
    fails = pd.read_csv(io.StringIO(r.content.decode("latin-1")))
    fails.columns = [c.strip() for c in fails.columns]
    fails = fails.rename(columns={
        "Bank Name":    "bank_name",
        "Cert":         "cert",
        "Closing Date": "closing_date",
    })
    fails["closing_date"] = pd.to_datetime(fails["closing_date"], format="mixed")
    fails["cert"] = pd.to_numeric(fails["cert"], errors="coerce").astype("Int64")
    fails.to_csv(f"{OUT}/failed_banks.csv", index=False)
    print(f"   {len(fails):,} failed banks saved")


# Run everything top-to-bottom
load_modern_era()
load_historical()
load_failed_banks()

print("\nDone. Raw parquets are in data/raw/, failed-bank list in data/outcomes/.")
