# 01_load_data.py
# Pulls Call Report data from the FDIC public API (one parquet per quarter)
# plus the failed-bank list. Outputs go to data/raw/ and data/outcomes/.

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

# Build a list of quarter-end dates (YYYYMMDD) from 2008Q1 through 2025Q4
quarter_ends = ["0331", "0630", "0930", "1231"]
quarters = []
for y in range(2008, 2026):
    for q in quarter_ends:
        quarters.append(f"{y}{q}")

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


# Pull every bank for one quarter. The API returns at most 10,000 rows per
# call, so we page through with a fixed-size for loop. There are ~5k-9k
# banks per quarter, so 3 pages is more than enough.
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
        # FDIC wraps each row in {"data": {...}} — unwrap it
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


# Loop through quarters and save one parquet per quarter (skip if cached)
for i, q in enumerate(quarters, 1):
    path = f"{RAW}/call_report_{q}.parquet"
    if os.path.exists(path):
        print(f"[{i}/{len(quarters)}] {q}: cached")
        continue

    print(f"[{i}/{len(quarters)}] {q}: downloading...")
    df = get_quarter(q)
    if not df.empty:
        df.to_parquet(path, index=False)
        print(f"   {len(df):,} rows")


# Failed bank list — small CSV, latin-1 encoding because of accented names
print("\nDownloading failed bank list...")
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
