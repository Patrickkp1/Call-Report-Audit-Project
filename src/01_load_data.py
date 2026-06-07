# 01_load_data.py
# Pull Call Report data + failed bank list from the FDIC.
# Saves one parquet file per quarter to data/raw/.

import os
import io
import time
import requests
import pandas as pd

# Where to save stuff
RAW = "data/raw"
OUT = "data/outcomes"
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# FDIC API
URL      = "https://api.fdic.gov/banks/financials"
FAIL_URL = "https://www.fdic.gov/bank-failures/download-data.csv"

# Quarters to pull: 1995Q3 through 2025Q4
# (61 quarters pre-Dodd-Frank + 61 quarters post-DF = balanced 122-quarter panel)
q_ends = ["0331", "0630", "0930", "1231"]
quarters = []
for y in range(1995, 2026):
    for q in q_ends:
        # Skip the 2 quarters before 1995Q3
        if y == 1995 and q in ("0331", "0630"):
            continue
        quarters.append(f"{y}{q}")

# Call Report fields I want (MDRM codes from FFIEC)
# Schedule RC = balance sheet, RC-R = capital, RC-N = past due,
# RI = income, RC-C = loans by type, RC-L = off-balance sheet
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


def get_quarter(repdte):
    # Pull all banks for one quarter. API caps at 10K rows per call, so we
    # page through with offsets 0, 10000, 20000 (~5k-14k banks per quarter
    # historically; consolidation has shrunk the count over time).
    rows = []
    for offset in range(0, 30000, 10000):
        params = {
            "filters": f"REPDTE:{repdte}",
            "fields":  ",".join(fields),
            "limit":   10000,
            "offset":  offset,
            "output":  "json",
        }
        r = requests.get(URL, params=params, timeout=120)
        page = r.json().get("data", [])
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


# Download each quarter (skip if already saved)
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


# Download the failed bank list (latin-1 encoding because of accented characters)
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
