# data_access.py
from __future__ import annotations
import sqlite3
import pandas as pd
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_conn(db_path: str):
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_fund_info(_conn) -> pd.DataFrame:
    q = "SELECT amc_name, fund_name, fund_code FROM fund_info ORDER BY amc_name, fund_name"
    try:
        df = pd.read_sql_query(q, _conn)
    except Exception:
        # minimal fallback
        df = pd.DataFrame(columns=["amc_name","fund_name","fund_code"])
    for c in ["amc_name","fund_name","fund_code"]:
        if c not in df.columns: df[c] = None
    return df

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_equities_info(_conn):
    q = """
    SELECT isin, company_name, industry_rating, market_cap
    FROM equities_info
    WHERE company_name IS NOT NULL
    ORDER BY company_name
    """
    return pd.read_sql_query(q, _conn)

@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_holdings_for_funds(_conn, fund_names: list[str]) -> pd.DataFrame:
    if not fund_names:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in fund_names)
    q = f"""
    SELECT fund_name, type, subtype, category, company_name,
           quantity, market_value, pct_net_assets, ytm, isin, industry_rating, amc_name
    FROM holdings_info
    WHERE fund_name IN ({placeholders})
    """
    df = pd.read_sql_query(q, _conn, params=fund_names)
    # normalize columns
    if "percentage" not in df.columns and "pct_net_assets" in df.columns:
        df["percentage"] = df["pct_net_assets"]
    if "pct_net_assets" not in df.columns and "percentage" in df.columns:
        df["pct_net_assets"] = df["percentage"]
    for c in ["pct_net_assets"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    for c in ["company_name","category","type","subtype","isin","amc_name","industry_rating"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("")
    return df


def _connect(db_path: Path | str | None = None):
    return sqlite3.connect(str(db_path))

def get_holdings_info(db_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load mutual-fund holdings with a normalized schema.
    Guarantees these columns (when possible):
      fund_code, isin, company_name, category, market_value, percentage

    - If the table has 'value' but not 'market_value', it aliases value -> market_value.
    - All numeric fields are coerced to numeric (NaN -> 0 for market_value).
    """
    with _connect(db_path) as con:
        # pull everything; schema may vary across sources
        df = pd.read_sql_query("SELECT * FROM holdings_info;", con)

    # --- normalize column casing ---
    lower_map = {c.lower(): c for c in df.columns}
    def has(col): return col in lower_map
    def get(col): return lower_map.get(col, None)

    # expected names (case-insensitive)
    fund_code_col   = get("fund_code")
    isin_col        = get("isin")
    name_col        = get("company_name") or get("name") or get("security_name")
    category_col    = get("category")
    mv_col          = get("market_value") or get("value")          # alias value -> market_value
    pct_col         = get("percentage") or get("pct_net_assets")   # optional

    # rename to canonical
    rename_map = {}
    if fund_code_col and fund_code_col != "fund_code":   rename_map[fund_code_col] = "fund_code"
    if isin_col and isin_col != "isin":                   rename_map[isin_col] = "isin"
    if name_col and name_col != "company_name":           rename_map[name_col] = "company_name"
    if category_col and category_col != "category":       rename_map[category_col] = "category"
    if mv_col and mv_col != "market_value":               rename_map[mv_col] = "market_value"
    if pct_col and pct_col != "percentage":               rename_map[pct_col] = "percentage"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure required columns exist
    for req in ["fund_code", "isin", "company_name", "category"]:
        if req not in df.columns:
            df[req] = ""

    # guarantee a market_value column
    if "market_value" not in df.columns:
        # last resort: create it (zeros) so upstream logic can still run
        df["market_value"] = 0.0

    # percentage is optional; create if missing
    if "percentage" not in df.columns:
        df["percentage"] = None

    # dtypes & cleaning
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
    if "percentage" in df.columns:
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")

    for col in ["fund_code", "isin", "company_name", "category"]:
        df[col] = df[col].astype(str).fillna("")

    return df


@st.cache_data(ttl=6*60*60)
def load_holdings_for_funds(_conn):
    q = f"""
    SELECT fund_name, type, subtype, category, company_name, quantity, market_value,
           pct_net_assets, ytm, isin, industry_rating, amc_name
    FROM holdings_info
    """
    return pd.read_sql_query(q, _conn)