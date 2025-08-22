# api.py
from __future__ import annotations
from datetime import datetime, timedelta
import requests
import pandas as pd
import streamlit as st

API_BASE = "https://api.mfapi.in/mf/"

def _retry_get(url: str, timeout: int = 8, tries: int = 2):
    """Small retry wrapper for GET requests."""
    err = None
    for _ in range(tries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            err = e
    raise err if err else RuntimeError("request failed")

def _closest_past(d: pd.DataFrame, target: datetime) -> float:
    """NAV from last observation with date <= target; else earliest NAV."""
    s = d[d["date"] <= target]
    return float(s["nav"].iloc[-1]) if not s.empty else float(d["nav"].iloc[0])

def _absolute_return(start_nav: float, end_nav: float) -> float:
    """
    Calculates the simple absolute return. Used for periods < 1 year.
    """
    if not pd.notna(start_nav) or not pd.notna(end_nav) or start_nav <= 0:
        return float("nan")
    try:
        return (end_nav / start_nav) - 1.0
    except Exception:
        return float("nan")

def _cagr(start_nav: float, end_nav: float, years: float) -> float:
    """
    Compound annual growth rate (decimal). Used for periods >= 1 year.
    """
    if not pd.notna(start_nav) or not pd.notna(end_nav) or start_nav <= 0 or years <= 0:
        return float("nan")
    try:
        return (end_nav / start_nav) ** (1.0 / years) - 1.0
    except Exception:
        return float("nan")

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_fund_performance(fund_code: int | str | None):
    """
    Fetch NAV time series from mfapi.in and compute returns.
    - Absolute returns for < 1Y.
    - CAGR for >= 1Y.
    Returns:
      {
        "as_of": "YYYY-MM-DD",
        "nav": float,
        "returns": {
          "1M","3M","6M","1Y","3Y","5Y","10Y","SI": float (decimal)
        }
      }
    or None when data is unavailable or fund_code is missing.
    """
    if fund_code in (None, "", 0):
        return None

    # --- fetch ---
    resp = _retry_get(f"{API_BASE}{fund_code}")
    data = resp.json() if resp is not None else {}
    hist = pd.DataFrame(data.get("data", []))

    # --- sanitize ---
    if hist.empty or "nav" not in hist or "date" not in hist:
        return None
    hist = hist.copy()
    hist["nav"] = pd.to_numeric(hist["nav"], errors="coerce")
    # mfapi dates are "dd-mm-YYYY"
    hist["date"] = pd.to_datetime(hist["date"], format="%d-%m-%Y", errors="coerce")
    hist = hist.dropna(subset=["nav", "date"]).sort_values("date").reset_index(drop=True)
    if hist.empty:
        return None

    # --- endpoints ---
    end_nav = float(hist["nav"].iloc[-1])
    end_dt = pd.Timestamp(hist["date"].iloc[-1]).to_pydatetime()

    # --- horizons (days) ---
    horizons = {
        "1M": 30,
        "3M": 90,
        "6M": 182,          # ~6 months
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
        "10Y": 365 * 10,
    }

    # --- compute returns for each horizon ---
    rets: dict[str, float] = {}
    for label, days in horizons.items():
        start_dt = end_dt - timedelta(days=days)
        start_nav = _closest_past(hist, start_dt)

        # *** EDITED LOGIC HERE ***
        # Use absolute return for periods shorter than a year
        if days < 365:
            rets[label] = _absolute_return(start_nav, end_nav)
        # Use CAGR for periods of one year or more
        else:
            years = days / 365.0
            rets[label] = _cagr(start_nav, end_nav, years)

    # --- since inception (always CAGR) ---
    start_nav_si = float(hist["nav"].iloc[0])
    start_dt_si = pd.Timestamp(hist["date"].iloc[0]).to_pydatetime()
    years_si = max((end_dt - start_dt_si).days / 365.0, 1e-6)
    rets["SI"] = _cagr(start_nav_si, end_nav, years_si)

    return {"as_of": end_dt.date().isoformat(), "nav": end_nav, "returns": rets}
