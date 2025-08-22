# core/transforms.py
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd

# AMCs with the special cash rule (per your earlier logic)
SPECIAL_CASH_AMCS = {
    "Axis Mutual Funds",
    "Motilal Oswal Asset Management Company",
}

# ---------- helpers ----------

def _normalize_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if "deriv" in c or "fut" in c or "opt" in c:
        return "DERIVATIVE"
    if "equity" in c or "stock" in c:
        return "EQUITY"
    if "debt" in c or "bond" in c or "g-sec" in c:
        return "DEBT"
    if "cash" in c or "treps" in c or "overnight" in c:
        return "CASH"
    return "UNKNOWN"

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _norm_mcap(x: str) -> str:
    s = (str(x) if x is not None else "").strip().lower()
    if not s:
        return "Unknown"
    if "large" in s:
        return "Large Cap"
    if "mid" in s:
        return "Mid Cap"
    if "small" in s:
        return "Small Cap"
    if "micro" in s:
        return "Small Cap"
    return "Unknown"

def _fold_small_slices(df: pd.DataFrame, label_col: str, value_col: str, threshold: float) -> pd.DataFrame:
    """Fold rows with share < threshold into 'Other'. Assumes df has a 'pct' column."""
    if df.empty:
        return df
    small = df["pct"] < (threshold * 100.0)
    if not small.any():
        return df.sort_values(value_col, ascending=False).reset_index(drop=True)
    big = df.loc[~small, [label_col, value_col, "pct"]]
    other_val = df.loc[small, value_col].sum()
    other_pct = df.loc[small, "pct"].sum()
    if other_val > 0:
        big = pd.concat(
            [big, pd.DataFrame([{label_col: "Other", value_col: other_val, "pct": other_pct}])],
            ignore_index=True,
        )
    # Re-normalize pct to 100 after folding
    pct_sum = big["pct"].sum()
    if pct_sum and pct_sum != 100:
        big["pct"] = big["pct"] * (100.0 / pct_sum)
    return big.sort_values(value_col, ascending=False).reset_index(drop=True)

# ---------- 1) MF Asset Allocation (item-level, scaled by adjusted AUM; keyed by fund_name) ----------

def mf_asset_allocation_by_fund_name_old(
    mf_portfolio: List[Dict],          # [{amc_name, fund_name, amount}]
    holdings_df: pd.DataFrame,         # fund_name, category, market_value, isin
) -> Tuple[pd.DataFrame, Dict]:
    """
    Item-level derivative netting by ISIN, AMC-specific cash rule, scale by adjusted AUM per fund.
    Returns:
      df_alloc: columns ['asset_class','value','pct']
      meta: {'display_as_bar': bool, 'total_mf': float}
    """
    if holdings_df is None or holdings_df.empty:
        df0 = pd.DataFrame(
            [{"asset_class": k, "value": 0.0, "pct": 0.0} for k in ["Equity", "Debt", "Cash"]]
        )
        return df0, {"display_as_bar": False, "total_mf": 0.0}

    df = holdings_df.copy()
    for col in ["fund_name", "category", "isin"]:
        if col not in df.columns:
            df[col] = ""
    if "market_value" not in df.columns:
        if "value" in df.columns:
            df = df.rename(columns={"value": "market_value"})
        else:
            df["market_value"] = 0.0

    df["fund_name_norm"] = df["fund_name"].astype(str).map(_norm)
    df["bucket"] = df["category"].map(_normalize_category)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    # selected funds keyed by fund_name
    selected: List[Dict] = []
    for r in mf_portfolio or []:
        amt = float(r.get("amount", 0) or 0)
        if amt <= 0:
            continue
        fn = _norm(str(r.get("fund_name", "")))
        if fn:
            selected.append({"fund_name_norm": fn, "amc": r.get("amc_name", ""), "amount": amt})

    total_mf = sum(x["amount"] for x in selected)
    if total_mf == 0:
        df0 = pd.DataFrame(
            [{"asset_class": k, "value": 0.0, "pct": 0.0} for k in ["Equity", "Debt", "Cash"]]
        )
        return df0, {"display_as_bar": False, "total_mf": 0.0}

    equity_total = debt_total = cash_total = 0.0
    any_fund_net_short = False

    for s in selected:
        sub = df.loc[df["fund_name_norm"] == s["fund_name_norm"]]
        amt = s["amount"]
        amc = (s["amc"] or "").strip()

        if sub.empty:
            cash_total += amt
            continue

        # raw bucket sums
        raw_debt_sum  = float(sub.loc[sub["bucket"] == "DEBT",       "market_value"].sum())
        raw_cash_sum  = float(sub.loc[sub["bucket"] == "CASH",       "market_value"].sum())
        raw_deriv_sum = float(sub.loc[sub["bucket"] == "DERIVATIVE", "market_value"].sum())

        # item-level maps and net equity per ISIN
        eq_by_isin  = sub.loc[sub["bucket"] == "EQUITY"].groupby("isin")["market_value"].sum().to_dict()
        der_by_isin = sub.loc[sub["bucket"] == "DERIVATIVE"].groupby("isin")["market_value"].sum().to_dict()

        matched_deriv_sum = 0.0
        net_equity_sum_raw = 0.0
        for isin in set(eq_by_isin) | set(der_by_isin):
            ev = eq_by_isin.get(isin, 0.0)
            dv = der_by_isin.get(isin, 0.0)
            if dv != 0:
                matched_deriv_sum += dv
            net_equity_sum_raw += (ev + dv)

        # cash rule (AMC-specific)
        unmatched_deriv = raw_deriv_sum - matched_deriv_sum
        if amc in SPECIAL_CASH_AMCS:
            net_cash_raw = raw_cash_sum - unmatched_deriv
        else:
            net_cash_raw = raw_cash_sum - raw_deriv_sum

        # adjusted AUM and scale
        adjusted_aum_raw = net_equity_sum_raw + raw_debt_sum + net_cash_raw
        if adjusted_aum_raw == 0:
            cash_total += amt
            continue

        scale = amt / adjusted_aum_raw
        equity_total += net_equity_sum_raw * scale
        debt_total   += raw_debt_sum       * scale
        cash_total   += net_cash_raw       * scale

        if net_equity_sum_raw < 0:
            any_fund_net_short = True

    df_alloc = pd.DataFrame(
        [
            {"asset_class": "Equity", "value": equity_total},
            {"asset_class": "Debt",   "value": debt_total},
            {"asset_class": "Cash",   "value": cash_total},
        ]
    )
    # df_alloc["pct"] = (df_alloc["value"] / total_mf * 100.0).where(total_mf != 0, 0.0)
    if total_mf != 0:
        df_alloc["pct"] = df_alloc["value"] * (100.0 / total_mf)
    else:
        df_alloc["pct"] = 0.0


    meta = {
        "display_as_bar": any_fund_net_short or (df_alloc["value"] < 0).any(),
        "total_mf": total_mf,
    }
    return df_alloc, meta

# ---------- 2) Build user-scaled net equity by ISIN (used by sector/mcap/top) ----------

def _equity_net_by_isin_scaled_for_user_funds(
    mf_portfolio: List[Dict],           # [{amc_name, fund_name, amount}]
    holdings_df: pd.DataFrame,          # fund_name, category, market_value, isin
) -> pd.Series:
    """
    Returns a Series indexed by ISIN with the user-scaled net equity rupees,
    computed at item level and scaled by each fund's adjusted AUM.
    """
    if holdings_df is None or holdings_df.empty:
        return pd.Series(dtype=float)

    df = holdings_df.copy()
    for col in ["fund_name", "category", "isin"]:
        if col not in df.columns:
            df[col] = ""
    if "market_value" not in df.columns:
        df["market_value"] = 0.0

    df["fund_name_norm"] = df["fund_name"].astype(str).map(_norm)
    df["bucket"] = df["category"].map(_normalize_category)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    # selected funds
    selected: List[Dict] = []
    for r in mf_portfolio or []:
        amt = float(r.get("amount", 0) or 0)
        if amt <= 0:
            continue
        fn = _norm(str(r.get("fund_name", "")))
        if fn:
            selected.append({"fund_name_norm": fn, "amc": r.get("amc_name", ""), "amount": amt})

    if not selected:
        return pd.Series(dtype=float)

    agg_by_isin: Dict[str, float] = {}

    for s in selected:
        sub = df.loc[df["fund_name_norm"] == s["fund_name_norm"]]
        amt = s["amount"]
        amc = (s["amc"] or "").strip()

        if sub.empty:
            continue

        raw_deriv_sum = float(sub.loc[sub["bucket"] == "DERIVATIVE", "market_value"].sum())
        raw_debt_sum  = float(sub.loc[sub["bucket"] == "DEBT",       "market_value"].sum())
        raw_cash_sum  = float(sub.loc[sub["bucket"] == "CASH",       "market_value"].sum())

        eq_by_isin  = sub.loc[sub["bucket"] == "EQUITY"].groupby("isin")["market_value"].sum().to_dict()
        der_by_isin = sub.loc[sub["bucket"] == "DERIVATIVE"].groupby("isin")["market_value"].sum().to_dict()

        matched_deriv_sum = 0.0
        net_equity_sum_raw = 0.0
        per_isin_net: Dict[str, float] = {}
        for isin in set(eq_by_isin) | set(der_by_isin):
            ev = eq_by_isin.get(isin, 0.0)
            dv = der_by_isin.get(isin, 0.0)
            if dv != 0:
                matched_deriv_sum += dv
            net_val = ev + dv  # item-level net
            per_isin_net[isin] = net_val
            net_equity_sum_raw += net_val

        unmatched_deriv = raw_deriv_sum - matched_deriv_sum
        if amc in SPECIAL_CASH_AMCS:
            net_cash_raw = raw_cash_sum - unmatched_deriv
        else:
            net_cash_raw = raw_cash_sum - raw_deriv_sum

        adjusted_aum_raw = net_equity_sum_raw + raw_debt_sum + net_cash_raw
        if adjusted_aum_raw == 0:
            continue

        scale = amt / adjusted_aum_raw
        for isin, net_val in per_isin_net.items():
            agg_by_isin[isin] = agg_by_isin.get(isin, 0.0) + net_val * scale

    if not agg_by_isin:
        return pd.Series(dtype=float)

    s = pd.Series(agg_by_isin, dtype=float)
    s = s[abs(s) > 1e-9]
    return s

# ---------- 3) Sector & Market-Cap exposure (net equity only; pies use positives) ----------

def mf_sector_and_mcap_exposure_by_fund_name_old(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,          # fund_name, category, market_value, isin
    equities_info: pd.DataFrame,        # isin, sector, market_cap
    small_slice_threshold: float = 0.03 # 3%
) -> Dict[str, pd.DataFrame | Dict]:
    net_by_isin = _equity_net_by_isin_scaled_for_user_funds(mf_portfolio, holdings_df)
    if net_by_isin.empty:
        empty = pd.DataFrame(columns=["label", "value", "pct"])
        return {"sector": empty, "market_cap": empty,
                "diagnostics": {"total_equity_pos": 0.0, "negatives_total": 0.0, "unknown_sector": 0, "unknown_mcap": 0}}

    eq = (equities_info or pd.DataFrame()).copy()
    if eq.empty or "isin" not in eq.columns:
        eq = pd.DataFrame({"isin": net_by_isin.index, "sector": "Unknown", "market_cap": "Unknown"})

    for col in ["isin", "sector", "market_cap"]:
        if col not in eq.columns:
            eq[col] = "Unknown" if col != "isin" else ""
    eq["isin"] = eq["isin"].astype(str)
    eq["sector"] = eq["sector"].fillna("Unknown").astype(str)
    eq["market_cap"] = eq["market_cap"].fillna("Unknown").astype(str).map(_norm_mcap)

    map_df = pd.DataFrame({"isin": net_by_isin.index, "value": net_by_isin.values})
    merged = map_df.merge(eq[["isin", "sector", "market_cap"]], on="isin", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["market_cap"] = merged["market_cap"].fillna("Unknown").map(_norm_mcap)

    pos = merged.loc[merged["value"] > 0].copy()
    neg_total = float(merged.loc[merged["value"] < 0, "value"].sum())
    total_pos = float(pos["value"].sum())

    def _build(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty or total_pos <= 0:
            return pd.DataFrame(columns=["label", "value", "pct"])
        g = df.groupby(col, as_index=False)["value"].sum()
        g = g.sort_values("value", ascending=False, ignore_index=True)
        g["pct"] = (g["value"] / total_pos) * 100.0
        g = g.rename(columns={col: "label"})
        g = _fold_small_slices(g, "label", "value", small_slice_threshold)
        return g

    sector_df = _build(pos, "sector")
    mcap_df   = _build(pos, "market_cap")

    diags = {
        "total_equity_pos": total_pos,
        "negatives_total": neg_total,
        "unknown_sector": int((merged["sector"] == "Unknown").sum()),
        "unknown_mcap":   int((merged["market_cap"] == "Unknown").sum()),
    }
    return {"sector": sector_df, "market_cap": mcap_df, "diagnostics": diags}

def mf_sector_and_mcap_exposure_by_fund_name_older(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,          # fund_name, category, market_value, isin
    equities_info: pd.DataFrame,        # isin, sector, market_cap
    small_slice_threshold: float = 0.03 # 3%
) -> Dict[str, pd.DataFrame | Dict]:
    # 1) Build user-scaled net equity by ISIN
    net_by_isin = _equity_net_by_isin_scaled_for_user_funds(mf_portfolio, holdings_df)
    if net_by_isin.empty:
        empty = pd.DataFrame(columns=["label", "value", "pct"])
        return {
            "sector": empty,
            "market_cap": empty,
            "diagnostics": {"total_equity_pos": 0.0, "negatives_total": 0.0, "unknown_sector": 0, "unknown_mcap": 0},
        }

    # ✅ FIX: never use a DataFrame in a boolean expression
    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()

    if eq.empty or "isin" not in eq.columns:
        # fallback mapping if master missing
        eq = pd.DataFrame({"isin": net_by_isin.index, "sector": "Unknown", "market_cap": "Unknown"})

    # normalize mapping
    for col in ["isin", "sector", "market_cap"]:
        if col not in eq.columns:
            eq[col] = "Unknown" if col != "isin" else ""
    eq["isin"] = eq["isin"].astype(str)
    eq["sector"] = eq["sector"].fillna("Unknown").astype(str)
    eq["market_cap"] = eq["market_cap"].fillna("Unknown").astype(str).map(_norm_mcap)

    map_df = pd.DataFrame({"isin": net_by_isin.index, "value": net_by_isin.values})
    merged = map_df.merge(eq[["isin", "sector", "market_cap"]], on="isin", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged["market_cap"] = merged["market_cap"].fillna("Unknown").map(_norm_mcap)

    # 2) Split positives for charting; keep negatives for diagnostics
    pos = merged.loc[merged["value"] > 0].copy()
    neg_total = float(merged.loc[merged["value"] < 0, "value"].sum())
    total_pos = float(pos["value"].sum())

    def _build(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty or total_pos <= 0:
            return pd.DataFrame(columns=["label", "value", "pct"])
        g = df.groupby(col, as_index=False)["value"].sum().sort_values("value", ascending=False, ignore_index=True)
        g["pct"] = (g["value"] / total_pos) * 100.0
        g = g.rename(columns={col: "label"})
        g = _fold_small_slices(g, "label", "value", small_slice_threshold)
        # re-normalize to exactly 100 after folding
        pct_sum = float(g["pct"].sum())
        if pct_sum != 0:
            g["pct"] = g["pct"] * (100.0 / pct_sum)
        return g

    sector_df = _build(pos, "sector")
    mcap_df   = _build(pos, "market_cap")

    diags = {
        "total_equity_pos": total_pos,
        "negatives_total": neg_total,
        "unknown_sector": int((merged["sector"] == "Unknown").sum()),
        "unknown_mcap":   int((merged["market_cap"] == "Unknown").sum()),
    }

    return {"sector": sector_df, "market_cap": mcap_df, "diagnostics": diags}


# ---------- 4) Top-N underlying net stock holdings ----------

def _first_non_empty(s: pd.Series) -> str:
    for v in s.astype(str):
        if v and v.strip():
            return v
    return ""

def _company_name_map(equities_info: pd.DataFrame, holdings_df: pd.DataFrame) -> Dict[str, str]:
    """Prefer equities_info.company_name; fall back to first non-empty name from holdings."""
    mapping: Dict[str, str] = {}
    if equities_info is not None and not equities_info.empty and "isin" in equities_info.columns:
        eq = equities_info.copy()
        if "company_name" not in eq.columns:
            eq["company_name"] = ""
        eq = eq[["isin", "company_name"]].dropna(subset=["isin"])
        for _, r in eq.iterrows():
            isin = str(r["isin"])
            nm = str(r.get("company_name", "") or "")
            if isin and nm and nm.strip():
                mapping[isin] = nm.strip()

    if holdings_df is not None and not holdings_df.empty:
        hd = holdings_df[["isin", "company_name"]].copy()
        hd["isin"] = hd["isin"].astype(str)
        hd["company_name"] = hd["company_name"].astype(str)
        fallback = (
            hd.groupby("isin", as_index=True)["company_name"]
              .apply(_first_non_empty)
              .to_dict()
        )
        for isin, nm in fallback.items():
            if isin and isin not in mapping and nm and nm.strip():
                mapping[isin] = nm.strip()
    return mapping

def _fund_presence_counts_by_isin_for_user_funds(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,
) -> Dict[str, int]:
    """Count in how many selected funds each ISIN appears with non-zero net (equity + derivative) value."""
    if holdings_df is None or holdings_df.empty:
        return {}

    df = holdings_df.copy()
    for col in ["fund_name", "category", "isin"]:
        if col not in df.columns:
            df[col] = ""
    if "market_value" not in df.columns:
        df["market_value"] = 0.0

    df["fund_name_norm"] = df["fund_name"].astype(str).map(_norm)
    df["bucket"] = df["category"].map(_normalize_category)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    selected = [_norm(str(r.get("fund_name", ""))) for r in (mf_portfolio or []) if float(r.get("amount", 0) or 0) > 0]
    selected = [fn for fn in selected if fn]
    if not selected:
        return {}

    counts: Dict[str, int] = {}
    for fn in selected:
        sub = df.loc[df["fund_name_norm"] == fn]
        if sub.empty:
            continue
        eq = sub.loc[sub["bucket"] == "EQUITY"].groupby("isin")["market_value"].sum()
        dr = sub.loc[sub["bucket"] == "DERIVATIVE"].groupby("isin")["market_value"].sum()
        net = (eq.add(dr, fill_value=0.0)).fillna(0.0)
        for isin, val in net.items():
            if abs(float(val)) > 0.0:
                counts[isin] = counts.get(isin, 0) + 1
    return counts

def mf_top_net_holdings_by_fund_name_old(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,          # fund_name, category, market_value, isin, company_name
    equities_info: pd.DataFrame,        # isin, company_name (optional but preferred)
    top_n: int = 10,
) -> Dict[str, pd.DataFrame | Dict]:
    """
    Top-N underlying net stock holdings:
    - item-level net (equity + derivative) per ISIN,
    - scale each fund by adjusted AUM (user amount / adjusted_aum_raw),
    - sum across funds, rank by positive exposure.
    Returns:
      {'top_holdings': DataFrame[isin, company_name, value, pct_of_equity, fund_count],
       'diagnostics': {'total_equity_pos': float, 'negatives_total': float}}
    """
    net_by_isin = _equity_net_by_isin_scaled_for_user_funds(mf_portfolio, holdings_df)
    if net_by_isin.empty:
        return {
            "top_holdings": pd.DataFrame(columns=["isin","company_name","value","pct_of_equity","fund_count"]),
            "diagnostics": {"total_equity_pos": 0.0, "negatives_total": 0.0},
        }

    pos = net_by_isin[net_by_isin > 0].sort_values(ascending=False)
    total_pos = float(pos.sum())
    neg_total = float(net_by_isin[net_by_isin < 0].sum())

    if total_pos <= 0:
        return {
            "top_holdings": pd.DataFrame(columns=["isin","company_name","value","pct_of_equity","fund_count"]),
            "diagnostics": {"total_equity_pos": 0.0, "negatives_total": neg_total},
        }

    presence = _fund_presence_counts_by_isin_for_user_funds(mf_portfolio, holdings_df)
    name_map = _company_name_map(equities_info, holdings_df)

    top = pos.head(top_n)
    rows = []
    for isin, val in top.items():
        nm = name_map.get(isin, "") or isin
        rows.append({
            "isin": isin,
            "company_name": nm,
            "value": float(val),
            "pct_of_equity": (float(val) / total_pos) * 100.0,
            "fund_count": int(presence.get(isin, 0)),
        })
    df_top = pd.DataFrame(rows).sort_values("value", ascending=False, ignore_index=True)

    diags = {"total_equity_pos": total_pos, "negatives_total": neg_total}
    return {"top_holdings": df_top, "diagnostics": diags}




def mf_fund_track_record_by_fund_name(
    mf_portfolio: List[Dict],      # [{amc_name, fund_name, amount}]
    fund_info: pd.DataFrame,       # needs fund_name; optional: amc, category, nav, return_1y/3y/5y etc.
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a table of the selected funds with user amounts and weights matched by fund_name.

    Output df columns (some optional if missing in fund_info):
      ['amc','fund_name','category','amount','weight','nav','ret_1y','ret_3y','ret_5y']

    meta includes:
      {'total_amount': float, 'w_1y': float|None, 'w_3y': float|None, 'w_5y': float|None}
    """
    # collect selected funds
    rows = []
    for r in mf_portfolio or []:
        amt = float(r.get("amount", 0) or 0)
        fn  = (r.get("fund_name","") or "").strip()
        if amt > 0 and fn:
            rows.append({"fund_name": fn, "amount": amt, "fund_name_norm": fn.lower()})
    if not rows:
        empty = pd.DataFrame(columns=["amc","fund_name","category","amount","weight","nav","ret_1y","ret_3y","ret_5y"])
        return empty, {"total_amount": 0.0, "w_1y": None, "w_3y": None, "w_5y": None}

    sel = pd.DataFrame(rows)

    # normalize fund_info
    fi = (fund_info.copy() if fund_info is not None else pd.DataFrame())
    if fi.empty:
        fi = pd.DataFrame({"fund_name": sel["fund_name"]})
    if "fund_name" not in fi.columns:
        fi["fund_name"] = ""
    if "amc" not in fi.columns:
        fi["amc"] = ""
    if "category" not in fi.columns:
        fi["category"] = ""
    if "nav" not in fi.columns:
        fi["nav"] = pd.NA

    # try to discover common return column names
    cols_l = {c.lower(): c for c in fi.columns}
    def pick(*cands):
        for c in cands:
            if c in cols_l:
                return cols_l[c]
        return None

    r1_col = pick("return_1y","1y","cagr_1y")
    r3_col = pick("return_3y","3y","cagr_3y")
    r5_col = pick("return_5y","5y","cagr_5y")

    fi["fund_name_norm"] = fi["fund_name"].astype(str).str.strip().str.lower()

    merged = sel.merge(
        fi[["fund_name_norm","fund_name","amc","category","nav"] +
           [c for c in [r1_col, r3_col, r5_col] if c]],
        on="fund_name_norm", how="left", suffixes=("","_fi")
    )
    merged["fund_name"] = merged["fund_name_fi"].fillna(merged["fund_name"])
    merged.drop(columns=[c for c in merged.columns if c.endswith("_fi")], inplace=True)

    total = float(merged["amount"].sum())
    merged["weight"] = 0.0 if total == 0 else merged["amount"] / total

    # rename return columns to canonical names if present
    out = merged.copy()
    if r1_col: out = out.rename(columns={r1_col: "ret_1y"})
    else:      out["ret_1y"] = pd.NA
    if r3_col: out = out.rename(columns={r3_col: "ret_3y"})
    else:      out["ret_3y"] = pd.NA
    if r5_col: out = out.rename(columns={r5_col: "ret_5y"})
    else:      out["ret_5y"] = pd.NA

    # coerce numeric for math safety
    for c in ["ret_1y","ret_3y","ret_5y","nav"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # weighted averages (if returns exist)
    def wavg(col):
        if col not in out.columns or out[col].isna().all():
            return None
        return float((out[col].fillna(0) * out["weight"]).sum())

    meta = {
        "total_amount": total,
        "w_1y": wavg("ret_1y"),
        "w_3y": wavg("ret_3y"),
        "w_5y": wavg("ret_5y"),
    }

    cols = ["amc","fund_name","category","amount","weight","nav","ret_1y","ret_3y","ret_5y"]
    # keep only columns that exist
    cols = [c for c in cols if c in out.columns]
    return out[cols], meta


# ---------- Direct Stocks, Combined, Overlap utilities ----------

import pandas as pd
from typing import List, Dict, Tuple

def stocks_equity_by_isin(
    stocks_portfolio: List[Dict],      # [{company_name, amount, isin?}]
    equities_info: pd.DataFrame        # needs at least: isin, company_name
) -> pd.Series:
    """
    Aggregate user direct stocks into a Series: ISIN -> rupees (positives only).
    Resolves ISIN by (a) provided isin, else (b) exact company_name match (case-insensitive).
    """
    if not stocks_portfolio:
        return pd.Series(dtype=float)

    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
    if "isin" not in eq.columns: eq["isin"] = ""
    if "company_name" not in eq.columns: eq["company_name"] = ""
    eq["company_name_norm"] = eq["company_name"].astype(str).str.strip().str.lower()
    name_to_isin = eq.set_index("company_name_norm")["isin"].to_dict()

    bucket: Dict[str, float] = {}
    for r in stocks_portfolio:
        amt = float(r.get("amount", 0) or 0)
        if amt <= 0: 
            continue
        isin = (r.get("isin") or "").strip()
        if not isin:
            nm = (r.get("company_name") or "").strip().lower()
            isin = name_to_isin.get(nm, "")
        if not isin:
            # unmapped; skip from charts (will still show in details if you want)
            continue
        bucket[isin] = bucket.get(isin, 0.0) + amt

    if not bucket:
        return pd.Series(dtype=float)
    s = pd.Series(bucket, dtype=float)
    s = s[s > 0]  # positives only for pies/bars
    return s

def _map_isin_to_attrs(series: pd.Series, equities_info: pd.DataFrame) -> pd.DataFrame:
    """Helper: returns DataFrame with columns [isin, value, company_name, sector, market_cap]."""
    if series is None or series.empty:
        return pd.DataFrame(columns=["isin","value","company_name","sector","market_cap"])
    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
    for col in ["isin","company_name","sector","market_cap"]:
        if col not in eq.columns: 
            eq[col] = "Unknown" if col != "isin" else ""
    eq["market_cap"] = eq["market_cap"].astype(str).map(_norm_mcap)
    m = pd.DataFrame({"isin": series.index.astype(str), "value": series.values})
    out = m.merge(eq[["isin","company_name","sector","market_cap"]], on="isin", how="left")
    out["company_name"] = out["company_name"].fillna("Unknown")
    out["sector"] = out["sector"].fillna("Unknown")
    out["market_cap"] = out["market_cap"].fillna("Unknown").map(_norm_mcap)
    return out

def sector_mcap_from_series_old(
    series: pd.Series, equities_info: pd.DataFrame, small_slice_threshold: float = 0.03
) -> Dict[str, pd.DataFrame]:
    """
    Build sector & market-cap pies from a positive-only ISIN series (₹).
    Returns {'sector': df, 'market_cap': df}; each df has columns [label, value, pct].
    """
    if series is None or series.empty:
        empty = pd.DataFrame(columns=["label","value","pct"])
        return {"sector": empty, "market_cap": empty}

    merged = _map_isin_to_attrs(series[series > 0], equities_info)
    total = float(merged["value"].sum())
    if total <= 0:
        empty = pd.DataFrame(columns=["label","value","pct"])
        return {"sector": empty, "market_cap": empty}

    def _build(col: str) -> pd.DataFrame:
        g = merged.groupby(col, as_index=False)["value"].sum().sort_values("value", ascending=False)
        g["pct"] = g["value"] / total * 100.0
        g = g.rename(columns={col: "label"})
        g = _fold_small_slices(g, "label", "value", small_slice_threshold)
        return g
    return {"sector": _build("sector"), "market_cap": _build("market_cap")}

def sector_mcap_from_series(
    s_direct: pd.Series,
    equities_info: pd.DataFrame,
    small_slice_threshold: float = 0.03,
) -> dict:
    """
    Build pies for DIRECT STOCKS using *industry_rating* as the sector dimension.
    Input:
      - s_direct: Series(index=isin, values=rupees)  [positives only considered for pies]
      - equities_info: must map ISIN -> industry_rating, market_cap
    Returns:
      {"sector": DataFrame[label,value,pct], "market_cap": DataFrame[label,value,pct]}
    """
    import pandas as pd
    import numpy as np

    # Empty guards
    if s_direct is None or not isinstance(s_direct, pd.Series) or s_direct.empty:
        empty = pd.DataFrame(columns=["label", "value", "pct"])
        return {"sector": empty, "market_cap": empty}

    pos = s_direct[s_direct > 0]
    if pos.empty:
        empty = pd.DataFrame(columns=["label", "value", "pct"])
        return {"sector": empty, "market_cap": empty}

    # Map ISIN -> attributes (use industry_rating for sector)
    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
    for c in ["isin", "industry_rating", "market_cap"]:
        if c not in eq.columns:
            eq[c] = "" if c == "isin" else "Unknown"
    eq["isin"] = eq["isin"].astype(str)
    eq["industry_rating"] = eq["industry_rating"].astype(str).replace({"": "Unknown"})
    eq["market_cap"] = eq["market_cap"].astype(str).replace({"": "Unknown"})

    m = pd.DataFrame({"isin": pos.index.astype(str), "value": pos.values}).merge(
        eq[["isin", "industry_rating", "market_cap"]], on="isin", how="left"
    )
    m["industry_rating"] = m["industry_rating"].fillna("Unknown")
    m["market_cap"] = m["market_cap"].fillna("Unknown")

    total = float(m["value"].sum())

    def _build(col: str) -> pd.DataFrame:
        g = (
            m.groupby(col, as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False, ignore_index=True)
        )
        g["pct"] = (g["value"] / total * 100.0) if total != 0 else 0.0
        g = g.rename(columns={col: "label"})
        # fold <3% into Other (reuse helper if present)
        try:
            folded = _fold_small_slices(g, "label", "value", small_slice_threshold)  # type: ignore
        except Exception:
            # fallback in case helper not available
            small = g["pct"] < (small_slice_threshold * 100.0)
            if small.any():
                other_val = g.loc[small, "value"].sum()
                other_pct = g.loc[small, "pct"].sum()
                folded = pd.concat(
                    [g.loc[~small, ["label", "value", "pct"]],
                     pd.DataFrame([{"label": "Other", "value": other_val, "pct": other_pct}])],
                    ignore_index=True,
                ).sort_values("value", ascending=False, ignore_index=True)
            else:
                folded = g
        # re-normalize pct to 100
        pct_sum = float(folded["pct"].sum())
        if pct_sum not in (0.0, 100.0):
            folded["pct"] = folded["pct"] * (100.0 / pct_sum)
        return folded

    return {
        "sector": _build("industry_rating"),  # <- KEY FIX
        "market_cap": _build("market_cap"),
    }



def top_holdings_from_series(
    series: pd.Series, equities_info: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Top-N holdings from a positive-only ISIN series. Returns columns:
    ['isin','company_name','value','pct_of_equity']
    """
    if series is None or series.empty:
        return pd.DataFrame(columns=["isin","company_name","value","pct_of_equity"])
    s = series[series > 0].sort_values(ascending=False).head(top_n)
    total = float(series[series > 0].sum())
    m = _map_isin_to_attrs(s, equities_info)
    m["pct_of_equity"] = (m["value"] / max(total, 1e-9)) * 100.0
    return m[["isin","company_name","value","pct_of_equity"]].sort_values("value", ascending=False, ignore_index=True)

def mf_net_equity_series_by_fund_name(
    mf_portfolio: List[Dict], holdings_df: pd.DataFrame
) -> pd.Series:
    """Public wrapper around the existing private item-level function; returns ISIN->₹ (can include negatives)."""
    return _equity_net_by_isin_scaled_for_user_funds(mf_portfolio, holdings_df)

def combined_equity_by_isin(mf_series: pd.Series, direct_series: pd.Series) -> pd.Series:
    """Combined ISIN series: MF net equity + Direct equity (fill_value=0, drop near-zero)."""
    mf_series = mf_series if mf_series is not None else pd.Series(dtype=float)
    direct_series = direct_series if direct_series is not None else pd.Series(dtype=float)
    s = mf_series.add(direct_series, fill_value=0.0)
    s = s[abs(s) > 1e-9]
    return s

def combined_composition_split(mf_series: pd.Series, direct_series: pd.Series) -> pd.DataFrame:
    """Two-row split for donut: MF Net Equity vs Direct Equity (positives only)."""
    mf_pos = float((mf_series[mf_series > 0]).sum()) if mf_series is not None else 0.0
    dx_pos = float((direct_series[direct_series > 0]).sum()) if direct_series is not None else 0.0
    total = mf_pos + dx_pos
    rows = [{"label":"MF Net Equity","value": mf_pos},{"label":"Direct Equity","value": dx_pos}]
    df = pd.DataFrame(rows)
    if total != 0:
        df["pct"] = df["value"] / total * 100.0
    else:
        df["pct"] = 0.0
    return df

def fund_net_equity_series_map(
    mf_portfolio: List[Dict], holdings_df: pd.DataFrame
) -> Dict[str, pd.Series]:
    """
    For each selected fund (by fund_name), returns a positive-only ISIN->₹ series
    using item-level netting + adjusted-AUM scaling (your existing logic).
    """
    out: Dict[str, pd.Series] = {}
    if not mf_portfolio:
        return out
    # normalize holdings once
    df = holdings_df.copy() if holdings_df is not None else pd.DataFrame()
    if df.empty:
        return out
    df["fund_name_norm"] = df["fund_name"].astype(str).map(_norm)
    df["bucket"] = df["category"].map(_normalize_category)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    # per fund compute the same as in mf_asset_allocation_by_fund_name, but output series
    for r in mf_portfolio:
        amt = float(r.get("amount", 0) or 0)
        fn = _norm(str(r.get("fund_name","")))
        amc = (r.get("amc_name","") or "").strip()
        if amt <= 0 or not fn:
            continue
        sub = df.loc[df["fund_name_norm"] == fn]
        if sub.empty:
            continue

        raw_deriv_sum = float(sub.loc[sub["bucket"] == "DERIVATIVE", "market_value"].sum())
        raw_debt_sum  = float(sub.loc[sub["bucket"] == "DEBT",       "market_value"].sum())
        raw_cash_sum  = float(sub.loc[sub["bucket"] == "CASH",       "market_value"].sum())

        eq_by_isin  = sub.loc[sub["bucket"] == "EQUITY"].groupby("isin")["market_value"].sum().to_dict()
        der_by_isin = sub.loc[sub["bucket"] == "DERIVATIVE"].groupby("isin")["market_value"].sum().to_dict()

        matched_deriv_sum = 0.0
        per_isin_net: Dict[str, float] = {}
        net_equity_sum_raw = 0.0
        for isin in set(eq_by_isin) | set(der_by_isin):
            ev = eq_by_isin.get(isin, 0.0)
            dv = der_by_isin.get(isin, 0.0)
            if dv != 0:
                matched_deriv_sum += dv
            net_val = ev + dv
            per_isin_net[isin] = net_val
            net_equity_sum_raw += net_val

        unmatched_deriv = raw_deriv_sum - matched_deriv_sum
        if amc in SPECIAL_CASH_AMCS:
            net_cash_raw = raw_cash_sum - unmatched_deriv
        else:
            net_cash_raw = raw_cash_sum - raw_deriv_sum
        adjusted_aum_raw = net_equity_sum_raw + raw_debt_sum + net_cash_raw
        if adjusted_aum_raw == 0:
            continue

        scale = amt / adjusted_aum_raw
        scaled = {k: v * scale for k, v in per_isin_net.items()}
        s = pd.Series(scaled, dtype=float)
        s = s[s > 0]  # positives only for overlap
        if not s.empty:
            # use the original fund_name for display
            out[r.get("fund_name","")] = s

    return out

def fund_overlap_matrix(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Overlap % matrix across funds (positives only), defined as:
      overlap(i,j) = sum_k min(si_k, sj_k) / min(sum(si), sum(sj))
    where sums are over positive exposures only.
    """
    names = list(series_map.keys())
    n = len(names)
    mat = [[None]*n for _ in range(n)]
    sums = []
    for nm in names:
        s = series_map[nm]
        sums.append(float(s[s > 0].sum()) if s is not None else 0.0)

    for i in range(n):
        si = series_map[names[i]]
        si_pos = si[si > 0] if si is not None else pd.Series(dtype=float)
        for j in range(n):
            if i == j:
                mat[i][j] = 1.0
                continue
            sj = series_map[names[j]]
            sj_pos = sj[sj > 0] if sj is not None else pd.Series(dtype=float)
            if si_pos.empty or sj_pos.empty:
                mat[i][j] = 0.0
                continue
            shared = set(si_pos.index) & set(sj_pos.index)
            if not shared:
                mat[i][j] = 0.0
                continue
            overlap_val = sum(min(float(si_pos.get(k,0.0)), float(sj_pos.get(k,0.0))) for k in shared)
            den = min(float(si_pos.sum()), float(sj_pos.sum()))
            mat[i][j] = 0.0 if den == 0 else overlap_val / den

    return pd.DataFrame(mat, index=names, columns=names)

def fund_vs_direct_overlap(series_map: Dict[str, pd.Series], direct_series: pd.Series) -> pd.DataFrame:
    """
    For each fund, compute overlap with user's direct equity (positives only).
    Returns columns: ['fund','fund_equity','overlap_value','overlap_pct_of_fund'].
    """
    rows = []
    dx = direct_series[direct_series > 0] if direct_series is not None else pd.Series(dtype=float)
    for fund, s in series_map.items():
        sp = s[s > 0] if s is not None else pd.Series(dtype=float)
        fund_sum = float(sp.sum())
        if fund_sum <= 0:
            rows.append({"fund": fund, "fund_equity": 0.0, "overlap_value": 0.0, "overlap_pct_of_fund": 0.0})
            continue
        shared = set(sp.index) & set(dx.index)
        overlap_val = sum(min(float(sp.get(k,0.0)), float(dx.get(k,0.0))) for k in shared)
        rows.append({
            "fund": fund,
            "fund_equity": fund_sum,
            "overlap_value": overlap_val,
            "overlap_pct_of_fund": (overlap_val / fund_sum) * 100.0
        })
    out = pd.DataFrame(rows)
    return out.sort_values("overlap_pct_of_fund", ascending=False, ignore_index=True)


# ---- add near top if not present
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# ---- small helpers ----
def _norm(s: str) -> str:
    return (str(s) if s is not None else "").strip().lower()

def _is_axis_or_motilal(amc_name: str) -> bool:
    s = _norm(amc_name)
    return ("axis" in s) or ("motilal oswal" in s)

def _fold_small_slices(df: pd.DataFrame, label_col: str, value_col: str, threshold: float = 0.03) -> pd.DataFrame:
    """Group rows with pct < threshold into 'Other'. df must already have pct column."""
    if df.empty:
        return df
    df = df.copy()
    small = df["pct"] < (threshold * 100.0)
    if small.any():
        other_val = df.loc[small, value_col].sum()
        other_pct = df.loc[small, "pct"].sum()
        df = pd.concat([df.loc[~small, [label_col, value_col, "pct"]],
                        pd.DataFrame([{label_col: "Other", value_col: other_val, "pct": other_pct}])],
                       ignore_index=True)
        # sort by value desc
        df = df.sort_values(value_col, ascending=False, ignore_index=True)
    return df

# =============================
# 1) Graph 1: Asset mix (Equity/Debt/Cash)
# =============================
def mf_asset_allocation_by_fund_name(
    mf_portfolio: List[Dict],      # [{amc_name, fund_name, amount}]
    holdings_df: pd.DataFrame,     # columns: fund_name, category, isin, company_name, market_value
) -> Tuple[pd.DataFrame, Dict]:
    """
    EXACT PDF LOGIC:
      For each fund:
        raw sums by category
        net_equity = raw_equity + raw_derivative
        net_debt   = raw_debt
        net_cash   = raw_cash                  (if AMC contains Axis/Motilal)
                   = raw_cash - raw_derivative (otherwise)
        adjusted_aum = net_equity + net_debt + net_cash
        scale = user_amount / adjusted_aum
      Portfolio totals = sum(scale * per-fund nets)

    Returns:
      df_alloc: ['asset_class','value','pct'] where pct is out of total user MF amount
      meta: {'display_as_bar': bool, 'total_mf': float}
    """
    if holdings_df is None or holdings_df.empty or not mf_portfolio:
        df0 = pd.DataFrame([{"asset_class": k, "value": 0.0, "pct": 0.0} for k in ["Equity","Debt","Cash"]])
        return df0, {"display_as_bar": False, "total_mf": 0.0}

    df = holdings_df.copy()
    needed = ["fund_name","category","isin","company_name","market_value"]
    for c in needed:
        if c not in df.columns:
            df[c] = "" if c != "market_value" else 0.0
    df["fund_name_norm"] = df["fund_name"].astype(str).str.strip().str.lower()
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    # normalize selection
    selected = []
    for r in (mf_portfolio or []):
        amt = float(r.get("amount", 0) or 0)
        fn  = (r.get("fund_name","") or "").strip()
        if amt > 0 and fn:
            selected.append({"fund_name": fn, "fund_name_norm": fn.lower(), "amc": r.get("amc_name",""), "amount": amt})
    total_mf = sum(x["amount"] for x in selected)
    if total_mf <= 0:
        df0 = pd.DataFrame([{"asset_class": k, "value": 0.0, "pct": 0.0} for k in ["Equity","Debt","Cash"]])
        return df0, {"display_as_bar": False, "total_mf": 0.0}

    equity_total = debt_total = cash_total = 0.0
    any_negative_equity = False

    for s in selected:
        sub = df.loc[df["fund_name_norm"] == s["fund_name_norm"]]
        if sub.empty:
            # treat as cash if we have no holdings
            cash_total += s["amount"]
            continue

        # raw sums
        raw_eq  = float(sub.loc[sub["category"] == "Equity",      "market_value"].sum())
        raw_deb = float(sub.loc[sub["category"] == "Debt",        "market_value"].sum())
        raw_csh = float(sub.loc[sub["category"] == "Cash",        "market_value"].sum())
        raw_der = float(sub.loc[sub["category"] == "Derivatives", "market_value"].sum())

        # PDF rules
        net_equity = raw_eq + raw_der
        net_debt   = raw_deb
        if _is_axis_or_motilal(s["amc"]):
            net_cash = raw_csh
        else:
            net_cash = raw_csh - raw_der

        adjusted_aum = net_equity + net_debt + net_cash
        if adjusted_aum <= 0:
            # fall back: treat as cash to avoid divide-by-zero
            cash_total += s["amount"]
            continue

        scale = s["amount"] / adjusted_aum
        equity_total += net_equity * scale
        debt_total   += net_debt   * scale
        cash_total   += net_cash   * scale

        if net_equity < 0:
            any_negative_equity = True

    df_alloc = pd.DataFrame(
        [
            {"asset_class": "Equity", "value": equity_total},
            {"asset_class": "Debt",   "value": debt_total},
            {"asset_class": "Cash & Others",   "value": cash_total},
        ]
    )
    # percent out of total user MF rupees
    df_alloc["pct"] = 0.0 if total_mf == 0 else (df_alloc["value"] / total_mf * 100.0)

    meta = {"display_as_bar": any_negative_equity or (df_alloc["value"] < 0).any(), "total_mf": total_mf}
    return df_alloc, meta

# =============================
# Common base for Graphs 2–4: per-ISIN net equity (user-scaled)
# =============================
def _mf_user_scaled_net_equity_series_by_isin(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,
) -> pd.Series:
    """
    For each fund:
      - adjusted_aum as above
      - For every Equity row, compute adjusted_value = equity_mv + sum(derivative_mv for same ISIN)
      - Scaled rupees for that ISIN = (adjusted_value / adjusted_aum) * user_amount
    Combine across funds and return Series (index=isin, values=rupees).
    """
    if holdings_df is None or holdings_df.empty or not mf_portfolio:
        return pd.Series(dtype=float)

    df = holdings_df.copy()
    for c in ["fund_name","category","isin","company_name","market_value"]:
        if c not in df.columns:
            df[c] = "" if c != "market_value" else 0.0
    df["fund_name_norm"] = df["fund_name"].astype(str).str.strip().str.lower()
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    bucket: Dict[str, float] = {}

    for r in mf_portfolio or []:
        amt = float(r.get("amount", 0) or 0)
        fn  = (r.get("fund_name","") or "").strip()
        if amt <= 0 or not fn:
            continue
        amc = r.get("amc_name","")
        sub = df.loc[df["fund_name_norm"] == fn.lower()]
        if sub.empty:
            continue

        raw_eq  = float(sub.loc[sub["category"] == "Equity",      "market_value"].sum())
        raw_deb = float(sub.loc[sub["category"] == "Debt",        "market_value"].sum())
        raw_csh = float(sub.loc[sub["category"] == "Cash",        "market_value"].sum())
        raw_der = float(sub.loc[sub["category"] == "Derivatives", "market_value"].sum())

        net_equity = raw_eq + raw_der
        net_debt   = raw_deb
        net_cash   = raw_csh if _is_axis_or_motilal(amc) else (raw_csh - raw_der)
        adjusted_aum = net_equity + net_debt + net_cash
        if adjusted_aum <= 0:
            continue

        scale = amt / adjusted_aum

        eq_df = sub.loc[sub["category"] == "Equity", ["isin","market_value"]].copy()
        der_df = sub.loc[sub["category"] == "Derivatives", ["isin","market_value"]].copy()

        # sum by ISIN
        eq_by_isin  = eq_df.groupby("isin", as_index=True)["market_value"].sum() if not eq_df.empty else pd.Series(dtype=float)
        der_by_isin = der_df.groupby("isin", as_index=True)["market_value"].sum() if not der_df.empty else pd.Series(dtype=float)

        # For every equity row (ISIN in eq_by_isin), add all derivatives for same ISIN
        for isin, eq_val in eq_by_isin.items():
            dv = float(der_by_isin.get(isin, 0.0))
            adj_val = float(eq_val) + dv
            if adj_val == 0:
                continue
            rupees = adj_val * scale
            bucket[isin] = bucket.get(isin, 0.0) + rupees

    if not bucket:
        return pd.Series(dtype=float)
    s = pd.Series(bucket, dtype=float)
    # keep everything (positives and negatives) for completeness; the charting can drop negatives where needed
    return s

# =============================
# 2) Sector & 3) Market-cap pies (from net equity series)
# =============================
def mf_sector_and_mcap_exposure_by_fund_name(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,
    equities_info: pd.DataFrame,
    small_slice_threshold: float = 0.03,
) -> Dict[str, pd.DataFrame]:
    """
    Build pies from the combined user-weighted net equity series:
      - Sector pie: group by equities_info['industry_rating']
      - Market-cap pie: group by equities_info['market_cap']
      Positives only for pie denominator; fold <3% into 'Other'
    Returns {'sector': df, 'market_cap': df} where df has [label, value, pct]
    """
    series = _mf_user_scaled_net_equity_series_by_isin(mf_portfolio, holdings_df)
    pos = series[series > 0] if isinstance(series, pd.Series) else pd.Series(dtype=float)
    if pos.empty:
        empty = pd.DataFrame(columns=["label","value","pct"])
        return {"sector": empty, "market_cap": empty}

    # Map ISIN -> attributes
    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
    for c in ["isin","industry_rating","market_cap"]:
        if c not in eq.columns:
            eq[c] = "" if c == "isin" else "Unknown"
    eq["isin"] = eq["isin"].astype(str)
    eq["industry_rating"] = eq["industry_rating"].astype(str).replace({"": "Unknown"})
    eq["market_cap"] = eq["market_cap"].astype(str).replace({"": "Unknown"})

    m = pd.DataFrame({"isin": pos.index.astype(str), "value": pos.values}).merge(
        eq[["isin","industry_rating","market_cap"]], on="isin", how="left"
    )
    m["industry_rating"] = m["industry_rating"].fillna("Unknown")
    m["market_cap"] = m["market_cap"].fillna("Unknown")

    total = float(m["value"].sum())

    def _build(col: str) -> pd.DataFrame:
        g = m.groupby(col, as_index=False)["value"].sum().sort_values("value", ascending=False, ignore_index=True)
        g["pct"] = (g["value"] / total) * 100.0 if total != 0 else 0.0
        g = g.rename(columns={col: "label"})
        g = _fold_small_slices(g, "label", "value", small_slice_threshold)
        # re-normalize pct (1 decimal place handled in the view)
        pct_sum = float(g["pct"].sum())
        if pct_sum != 0:
            g["pct"] = g["pct"] * (100.0 / pct_sum)
        return g

    return {"sector": _build("industry_rating"), "market_cap": _build("market_cap")}

# =============================
# 4) Top-10 underlying net stock holdings (from net equity series)
# =============================
def mf_top_net_holdings_by_fund_name(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,
    equities_info: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank ISINs by user-weighted net equity rupees (positives only), show top N.
    Percent column is now % of TOTAL MF INVESTMENT (not % of equity).
    Returns: ['isin','company_name','value','pct_of_mf']
    """

    # user-weighted net equity per ISIN
    series = _mf_user_scaled_net_equity_series_by_isin(mf_portfolio, holdings_df)
    pos = series[series > 0] if isinstance(series, pd.Series) else pd.Series(dtype=float)

    # total rupees user invested in all MFs (denominator for percentage)
    total_mf = 0.0
    for r in (mf_portfolio or []):
        fn = (r.get("fund_name") or "").strip()
        amt = float(r.get("amount", 0) or 0)
        if fn and amt > 0:
            total_mf += amt

    if pos.empty:
        return pd.DataFrame(columns=["isin", "company_name", "value", "pct_of_mf"])

    s = pos.sort_values(ascending=False).head(top_n)

    # map ISIN -> company_name
    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
    if "isin" not in eq.columns:
        eq["isin"] = ""
    if "company_name" not in eq.columns:
        eq["company_name"] = ""
    name_map = dict(eq[["isin", "company_name"]].astype(str).values)

    out = pd.DataFrame({
        "isin": s.index.astype(str),
        "company_name": [name_map.get(isin, isin) for isin in s.index.astype(str)],
        "value": s.values,
    })
    out["pct_of_mf"] = (out["value"] / total_mf * 100.0) if total_mf > 0 else 0.0

    return out.sort_values("value", ascending=False, ignore_index=True)

def _normalize_pct_series_to_percent(s: pd.Series) -> pd.Series:
    """If sums ≲ 1.5, treat as fraction and convert to percent."""
    if s is None or s.empty:
        return s
    total = float(s.sum())
    return (s * 100.0) if total <= 1.5 else s

def _fund_equity_pct_by_isin(holdings_df: pd.DataFrame, fund_name: str) -> pd.Series:
    """
    Series(index=isin, values=pct of net assets in Equity only) for a single fund.
    If multiple rows exist per ISIN, values are summed.
    """
    if holdings_df is None or holdings_df.empty or not fund_name:
        return pd.Series(dtype=float)

    df = holdings_df.copy()
    for c in ["fund_name", "category", "isin"]:
        if c not in df.columns:
            df[c] = ""
    pct_col = "pct_net_assets" if "pct_net_assets" in df.columns else ("percentage" if "percentage" in df.columns else None)
    if pct_col is None:
        return pd.Series(dtype=float)

    sub = df[(df["fund_name"].astype(str).str.strip() == fund_name.strip()) & (df["category"] == "Equity")]
    if sub.empty:
        return pd.Series(dtype=float)

    sub["isin"] = sub["isin"].astype(str)
    s = sub.groupby("isin")[pct_col].sum(min_count=1)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = _normalize_pct_series_to_percent(s)
    # keep non-negative (typically ≥0)
    s = s[s >= 0]
    return s.sort_values(ascending=False)

def mf_overlap_matrix_by_pct_net_assets(
    mf_portfolio: List[Dict],
    holdings_df: pd.DataFrame,
    equities_info: pd.DataFrame | None = None,
    top_common_names: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Build an NxN symmetric matrix of overlap % between selected funds (Equity only)
    using: overlap(A,B) = sum_i min(pctA_i, pctB_i).
    Returns dict with:
      - 'funds'         : list of fund names in order
      - 'overlap_pct'   : DataFrame (NxN) of percentages (diagonal=100)
      - 'common_count'  : DataFrame (NxN) counts of common ISINs
      - 'top_common'    : DataFrame (NxN) csv of top common names by min(%)
    """
    # Selected funds (keep order, dedupe)
    funds: List[str] = []
    for r in (mf_portfolio or []):
        fn = (r.get("fund_name") or "").strip()
        if fn:
            funds.append(fn)
    funds = list(dict.fromkeys(funds))

    n = len(funds)
    if n == 0:
        return {"funds": [], "overlap_pct": pd.DataFrame(), "common_count": pd.DataFrame(), "top_common": pd.DataFrame()}

    # Precompute per-fund equity % by ISIN
    series_map: Dict[str, pd.Series] = {fn: _fund_equity_pct_by_isin(holdings_df, fn) for fn in funds}

    # Optional name map for hover strings
    name_map: Dict[str, str] = {}
    if equities_info is not None and not equities_info.empty:
        eq = equities_info.copy()
        if "isin" not in eq.columns: eq["isin"] = ""
        if "company_name" not in eq.columns: eq["company_name"] = ""
        name_map = dict(eq[["isin","company_name"]].astype(str).values)

    # Matrices
    import numpy as np
    M = pd.DataFrame(np.zeros((n, n), dtype=float), index=funds, columns=funds)
    C = pd.DataFrame(np.zeros((n, n), dtype=int),   index=funds, columns=funds)
    T = pd.DataFrame([[""]*n for _ in range(n)],    index=funds, columns=funds)

    for i, fa in enumerate(funds):
        sa = series_map.get(fa, pd.Series(dtype=float))
        for j, fb in enumerate(funds):
            if i == j:
                M.iat[i, j] = 100.0
                C.iat[i, j] = int(len(sa)) if isinstance(sa, pd.Series) else 0
                T.iat[i, j] = ""
                continue
            sb = series_map.get(fb, pd.Series(dtype=float))
            if sa.empty or sb.empty:
                M.iat[i, j] = 0.0
                C.iat[i, j] = 0
                T.iat[i, j] = ""
                continue
            common = list(set(sa.index) & set(sb.index))
            C.iat[i, j] = len(common)
            if not common:
                M.iat[i, j] = 0.0
                T.iat[i, j] = ""
            else:
                va = sa.reindex(common)
                vb = sb.reindex(common)
                mins = pd.concat([va, vb], axis=1).min(axis=1)
                overlap = float(mins.sum())
                M.iat[i, j] = overlap
                if top_common_names > 0:
                    top = mins.sort_values(ascending=False).head(top_common_names)
                    labels = []
                    for isin in top.index:
                        labels.append(name_map.get(str(isin), str(isin)))
                    T.iat[i, j] = ", ".join(labels)

    return {"funds": funds, "overlap_pct": M, "common_count": C, "top_common": T}


__all__ = [
    "mf_asset_allocation_by_fund_name",
    "_equity_net_by_isin_scaled_for_user_funds",
    "mf_sector_and_mcap_exposure_by_fund_name",
    "mf_top_net_holdings_by_fund_name",
    "mf_fund_track_record_by_fund_name",
]

__all__ += [
    "stocks_equity_by_isin",
    "sector_mcap_from_series",
    "top_holdings_from_series",
    "mf_net_equity_series_by_fund_name",
    "combined_equity_by_isin",
    "combined_composition_split",
    "fund_net_equity_series_map",
    "fund_overlap_matrix",
    "fund_vs_direct_overlap",
    "_map_isin_to_attrs"
]