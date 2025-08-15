# page_analysis.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from core.ui import card_start, card_end, pie_grid, sleek_table
from core.data_access import load_holdings_for_funds
from core.api import get_fund_performance
from core.transforms import (
    group_small_slices,
    compute_mf_exposures,
    build_fund_net_equity_weights,
    compute_overlap_matrix,
)

def _fmt_pct(x):
    try: return f"{float(x)*100:,.2f}%"
    except: return "—"

def _compact_bar(fig, n):
    fig.update_layout(height=420 if n >= 8 else 340)
    return fig

def render(equities_info: pd.DataFrame, conn):
    st.subheader("Step 3 · Analysis")

    # ---------- Mutual Funds ----------
    mf_portfolio = st.session_state.get("mf_portfolio", [])
    fund_names = [r["fund_name"] for r in mf_portfolio]
    mf_holdings = load_holdings_for_funds(conn, fund_names)

    res = compute_mf_exposures(mf_holdings, mf_portfolio)
    totals = res["totals"]
    assets = res["net_equity_assets"]   # isin, company_name, industry_rating, exposure

    total_mf = totals["total_mf"]; equity_total = totals["equity"]; debt_total = totals["debt"]; cash_total = totals["cash"]
    any_neg = totals["any_negative_fund_equity"]

    # Asset allocation (pie unless negative)
    st.markdown("### Where is your MF money allocated across Equity, Debt, and Cash & Others?")
    alloc = pd.DataFrame([
        {"Class": "Equity", "₹": equity_total},
        {"Class": "Debt", "₹": debt_total},
        {"Class": "Cash & Others", "₹": cash_total},
    ])
    if (alloc["₹"] >= 0).all() and not any_neg:
        figA = px.pie(alloc, names="Class", values="₹")
        figA.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
    else:
        figA = px.bar(alloc.sort_values("₹", ascending=True), x="₹", y="Class", orientation="h", labels={"₹":"Exposure (₹)"})
        figA = _compact_bar(figA, len(alloc))
    st.plotly_chart(figA, use_container_width=True)

    # Sector & Market-cap (MF equity only)
    if not assets.empty:
        look = equities_info[["isin","company_name","industry_rating","market_cap","sector"]]
        joined = assets.merge(look, on=["isin","company_name","industry_rating"], how="left")
        sector = (joined.groupby("sector", as_index=False)["exposure"].sum()
                        .rename(columns={"sector":"Sector","exposure":"Exposure"}))
        mcap = (joined.groupby("market_cap", as_index=False)["exposure"].sum()
                        .rename(columns={"market_cap":"Market-cap","exposure":"Exposure"}))
        sector = group_small_slices(sector, "Sector", "Exposure", 0.03, "Other")
        mcap = group_small_slices(mcap, "Market-cap", "Exposure", 0.03, "Other")

        fig1 = px.pie(sector, names="Sector", values="Exposure")
        fig1.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
        fig2 = px.pie(mcap, names="Market-cap", values="Exposure")
        fig2.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>")
        pie_grid([fig1, fig2])

        st.markdown("### What MF underlying stocks are you most exposed to?")
        top = joined.groupby(["isin","company_name"], as_index=False)["exposure"].sum().sort_values("exposure", ascending=False).head(10)
        figT = px.bar(top.sort_values("exposure"), x="exposure", y="company_name", orientation="h", labels={"exposure":"Exposure (₹)","company_name":""})
        figT = _compact_bar(figT, len(top))
        st.plotly_chart(figT, use_container_width=True)
    else:
        st.info("No MF equity exposure to chart yet.")

    # Performance table (optional: only if fund_code available)
    perf_rows = []
    for r in mf_portfolio:
        perf = None
        # if you have fund_code in fund_info, attach it into portfolio and use here; else skip
        code = r.get("fund_code")
        if code:
            try:
                perf = get_fund_performance(code)
            except Exception:
                perf = None
        perf_rows.append({
            "Fund": r["fund_name"],
            "NAV (₹)": None if not perf else f"{perf['nav']:,.2f}",
            "As of": None if not perf else perf["as_of"],
            "3Y": None if not perf else _fmt_pct(perf["returns"].get("3Y")),
            "5Y": None if not perf else _fmt_pct(perf["returns"].get("5Y")),
            "10Y": None if not perf else _fmt_pct(perf["returns"].get("10Y")),
            "SI": None if not perf else _fmt_pct(perf["returns"].get("SI")),
        })
    if perf_rows:
        st.markdown("#### Fund Returns (CAGR %) — Latest available")
        sleek_table(pd.DataFrame(perf_rows))

    # ---------- Direct Stocks ----------
    st.markdown("---")
    st.markdown("## Direct Stocks")
    stk = st.session_state.get("stock_portfolio", [])
    if not stk:
        st.info("No direct stocks added.")
    else:
        df_stk = pd.DataFrame(stk)
        look = equities_info[["company_name","sector","market_cap"]]
        joined = df_stk.merge(look, on="company_name", how="left")
        # charts
        s_sector = joined.groupby("sector", as_index=False)["amount"].sum().rename(columns={"sector":"Sector","amount":"Exposure"})
        s_mcap = joined.groupby("market_cap", as_index=False)["amount"].sum().rename(columns={"market_cap":"Market-cap","amount":"Exposure"})
        s_sector = group_small_slices(s_sector, "Sector", "Exposure", 0.03, "Other")
        s_mcap = group_small_slices(s_mcap, "Market-cap", "Exposure", 0.03, "Other")
        figS1 = px.pie(s_sector, names="Sector", values="Exposure"); figS1.update_traces(textinfo="label+percent")
        figS2 = px.pie(s_mcap, names="Market-cap", values="Exposure"); figS2.update_traces(textinfo="label+percent")
        pie_grid([figS1, figS2])
        # top
        t = joined.groupby("company_name", as_index=False)["amount"].sum().rename(columns={"amount":"Exposure"}).sort_values("Exposure", ascending=False).head(10)
        figSB = px.bar(t.sort_values("Exposure"), x="Exposure", y="company_name", orientation="h")
        st.plotly_chart(figSB, use_container_width=True)

    # ---------- Combined & Overlap ----------
    st.markdown("---")
    st.markdown("## Fund Overlap (Equity exposure)")
    weights = build_fund_net_equity_weights(mf_holdings, mf_portfolio)
    if len(weights) >= 2:
        M = compute_overlap_matrix(weights)  # 0..1
        sleek_table((M * 100).round(2).astype(str) + "%")
    else:
        st.info("Add at least two funds to view overlap.")
