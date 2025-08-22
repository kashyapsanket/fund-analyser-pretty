# views/page_analysis.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
# UI helpers (already in your ui.py)
from core.ui import page_head, sticky_footer

# MF transforms (we added these in core/transforms.py)
from core.transforms import *

from core.api import get_fund_performance

CHART_KEYS = {
    # Mutual Funds
    "mf_asset_mix": "mf_asset_mix",
    "mf_sector_pie": "mf_sector_pie",
    "mf_mcap_pie": "mf_mcap_pie",
    "mf_top10_bar": "mf_top10_bar",

    # Direct Stocks
    "ds_top10_bar": "ds_top10_bar",
    "ds_sector_pie": "ds_sector_pie",
    "ds_mcap_pie": "ds_mcap_pie",

    # Combined
    "cmb_asset_mix": "cmb_asset_mix",
    "cmb_equity_source": "cmb_equity_source",
    "cmb_sector_pie": "cmb_sector_pie",
    "cmb_mcap_pie": "cmb_mcap_pie",
    "cmb_top10_bar": "cmb_top10_bar",

    # Overlap
    "ov_heatmap": "ov_heatmap",
}

def st_plot(fig, key):
    """Standardized plotly renderer with a stable unique key."""
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key=key)


def _fmt_inr(v: float) -> str:
    try:
        return f"₹{float(v):,.0f}"
    except Exception:
        return "₹0"

def render(
    equities_info: pd.DataFrame,
    fund_info: pd.DataFrame | None = None,      # kept for compatibility; not used in MF calcs now
    holdings_info: pd.DataFrame | None = None,
):
    # ---------- Header ----------
    page_head(
        title="Portfolio Analysis",
        subtitle="Personalized insights about your portfolio",
        icon="📊",
    )

    # ---------- Tabs ----------
    tabs = st.tabs(["Mutual Funds", "Direct Stocks", "Combined", "Overlap"])

    # ===========================================================
    # TAB 1: MUTUAL FUNDS
    # ===========================================================

# --- MUTUAL FUNDS TAB ---------------------------------------------------------
# --- MUTUAL FUNDS TAB ---------------------------------------------------------
    with tabs[0]:
        # ── helpers ────────────────────────────────────────────────────────────────
        PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
        EPS = 1e-8

        def _z(v: float) -> float:
            """Clamp tiny magnitudes to 0.0 so '-0.0' never shows up."""
            return 0.0 if abs(float(v)) < EPS else float(v)

        def donut(labels, values, height=360):
            vals = [_z(v) for v in values]
            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=vals,
                    hole=0.45,
                    textinfo="percent",
                    hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
                )
            )
            fig.update_layout(
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),  # safe margins (titles shown in Streamlit, not Plotly)
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5),
            )
            return fig

        def hbar(x, y, height=360):
            """Generic hbar that computes % from x."""
            xx = [_z(v) for v in x]
            total = max(float(np.array(xx).sum()), 1e-9)
            fig = go.Figure()
            fig.add_bar(
                x=xx,
                y=y,
                orientation="h",
                text=[f"{(float(v)/total*100):.1f}%" for v in xx],
                textposition="auto",
                hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
            )
            fig.update_layout(
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),
                xaxis_title="Rupees",
                yaxis_title="",
            )
            return fig

        def hbar_with_text(x, y, text, height=360):
            """Hbar that uses the given text (e.g., '% of Equity') as bar labels."""
            xx = [_z(v) for v in x]
            tt = [str(t) for t in text]
            fig = go.Figure()
            fig.add_bar(
                x=xx,
                y=y,
                orientation="h",
                text=tt,
                textposition="auto",
                hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
            )
            fig.update_layout(
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),
                xaxis_title="Rupees",
                yaxis_title="",
            )
            return fig

        def chart_heading(title: str, subtitle: str = ""):
            st.markdown(
                f"""
                <div style="padding:6px 4px 2px 4px">
                <div style="font-weight:700; font-size:1.02rem; line-height:1.35">{title}</div>
                <div style="color:#A0A0A0; font-size:0.84rem; margin-top:2px">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── content ────────────────────────────────────────────────────────────────
        st.caption("Deep dive into the underlying composition of your mutual fund investments.")

        mf_rows = st.session_state.get("mf_portfolio", []) or []
        if not mf_rows:
            st.info("Add mutual funds in Step 1 to see your personalized breakdown.")
        elif holdings_info is None or holdings_info.empty:
            st.info("No holdings data available. Ensure holdings_info is loaded.")
        else:
            # 1) Asset mix (Equity / Debt / Cash)
            alloc_df, meta = mf_asset_allocation_by_fund_name(mf_rows, holdings_info)

            df_show = alloc_df.copy()
            df_show["value"] = df_show["value"].apply(_z)  # kill '-0.0'
            df_show["% of MF"] = df_show["pct"].map(lambda v: f"{float(v):.1f}%")
            df_show["Value (₹)"] = df_show["value"].map(lambda v: f"₹{float(v):,.0f}")

            chart_heading(
                "Which Asset Classes have I invested in?",
                "Artibrage Funds are stored as Debt. Gold, ETFs, and otheer commodities are listed as Cash."
            )
            if meta.get("display_as_bar", False):
                plot_df = df_show.sort_values("value", ascending=True)
                st.plotly_chart(hbar(plot_df["value"], plot_df["asset_class"]), use_container_width=True, config=PLOT_CFG, key="alpha")
            else:
                st.plotly_chart(donut(df_show["asset_class"], df_show["value"]), use_container_width=True, config=PLOT_CFG, key = "alpha_2")

            with st.expander("Details", expanded=False):
                st.dataframe(
                    df_show[["asset_class", "Value (₹)", "% of MF"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("---")

            # 2) Sector & 3) Market-cap distribution (user-weighted net equity, positives only)
            sm = mf_sector_and_mcap_exposure_by_fund_name(
                mf_rows, holdings_info, equities_info, small_slice_threshold=0.03
            )
            sector_df = sm["sector"]
            mcap_df = sm["market_cap"]

            # Sector
            if sector_df.empty:
                st.info("No Equities are available for Sector Distribution")
            else:
                sd = sector_df.copy()
                sd["% of Equity"] = sd["pct"].map(lambda v: f"{float(v):.1f}%")
                sd["Value (₹)"] = sd["value"].map(lambda v: f"₹{float(v):,.0f}")

                chart_heading(
                    "What sectors have I invested in?",
                    "Sectoral distribution as a percentage of net equity in your portfolio"
                )
                st.plotly_chart(donut(sd["label"], sd["value"]), use_container_width=True, config=PLOT_CFG, key="alpha_3")
                with st.expander("Details", expanded=False):
                    st.dataframe(
                        sd[["label", "Value (₹)", "% of Equity"]].rename(columns={"label": "Sector"}),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.markdown("---")

            # Market-cap
            if mcap_df.empty:
                st.info("No Equities are available for Market-Cap Distribution")
            else:
                mc = mcap_df.copy()
                mc["% of Equity"] = mc["pct"].map(lambda v: f"{float(v):.1f}%")
                mc["Value (₹)"] = mc["value"].map(lambda v: f"₹{float(v):,.0f}")

                chart_heading(
                    "What Market-Caps have I invested in?",
                    "Market-Cap distribution as a percentage of net equity in your portfolio"
                )
                st.plotly_chart(donut(mc["label"], mc["value"]), use_container_width=True, config=PLOT_CFG, key="aplha_4")
                with st.expander("Details", expanded=False):
                    st.dataframe(
                        mc[["label", "Value (₹)", "% of Equity"]].rename(columns={"label": "Market Cap"}),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.markdown("---")

            # 4) Top-10 underlying net stocks (positives only)
            df_top = mf_top_net_holdings_by_fund_name(mf_rows, holdings_info, equities_info, top_n=10)
            if df_top.empty:
                st.info("No Equities present to show top 10 Holdings")
            else:
                tt = df_top.copy()
                # use the canonical percent from transforms for both chart + table
                tt["% of Portfolio"] = tt["pct_of_mf"].map(lambda v: f"{float(v):.1f}%")

                chart_heading(
                    "What are my top 10 Stock Holdings?",
                    "Top 10 holdings as a percentage of overall mutual fund portfolio"
                )
                st.plotly_chart(
                    hbar_with_text(tt["value"], tt["company_name"], tt["% of Portfolio"]),
                    use_container_width=True,
                    config=PLOT_CFG,
                    key = "alpha_5"
                )
                with st.expander("Details", expanded=False):
                    show = tt.assign(**{"Value (₹)": tt["value"].map(lambda v: f"₹{float(v):,.0f}")})[
                        ["company_name", "isin", "Value (₹)", "% of Portfolio"]
                    ].rename(columns={"company_name": "Company"})
                    st.dataframe(show, hide_index=True, use_container_width=True)

            st.markdown("---")

            # 5) Fund Track Record (API-driven; weighted summary row inside the same table)
            chart_heading(
                "Historical Performance",
                "Returns upto 1 year are absolute; For more than 1 year annualized returns are shown."
            )

            import re

            def _norm_name(x: str) -> str:
                """Case/whitespace-insensitive normalizer for joining by fund_name."""
                s = "" if x is None else str(x)
                s = re.sub(r"\s+", " ", s).strip()
                return s.casefold()

            # Build selected funds strictly by FUND NAME (then fetch code from fund_info)
            sel = []
            for r in (mf_rows or []):
                fn = (r.get("fund_name") or "").strip()
                try:
                    amt = float(r.get("amount", 0) or 0)
                except Exception:
                    amt = 0.0
                if fn and amt > 0:
                    sel.append({"fund_name": fn, "amount": amt})

            if not sel:
                st.info("No funds to show. Add funds in Step 1.")
            else:
                sel_df = pd.DataFrame(sel)
                sel_df["fn_norm"] = sel_df["fund_name"].map(_norm_name)

                # Prepare fund_info and map fund_name -> fund_code (authoritative)
                fi = fund_info.copy() if fund_info is not None else pd.DataFrame()
                for c in ["fund_name", "fund_code", "amc", "category"]:
                    if c not in fi.columns:
                        fi[c] = None
                fi["fn_norm"] = fi["fund_name"].map(_norm_name)

                # Drop dupes on the normalized name to avoid 1:N merges (keep first occurrence)
                fi_map = fi.drop_duplicates(subset=["fn_norm"])[["fn_norm", "fund_name", "fund_code", "amc", "category"]]

                # Join by normalized fund_name
                m = sel_df.merge(fi_map, on="fn_norm", how="left")

                total_amt = float(m["amount"].sum()) or 0.0
                m["weight"] = 0.0 if total_amt == 0 else m["amount"] / total_amt

                # Fetch NAV/returns per mapped fund_code
                rows = []
                for _, r in m.iterrows():
                    code = r.get("fund_code", None)
                    perf = None
                    try:
                        if pd.notna(code) and code not in ("", 0):
                            perf = get_fund_performance(int(code))
                    except Exception:
                        perf = None

                    rets = (perf or {}).get("returns", {}) or {}
                    rows.append({
                        # Prefer canonical name from fund_info when available, else the selected name
                        "Fund": r.get("fund_name_y") if pd.notna(r.get("fund_name_y")) else r.get("fund_name_x") or r.get("fund_name") or "",
                        "Amount": float(r.get("amount", 0.0)),
                        "Weight": float(r.get("weight", 0.0)),  # 0..1
                        "NAV": (perf or {}).get("nav", np.nan),
                        "As of": (perf or {}).get("as_of", "—"),
                        "1M": rets.get("1M", np.nan),
                        "3M": rets.get("3M", np.nan),
                        "6M": rets.get("6M", np.nan),
                        "1Y": rets.get("1Y", np.nan),
                        "3Y": rets.get("3Y", np.nan),
                        "5Y": rets.get("5Y", np.nan),
                        "10Y": rets.get("10Y", np.nan),
                        "SI": rets.get("SI", np.nan),
                    })

                df_tr = pd.DataFrame(rows)

                # Weighted summary row (in the same table)
                def wavg(col: str):
                    vals = df_tr[col]
                    if vals.notna().any():
                        return float((vals.fillna(0.0) * df_tr["Weight"]).sum())
                    return np.nan

                summary_row = {
                    "Fund": "Weighted Average Portfolio Returns",
                    "Amount": df_tr["Amount"].sum(),
                    "Weight": 1.0,
                    "NAV": np.nan,
                    "As of": "—",
                    "1M": wavg("1M"), "3M": wavg("3M"), "6M": wavg("6M"),
                    "1Y": wavg("1Y"), "3Y": wavg("3Y"), "5Y": wavg("5Y"),
                    "10Y": wavg("10Y"), "SI": wavg("SI"),
                }

                df_tr_display = pd.concat([df_tr, pd.DataFrame([summary_row])], ignore_index=True)

                # Format for display
                show = df_tr_display.copy()
                show["Amount (₹)"] = show["Amount"].map(lambda v: f"₹{_z(v):,.0f}")
                show["Weight %"]   = show["Weight"].map(lambda v: f"{float(v)*100:.1f}%")
                show["NAV"]        = show["NAV"].map(lambda v: "—" if pd.isna(v) else f"{float(v):,.2f}")
                for k in ["1M","3M","6M","1Y","3Y","5Y","10Y","SI"]:
                    show[k] = show[k].map(lambda v: "—" if pd.isna(v) else f"{float(v)*100:.1f}%")

                cols = ["Fund","Amount (₹)","Weight %","NAV","As of","1M","3M","6M","1Y","3Y","5Y","10Y","SI"]
                st.dataframe(show[cols], hide_index=True, use_container_width=True)

            
                
    # ===========================================================
    # TAB 2: DIRECT STOCKS (placeholder for next steps)
    # ===========================================================
# --- DIRECT STOCKS TAB --------------------------------------------------------
    with tabs[1]:
        # ── helpers (mirrors Mutual Funds tab) ─────────────────────────────────────
        PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
        EPS = 1e-8

        def _z(v: float) -> float:
            return 0.0 if abs(float(v)) < EPS else float(v)

        def _fmt_inr(v: float) -> str:
            return f"₹{_z(float(v)):,.0f}"

        def _parse_amount(val) -> float:
            """Parse amounts like '₹10,00,000', '10,000', 10000.5 → float."""
            if val is None:
                return 0.0
            if isinstance(val, (int, float, np.number)):
                try:
                    return float(val)
                except Exception:
                    return 0.0
            s = re.sub(r"[^0-9\.\-]", "", str(val))
            try:
                return float(s) if s not in ("", "-", ".", "-.", ".-") else 0.0
            except Exception:
                return 0.0

        def donut(labels, values, height=360):
            vals = [_z(v) for v in values]
            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=vals,
                    hole=0.45,
                    textinfo="percent",
                    hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
                )
            )
            fig.update_layout(
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),  # titles are outside; safe margins
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5),
            )
            return fig

        def hbar_with_text(x, y, text, height=360):
            xx = [_z(v) for v in x]
            tt = [str(t) for t in text]
            fig = go.Figure()
            fig.add_bar(
                x=xx,
                y=y,
                orientation="h",
                text=tt,
                textposition="auto",
                hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
            )
            fig.update_layout(
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),
                xaxis_title="Rupees",
                yaxis_title="",
            )
            return fig

        def chart_heading(title: str, subtitle: str = ""):
            st.markdown(
                f"""
                <div style="padding:6px 4px 2px 4px">
                <div style="font-weight:700; font-size:1.02rem; line-height:1.35">{title}</div>
                <div style="color:#A0A0A0; font-size:0.84rem; margin-top:2px">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def render_empty_state():
            st.markdown(
                """
                <div style="
                    border:1px solid #333333; border-radius:12px;
                    padding:24px; text-align:center; background:#1a1b22;">
                <div style="font-size:32px; line-height:1">📉</div>
                <div style="margin-top:6px; font-weight:600;">No direct stocks yet</div>
                <div style="color:#A0A0A0; margin-top:4px;">
                    Add your equity positions in <b>Step 2: Direct Stocks</b> to see this view.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── content ────────────────────────────────────────────────────────────────
        st.caption("Breakdown of your direct equity positions.")

        # Raw rows from session
        stocks_rows = st.session_state.get("stock_portfolio", []) or []

        # Show a friendly placeholder if the user hasn't added any stocks or all amounts are non-positive
        if (not stocks_rows) or (not any(_parse_amount(r.get("amount")) > 0 for r in stocks_rows)):
            render_empty_state()
        else:
            # Build direct-equity rupee exposure by ISIN
            s_direct = stocks_equity_by_isin(stocks_rows, equities_info)  # Series[isin] = rupees
            total_direct = float(s_direct.sum()) if isinstance(s_direct, pd.Series) and not s_direct.empty else 0.0

            if total_direct <= 0:
                # Inputs exist but could not be mapped to ISINs — still show a placeholder
                render_empty_state()
            else:
                # KPIs
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Total Direct Equity", _fmt_inr(total_direct))
                with k2:
                    st.metric("Positions", f"{len(s_direct)}")
                with k3:
                    df_one = top_holdings_from_series(s_direct, equities_info, top_n=1)
                    top_val = float(df_one["value"].iloc[0]) if not df_one.empty else 0.0
                    st.metric("Largest Holding", _fmt_inr(top_val))

                st.markdown("---")

                # Top 10 direct holdings (bars use the exact % shown in details)
                df_top = top_holdings_from_series(s_direct, equities_info, top_n=10)
                if df_top.empty:
                    render_empty_state()
                else:
                    df_top = df_top.copy()
                    df_top["pct_of_direct"] = 0.0 if total_direct == 0 else (df_top["value"] / total_direct * 100.0)
                    df_top["% of Direct"] = df_top["pct_of_direct"].map(lambda v: f"{float(v):.1f}%")

                    chart_heading(
                        "Top 10 Direct Holdings",
                        "Rupee exposure and share of your direct equity."
                    )
                    st.plotly_chart(
                        hbar_with_text(df_top["value"], df_top["company_name"], df_top["% of Direct"]),
                        use_container_width=True,
                        config=PLOT_CFG,
                        key = "beta"
                    )
                    with st.expander("Details", expanded=False):
                        show = df_top.assign(**{"Value (₹)": df_top["value"].map(_fmt_inr)})[
                            ["company_name", "isin", "Value (₹)", "% of Direct"]
                        ].rename(columns={"company_name": "Company"})
                        st.dataframe(show, hide_index=True, use_container_width=True)

                st.markdown("---")

                # Sector & Market-cap pies (positives only), with same headings & details
                pies = sector_mcap_from_series(s_direct, equities_info, small_slice_threshold=0.03)
                c1, c2 = st.columns(2)

                with c1:
                    sd = pies.get("sector", pd.DataFrame())
                    if sd is None or sd.empty:
                        st.info("No sector data.")
                    else:
                        sd = sd.copy()
                        sd["% of Equity"] = sd["pct"].map(lambda v: f"{float(v):.1f}%")
                        sd["Value (₹)"] = sd["value"].map(_fmt_inr)

                        chart_heading(
                            "Sector Distribution",
                            "Positive direct equity only. Small slices grouped as Other."
                        )
                        st.plotly_chart(donut(sd["label"], sd["value"]), use_container_width=True, config=PLOT_CFG, key="beta_2")
                        with st.expander("Details", expanded=False):
                            st.dataframe(
                                sd[["label","Value (₹)","% of Equity"]].rename(columns={"label": "Sector"}),
                                hide_index=True,
                                use_container_width=True,
                            )

                with c2:
                    mc = pies.get("market_cap", pd.DataFrame())
                    if mc is None or mc.empty:
                        st.info("No market-cap data.")
                    else:
                        mc = mc.copy()
                        mc["% of Equity"] = mc["pct"].map(lambda v: f"{float(v):.1f}%")
                        mc["Value (₹)"] = mc["value"].map(_fmt_inr)

                        chart_heading(
                            "Market-Cap Distribution",
                            "Positive direct equity only. Small slices grouped as Other."
                        )
                        st.plotly_chart(donut(mc["label"], mc["value"]), use_container_width=True, config=PLOT_CFG, key="beta_3")
                        with st.expander("Details", expanded=False):
                            st.dataframe(
                                mc[["label","Value (₹)","% of Equity"]].rename(columns={"label": "Market Cap"}),
                                hide_index=True,
                                use_container_width=True,
                            )

                st.markdown("---")

                # Full position details
                with st.expander("All Positions (details)", expanded=False):
                    details = pd.DataFrame({"isin": s_direct.index.astype(str), "value": s_direct.values})
                    eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
                    for c in ["isin","company_name","sector","market_cap","industry_rating"]:
                        if c not in eq.columns:
                            eq[c] = "" if c != "isin" else eq.get("isin", "")
                    eq["isin"] = eq["isin"].astype(str)
                    details = details.merge(
                        eq[["isin","company_name","sector","market_cap","industry_rating"]],
                        on="isin", how="left"
                    )
                    details = details.rename(columns={"industry_rating":"rating"})
                    details["Value (₹)"] = details["value"].map(_fmt_inr)
                    st.dataframe(
                        details[["company_name","isin","sector","market_cap","rating","Value (₹)"]]
                        .rename(columns={"company_name":"Company","rating":"Industry Rating"}),
                        hide_index=True, use_container_width=True
                    )


# ===========================================================
# TAB 3: COMBINED (MF Net Equity + Direct)
# ===========================================================

# --- COMBINED PORTFOLIO TAB ---------------------------------------------------
    with tabs[2]:
        # avoid UnboundLocalError by importing under an alias once
        # ── helpers (shared visual + math bits) ────────────────────────────────────
        PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
        EPS = 1e-8

        def _z(v: float) -> float:
            return 0.0 if abs(float(v)) < EPS else float(v)

        def _fmt_inr(v: float) -> str:
            return f"₹{_z(float(v)):,.0f}"

        def chart_heading(title: str, subtitle: str = ""):
            st.markdown(
                f"""
                <div style="padding:6px 4px 2px 4px">
                <div style="font-weight:700; font-size:1.02rem; line-height:1.35">{title}</div>
                <div style="color:#A0A0A0; font-size:0.84rem; margin-top:2px">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def donut(labels, values, height=360):
            vals = [_z(v) for v in values]
            fig = go.Figure(
                go.Pie(
                    labels=labels, values=vals, hole=0.45, textinfo="percent",
                    hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
                )
            )
            fig.update_layout(
                height=height, margin=dict(l=16, r=16, t=16, b=16),
                showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5),
            )
            return fig

        def hbar_with_text(x, y, text, height=360):
            xx = [_z(v) for v in x]
            tt = [str(t) for t in text]
            fig = go.Figure()
            fig.add_bar(
                x=xx, y=y, orientation="h",
                text=tt, textposition="auto",
                hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
            )
            fig.update_layout(
                height=height, margin=dict(l=16, r=16, t=16, b=16),
                xaxis_title="", yaxis_title="",
            )
            return fig

        def _fold_small_slices(df: pd.DataFrame, label_col: str, value_col: str, pct_col: str, thr: float = 0.03) -> pd.DataFrame:
            if df.empty:
                return df
            small = df[pct_col] < (thr * 100.0)
            if not small.any():
                return df
            big = df.loc[~small, [label_col, value_col, pct_col]]
            other = pd.DataFrame([{
                label_col: "Other",
                value_col: float(df.loc[small, value_col].sum()),
                pct_col: float(df.loc[small, pct_col].sum()),
            }])
            out = pd.concat([big, other], ignore_index=True).sort_values(value_col, ascending=False, ignore_index=True)
            ps = float(out[pct_col].sum())
            if ps not in (0.0, 100.0):
                out[pct_col] = out[pct_col] * (100.0 / ps)
            return out

        def _is_axis_or_motilal(amc: str) -> bool:
            s = (str(amc) or "").strip().lower()
            return ("axis" in s) or ("motilal oswal" in s)

        def _build_mf_series_fallback(mf_rows: list[dict], holdings_df: pd.DataFrame) -> pd.Series:
            """User-scaled net equity per ISIN from MFs (Equity + matched Derivatives), positives only."""
            if holdings_df is None or holdings_df.empty or not mf_rows:
                return pd.Series(dtype=float)
            df = holdings_df.copy()
            for c in ["fund_name","category","isin","company_name","market_value","amc"]:
                if c not in df.columns:
                    df[c] = "" if c != "market_value" else 0.0
            df["fund_name_norm"] = df["fund_name"].astype(str).str.strip().str.lower()
            df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

            result: dict[str, float] = {}
            for r in mf_rows:
                fn = (r.get("fund_name") or "").strip()
                amt = float(r.get("amount", 0) or 0)
                amc = r.get("amc_name", "")
                if not fn or amt <= 0:
                    continue
                sub = df[df["fund_name_norm"] == fn.lower()]
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

                eq = sub[sub["category"] == "Equity"]
                der = sub[sub["category"] == "Derivatives"]
                der_map = der.groupby("isin")["market_value"].sum().to_dict()
                for _, row in eq.iterrows():
                    isin = str(row.get("isin",""))
                    base = float(row.get("market_value", 0.0))
                    adj  = base + float(der_map.get(isin, 0.0))
                    if adj <= 0:
                        continue
                    result[isin] = result.get(isin, 0.0) + (adj * scale)

            return pd.Series(result, dtype=float)

        def _combined_equity_series(mf_rows: list[dict], holdings_df: pd.DataFrame, s_direct: pd.Series) -> pd.Series:
            """MF net (scaled) + Direct; positives only for pies/top-10."""
            # MF side
            try:
                if stocks_equity_by_isin is None:
                    pass  # unrelated helper; we compute MF series below
            except Exception:
                pass
            # Get MF series via fallback (we don't need stocks_equity_by_isin here)
            s_mf = _build_mf_series_fallback(mf_rows, holdings_df)
            s_mf = s_mf[s_mf > 0] if isinstance(s_mf, pd.Series) and not s_mf.empty else pd.Series(dtype=float)
            # Direct side
            s_dir = s_direct[s_direct > 0] if isinstance(s_direct, pd.Series) and not s_direct.empty else pd.Series(dtype=float)
            combined = s_mf.add(s_dir, fill_value=0.0)
            return combined[combined > 0] if not combined.empty else pd.Series(dtype=float)

        def _pies_from_series(s: pd.Series, equities_info: pd.DataFrame, label_col: str, small_thr: float = 0.03) -> pd.DataFrame:
            """Build donut data with columns [label,value,pct] using a label from equities_info."""
            if s is None or not isinstance(s, pd.Series) or s.empty:
                return pd.DataFrame(columns=["label","value","pct"])
            eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
            for c in ["isin", label_col]:
                if c not in eq.columns:
                    eq[c] = "" if c == "isin" else "Unknown"
            eq["isin"] = eq["isin"].astype(str)
            eq[label_col] = eq[label_col].astype(str).replace({"": "Unknown"})

            m = pd.DataFrame({"isin": s.index.astype(str), "value": s.values}).merge(
                eq[["isin", label_col]], on="isin", how="left"
            )
            m[label_col] = m[label_col].fillna("Unknown")
            total = float(m["value"].sum())
            g = (
                m.groupby(label_col, as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False, ignore_index=True)
            )
            g["pct"] = (g["value"] / total * 100.0) if total != 0 else 0.0
            g = g.rename(columns={label_col: "label"})
            g = _fold_small_slices(g, "label", "value", "pct", thr=small_thr)
            return g

        def _parse_amount(val) -> float:
            if val is None:
                return 0.0
            if isinstance(val, (int, float, np.number)):
                try:
                    return float(val)
                except Exception:
                    return 0.0
            s = re.sub(r"[^0-9\.\-]", "", str(val))
            try:
                return float(s) if s not in ("", "-", ".", "-.", ".-") else 0.0
            except Exception:
                return 0.0

        # ── content ────────────────────────────────────────────────────────────────
        st.caption("Breakdown of your combined portfolio (mutual funds + stocks)")

        # Raw inputs
        mf_rows    = st.session_state.get("mf_portfolio", []) or []
        stock_rows = st.session_state.get("stock_portfolio", []) or []

        # Direct series safely via alias
        if stocks_equity_by_isin is not None:
            try:
                s_direct = stocks_equity_by_isin(stock_rows, equities_info)  # Series[isin] = rupees
            except Exception:
                s_direct = pd.Series(dtype=float)
        else:
            s_direct = pd.Series(dtype=float)

        total_direct_investment = float(s_direct[s_direct > 0].sum()) if isinstance(s_direct, pd.Series) else 0.0
        total_mf_investment = sum(_parse_amount(r.get("amount")) for r in mf_rows if (r.get("fund_name") or "").strip())
        total_combined_investment = total_mf_investment + total_direct_investment

        if (total_combined_investment <= 0) or (not mf_rows and (s_direct is None or s_direct.empty)):
            st.info("Add mutual funds (Step 1) and/or direct stocks (Step 2) to see the combined view.")
        else:
            # 1) Combined Asset Allocation (Equity/Debt/Cash)
            alloc_df, _meta = mf_asset_allocation_by_fund_name(mf_rows, holdings_info)
            mf_equity = float(alloc_df.loc[alloc_df["asset_class"] == "Equity", "value"].sum()) if not alloc_df.empty else 0.0
            mf_debt   = float(alloc_df.loc[alloc_df["asset_class"] == "Debt",   "value"].sum()) if not alloc_df.empty else 0.0
            mf_cash   = float(alloc_df.loc[alloc_df["asset_class"] == "Cash",   "value"].sum()) if not alloc_df.empty else 0.0

            direct_equity = float(s_direct.sum()) if isinstance(s_direct, pd.Series) and not s_direct.empty else 0.0

            comb_equity = _z(mf_equity + direct_equity)
            comb_debt   = _z(mf_debt)
            comb_cash   = _z(mf_cash)

            chart_heading(
                "Combined Asset Allocation",
                "MF (scaled net buckets) + Direct stocks. If Equity is negative, view switches to a bar."
            )
            df_mix = pd.DataFrame([
                {"asset_class":"Equity","value": comb_equity},
                {"asset_class":"Debt",  "value": comb_debt},
                {"asset_class":"Cash & Others",  "value": comb_cash},
            ])
            denom = max(total_combined_investment, 1e-9)
            df_mix["pct"] = df_mix["value"] / denom * 100.0
            df_mix["Value (₹)"] = df_mix["value"].map(_fmt_inr)
            df_mix["% of Total"] = df_mix["pct"].map(lambda v: f"{float(v):.1f}%")

            display_as_bar = (comb_equity < -EPS) or (df_mix["value"] < -EPS).any()
            if display_as_bar:
                plot_df = df_mix.sort_values("value", ascending=True)
                st.plotly_chart(
                    hbar_with_text(plot_df["value"], plot_df["asset_class"], plot_df["% of Total"]),
                    use_container_width=True, config=PLOT_CFG, key="gamma"
                )
            else:
                st.plotly_chart(
                    donut(df_mix["asset_class"], df_mix["value"]),
                    use_container_width=True, config=PLOT_CFG, key="gamma_3"
                )
            with st.expander("Details", expanded=False):
                st.dataframe(df_mix[["asset_class","Value (₹)","% of Total"]], hide_index=True, use_container_width=True)

            st.markdown("---")

            # 2) Equity Source Split (MF vs Direct) — within Equity only
            mf_eq = _z(mf_equity)
            dir_eq = _z(direct_equity)
            chart_heading(
                "How are your equities divided between mutual funds and stocks?",
                "Share of your combined equity coming from Mutual Funds vs Direct stocks."
            )
            if mf_eq < -EPS:
                df_src = pd.DataFrame([
                    {"source":"Direct Equity","value": dir_eq},
                    {"source":"MF Equity","value": mf_eq},
                ])
                total_eq = max(abs(df_src["value"]).sum(), 1e-9)
                df_src["% of Equity"] = df_src["value"] / total_eq * 100.0
                st.plotly_chart(
                    hbar_with_text(df_src["value"], df_src["source"], df_src["% of Equity"].map(lambda v: f"{float(v):.1f}%")),
                    use_container_width=True, config=PLOT_CFG, key="gamma_4"
                )
            else:
                st.plotly_chart(donut(["Direct Equity","MF Equity"], [dir_eq, mf_eq]), use_container_width=True, config=PLOT_CFG, key="gamma_5")

            st.markdown("---")

            # 3 & 4) Combined pies (industry_rating + market_cap) from combined equity series
            comb_series = _combined_equity_series(mf_rows, holdings_info, s_direct)
            if comb_series is None or comb_series.empty:
                st.info("No positive combined equity exposure available to analyze.")
            else:
                c1, c2 = st.columns(2)

                with c1:
                    chart_heading(
                        "What sectors have I invested in?",
                        "Sectoral distribution of the combined portfolio"
                    )
                    sector_df = _pies_from_series(comb_series, equities_info, label_col="industry_rating", small_thr=0.03)
                    if sector_df.empty:
                        st.info("No industry data.")
                    else:
                        st.plotly_chart(donut(sector_df["label"], sector_df["value"]), use_container_width=True, config=PLOT_CFG, key="gamma_6")
                        with st.expander("Details", expanded=False):
                            show = sector_df.copy()
                            show["Value (₹)"] = show["value"].map(_fmt_inr)
                            show["% of Equity"] = show["pct"].map(lambda v: f"{float(v):.1f}%")
                            st.dataframe(
                                show[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Industry (rating)"}),
                                hide_index=True, use_container_width=True
                            )

                with c2:
                    chart_heading(
                        "What market-caps have I invested in?",
                        "Market-Cap distribution of the combined portfolio"
                    )
                    mcap_df = _pies_from_series(comb_series, equities_info, label_col="market_cap", small_thr=0.03)
                    if mcap_df.empty:
                        st.info("No market-cap data.")
                    else:
                        st.plotly_chart(donut(mcap_df["label"], mcap_df["value"]), use_container_width=True, config=PLOT_CFG, key="gamma_7")
                        with st.expander("Details", expanded=False):
                            show = mcap_df.copy()
                            show["Value (₹)"] = show["value"].map(_fmt_inr)
                            show["% of Equity"] = show["pct"].map(lambda v: f"{float(v):.1f}%")
                            st.dataframe(
                                show[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Market Cap"}),
                                hide_index=True, use_container_width=True
                            )

            st.markdown("---")

            # 5) Top-10 Combined Underlying Stock Holdings
            if comb_series is None or comb_series.empty:
                st.info("No positive combined equity exposure available to rank.")
            else:
                s_sorted = comb_series.sort_values(ascending=False).head(10)
                denom = max(total_combined_investment, 1e-9)  # % of TOTAL Combined Investment
                pct_of_investment = (s_sorted / denom * 100.0).astype(float)

                # Map ISIN -> company name
                eq = equities_info.copy() if equities_info is not None else pd.DataFrame()
                if "isin" not in eq.columns: eq["isin"] = ""
                if "company_name" not in eq.columns: eq["company_name"] = ""
                eq["isin"] = eq["isin"].astype(str)
                name_map = dict(eq[["isin","company_name"]].astype(str).values)

                df_top = pd.DataFrame({
                    "isin": s_sorted.index.astype(str),
                    "company_name": [name_map.get(isin, isin) for isin in s_sorted.index.astype(str)],
                    "value": s_sorted.values,
                    "pct_of_total_investment": pct_of_investment.values,
                })

                chart_heading(
                    "Top 10 Combined Underlying Stock Holdings",
                    "Labels show % of your total Combined Investment (MF invested + Direct invested)."
                )
                labels_pct = df_top["pct_of_total_investment"].map(lambda v: f"{float(v):.1f}%")
                st.plotly_chart(
                    hbar_with_text(df_top["value"], df_top["company_name"], labels_pct),
                    use_container_width=True, config=PLOT_CFG, key="gamma_8"
                )
                with st.expander("Details", expanded=False):
                    show = df_top.copy()
                    show["Value (₹)"] = show["value"].map(_fmt_inr)
                    show["% of Total Investment"] = show["pct_of_total_investment"].map(lambda v: f"{float(v):.1f}%")
                    st.dataframe(
                        show[["company_name","isin","Value (₹)","% of Total Investment"]]
                            .rename(columns={"company_name":"Company"}),
                        hide_index=True, use_container_width=True
                    )


    # ===========================================================
    # TAB 4: OVERLAP (placeholder for next steps)
    # ===========================================================
    '''
    with tabs[3]:
        from core.transforms import mf_overlap_matrix_by_pct_net_assets

        PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

        def chart_heading(title: str, subtitle: str = ""):
            st.markdown(
                f"""
                <div style="padding:6px 4px 2px 4px">
                <div style="font-weight:700; font-size:1.02rem; line-height:1.35">{title}</div>
                <div style="color:#A0A0A0; font-size:0.84rem; margin-top:2px">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("Fund-to-fund overlap to determine similarity between funds")

        mf_rows = st.session_state.get("mf_portfolio", []) or []
        if len([r for r in mf_rows if (r.get("fund_name") or "").strip()]) < 2:
            st.info("Add at least two mutual funds in Step 1 to see the overlap matrix.")
        else:
            out = mf_overlap_matrix_by_pct_net_assets(
                mf_rows,
                holdings_info,
                equities_info=equities_info,
                top_common_names=3,
            )

            M = out.get("overlap_pct", pd.DataFrame())
            C = out.get("common_count", pd.DataFrame())
            T = out.get("top_common", pd.DataFrame())

            if M is None or M.empty:
                st.info("No overlapping equity holdings found among the selected funds.")
            else:
                # Heatmap with inline % labels and rich hover
                chart_heading(
                    "How similar are the mutual funds you have invested in?",
                    "Mutual Fund Overlap matrix"
                )

                # Build text/hover matrices
                txt = M.applymap(lambda v: f"{float(v):.1f}%")
                hover = []
                rows = M.index.tolist()
                cols = M.columns.tolist()
                for i, rname in enumerate(rows):
                    row_ht = []
                    for j, cname in enumerate(cols):
                        ov = float(M.iat[i, j])
                        cnt = int(C.iat[i, j]) if not C.empty else 0
                        top = str(T.iat[i, j]) if not T.empty else ""
                        h = f"A: {rname}<br>B: {cname}<br>Overlap: {ov:.2f}%<br>Common names: {cnt}"
                        if top:
                            h += f"<br>Top: {top}"
                        row_ht.append(h)
                    hover.append(row_ht)

                fig = go.Figure(
                    data=go.Heatmap(
                        z=M.values,
                        x=M.columns.tolist(),
                        y=M.index.tolist(),
                        colorscale="Blues",
                        zmin=0, zmax=100,
                        text=txt.values,
                        texttemplate="%{text}",
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=np.array(hover),
                        colorbar=dict(title="%"),
                    )
                )
                fig.update_layout(
                    height=520,
                    margin=dict(l=16, r=16, t=16, b=16),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key="gamma_9")

                with st.expander("Pairwise details", expanded=False):
                    # Flatten to a long table for sorting/filtering
                    pairs = []
                    for i, fa in enumerate(rows):
                        for j, fb in enumerate(cols):
                            if i >= j:  # keep upper triangle only
                                continue
                            pairs.append({
                                "Fund A": fa,
                                "Fund B": fb,
                                "Overlap %": float(M.iat[i, j]),
                                "Common names (#)": int(C.iat[i, j]) if not C.empty else 0,
                                "Top common names": (str(T.iat[i, j]) if not T.empty else ""),
                            })
                    df_pairs = pd.DataFrame(pairs).sort_values(["Overlap %","Common names (#)"], ascending=[False, False])
                    df_pairs["Overlap %"] = df_pairs["Overlap %"].map(lambda v: f"{float(v):.2f}%")
                    st.dataframe(
                        df_pairs[["Fund A","Fund B","Overlap %","Common names (#)","Top common names"]],
                        hide_index=True,
                        use_container_width=True,
                    )
    '''

    with tabs[3]:
        from core.transforms import mf_overlap_matrix_by_pct_net_assets

        PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

        def chart_heading(title: str, subtitle: str = ""):
            st.markdown(
                f"""
                <div style="padding:6px 4px 2px 4px">
                <div style="font-weight:700; font-size:1.02rem; line-height:1.35">{title}</div>
                <div style="color:#A0A0A0; font-size:0.84rem; margin-top:2px">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Compact label generator: Fund A, Fund B, ... Fund Z, Fund AA, Fund AB, ...
        def _fund_labels(n: int):
            labels = []
            i = 0
            while len(labels) < n:
                q = i
                s = ""
                while True:
                    s = chr(65 + (q % 26)) + s
                    q = q // 26 - 1
                    if q < 0:
                        break
                labels.append(f"Fund {s}")
                i += 1
            return labels

        st.caption("Fund-to-fund overlap to determine similarity between funds")

        mf_rows = st.session_state.get("mf_portfolio", []) or []
        if len([r for r in mf_rows if (r.get("fund_name") or "").strip()]) < 2:
            st.info("Add at least two mutual funds in Step 1 to see the overlap matrix.")
        else:
            out = mf_overlap_matrix_by_pct_net_assets(
                mf_rows,
                holdings_info,
                equities_info=equities_info,
                top_common_names=3,
            )

            M = out.get("overlap_pct", pd.DataFrame())
            C = out.get("common_count", pd.DataFrame())
            T = out.get("top_common", pd.DataFrame())

            if M is None or M.empty:
                st.info("No overlapping equity holdings found among the selected funds.")
            else:
                # Build compact labels for axes (keep real names separately for details)
                rows_real = M.index.tolist()
                cols_real = M.columns.tolist()
                n_funds   = len(rows_real)
                labels    = _fund_labels(n_funds)
                real_to_label = {real: labels[i] for i, real in enumerate(rows_real)}
                # Axis labels become compact
                x_labels = [real_to_label[c] for c in cols_real]
                y_labels = [real_to_label[r] for r in rows_real]

                chart_heading(
                    "How similar are the mutual funds you have invested in?",
                    "Mutual Fund Overlap matrix"
                )

                # Build text/hover matrices (show compact labels in hover too)
                txt = M.applymap(lambda v: f"{float(v):.1f}%")
                hover = []
                for i, rname in enumerate(rows_real):
                    row_ht = []
                    for j, cname in enumerate(cols_real):
                        ov  = float(M.iat[i, j])
                        cnt = int(C.iat[i, j]) if not C.empty else 0
                        top = str(T.iat[i, j]) if not T.empty else ""
                        h = f"A: {real_to_label[rname]}<br>B: {real_to_label[cname]}<br>Overlap: {ov:.2f}%<br>Common names: {cnt}"
                        if top:
                            h += f"<br>Top: {top}"
                        row_ht.append(h)
                    hover.append(row_ht)

                fig = go.Figure(
                    data=go.Heatmap(
                        z=M.values,
                        x=x_labels,               # compact labels on X
                        y=y_labels,               # compact labels on Y
                        colorscale="Blues",
                        zmin=0, zmax=100,
                        text=txt.values,
                        texttemplate="%{text}",
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=np.array(hover),
                        colorbar=dict(title="%"),
                    )
                )
                fig.update_layout(
                    height=520,
                    margin=dict(l=16, r=16, t=16, b=16),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG, key="gamma_9")

                # NEW: Legend (dropdown) mapping compact labels -> real fund names
                with st.expander("Fund label legend (show real names)", expanded=False):
                    map_df = pd.DataFrame({
                        "Label": [real_to_label[name] for name in rows_real],
                        "Fund Name": rows_real,
                    })
                    st.dataframe(map_df, hide_index=True, use_container_width=True)

                # Pairwise details (UNCHANGED: still uses real names)
                with st.expander("Pairwise details", expanded=False):
                    # Flatten to a long table for sorting/filtering
                    pairs = []
                    for i, fa in enumerate(rows_real):
                        for j, fb in enumerate(cols_real):
                            if i >= j:  # keep upper triangle only
                                continue
                            pairs.append({
                                "Fund A": fa,
                                "Fund B": fb,
                                "Overlap %": float(M.iat[i, j]),
                                "Common names (#)": int(C.iat[i, j]) if not C.empty else 0,
                                "Top common names": (str(T.iat[i, j]) if not T.empty else ""),
                            })
                    df_pairs = pd.DataFrame(pairs).sort_values(["Overlap %","Common names (#)"], ascending=[False, False])
                    df_pairs["Overlap %"] = df_pairs["Overlap %"].map(lambda v: f"{float(v):.2f}%")
                    st.dataframe(
                        df_pairs[["Fund A","Fund B","Overlap %","Common names (#)","Top common names"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                             
    # ---------- Nav row (Back left, Reset right — like Stocks page) ----------
    st.markdown('<div class="skw-nav">', unsafe_allow_html=True)
    left, spacer, right = st.columns([1, 6, 1])

    with left:
        if st.button("← Back", key="an_back"):
            st.session_state.page_idx = 2  # back to Stocks step
            st.rerun()

    with right:
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("Reset portfolio", key="an_reset", help="Clear all MF & Stock entries"):
            st.session_state.mf_portfolio = []
            st.session_state.stock_portfolio = []
            st.toast("Portfolio reset.", icon="🧹")
            st.session_state.page_idx = 1  # to Mutual Funds
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Sticky footer ----------
    sticky_footer("Mutual fund investments are subject to market risks. Read all scheme related documents.")
