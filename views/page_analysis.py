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
    '''
    with tabs[0]:
        st.caption("Underlying distribution, sector & market-cap splits, top holdings and fund track record.")

        mf_rows = st.session_state.get("mf_portfolio", [])
        if not mf_rows:
            st.info("Add mutual funds in Step 1 to see your personalized breakdown.")
        elif holdings_info is None or holdings_info.empty:
            st.info("No holdings data available. Ensure holdings_info is loaded.")
        else:
            # ---------- Mini chart helpers ----------
            def donut(labels, values, title, subtitle, height=320):
                fig = go.Figure(
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.45,
                        textinfo="percent",  # keep inner labels light
                        hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=height,
                    margin=dict(l=10, r=10, t=64, b=10),
                    showlegend=True,
                    legend=dict(orientation="v", x=1.02, y=0.5),
                    title={
                        "text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                        "y": 0.98, "x": 0.02, "xanchor": "left", "yanchor": "top",
                    },
                )
                return fig

            def hbar(x, y, title, subtitle, height=300):
                total = max(float(x.sum()), 1e-9)
                fig = go.Figure()
                fig.add_bar(
                    x=x, y=y, orientation="h",
                    text=[f"{(v/total*100):.1f}%" for v in x],
                    textposition="auto",
                    hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
                )
                fig.update_layout(
                    height=height,
                    margin=dict(l=10, r=10, t=64, b=10),
                    xaxis_title="Rupees", yaxis_title="",
                    title={
                        "text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                        "y": 0.98, "x": 0.02, "xanchor": "left", "yanchor": "top",
                    },
                )
                return fig

            # ---------- 1) Underlying Asset Distribution ----------
            alloc_df, meta = mf_asset_allocation_by_fund_name(mf_rows, holdings_info)
            total_mf = float(meta.get("total_mf", 0.0))
            show_bar = bool(meta.get("display_as_bar", False))

            if total_mf <= 0 or alloc_df["value"].abs().sum() == 0:
                st.info("No calculable exposure yet for the selected funds.")
            else:
                if show_bar:
                    order = ["Equity", "Debt", "Cash"]
                    plot_df = alloc_df.set_index("asset_class").loc[order].reset_index()
                    st.plotly_chart(
                        hbar(
                            x=plot_df["value"],
                            y=plot_df["asset_class"],
                            title="Your Investment’s Underlying Distribution",
                            subtitle="Scaled to your investments; derivatives are netted into Equity/Cash per AMC rules.",
                        ),
                        use_container_width=True,
                    )
                else:
                    st.plotly_chart(
                        donut(
                            labels=alloc_df["asset_class"],
                            values=alloc_df["value"],
                            title="Your Investment’s Underlying Distribution",
                            subtitle="Scaled to your investments; derivatives are netted into Equity/Cash per AMC rules.",
                        ),
                        use_container_width=True,
                    )

                with st.expander("Details", expanded=False):
                    st.dataframe(
                        alloc_df.assign(**{"Value (₹)": alloc_df["value"].map(_fmt_inr)})[
                            ["asset_class", "Value (₹)", "pct"]
                        ].rename(columns={"asset_class": "Asset Class", "pct": "% of MF"}),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("---")

            # ---------- 2) Sector & Market-Cap Distribution (net equity only) ----------
            sm_out = mf_sector_and_mcap_exposure_by_fund_name(
                mf_portfolio=mf_rows,
                holdings_df=holdings_info,
                equities_info=equities_info,
                small_slice_threshold=0.03,
            )
            sector_df = sm_out["sector"]
            mcap_df   = sm_out["market_cap"]
            diags     = sm_out["diagnostics"]

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    donut(
                        labels=sector_df["label"],
                        values=sector_df["value"],
                        title="Sector Distribution",
                        subtitle="Net equity (positives only); small slices grouped as Other.",
                    ),
                    use_container_width=True,
                ) if not sector_df.empty else st.info("No positive net equity exposure for sectors.")

            with c2:
                st.plotly_chart(
                    donut(
                        labels=mcap_df["label"],
                        values=mcap_df["value"],
                        title="Market-Cap Distribution",
                        subtitle="Net equity (positives only); small slices grouped as Other.",
                    ),
                    use_container_width=True,
                ) if not mcap_df.empty else st.info("No positive net equity exposure for market-cap.")

            if diags.get("negatives_total", 0) < 0:
                st.caption("Note: shorts are excluded from the pies and shown only in diagnostics.")

            st.markdown("---")

            # ---------- 3) Top 10 Underlying Net Stock Holdings ----------
            top_out = mf_top_net_holdings_by_fund_name(
                mf_portfolio=mf_rows,
                holdings_df=holdings_info,
                equities_info=equities_info,
                top_n=10,
            )
            df_top = top_out["top_holdings"]
            if df_top.empty:
                st.info("No positive net equity exposure available to rank.")
            else:
                st.plotly_chart(
                    hbar(
                        x=df_top["value"],
                        y=df_top["company_name"],
                        title="Top 10 Underlying Net Stock Holdings",
                        subtitle="Net equity (positives only); hover for rupee values and fund count.",
                        height=360,
                    ),
                    use_container_width=True,
                )
                with st.expander("Details", expanded=False):
                    show_df = df_top.assign(**{"Value (₹)": df_top["value"].map(_fmt_inr)})[
                        ["company_name", "isin", "Value (₹)", "pct_of_equity", "fund_count"]
                    ].rename(columns={"company_name": "Company", "pct_of_equity": "% of Equity", "fund_count": "Funds"})
                    st.dataframe(show_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Fund Track Record")
            st.caption("Weights reflect your amounts; returns are CAGR from the API as of the latest NAV.")

            # 1) Build selected funds table (fund_name, amc, amount, weight)
            sel_rows = []
            for r in (mf_rows or []):
                fn = (r.get("fund_name", "") or "").strip()
                amt = float(r.get("amount", 0) or 0)
                if fn and amt > 0:
                    sel_rows.append({"fund_name": fn, "amc": r.get("amc_name", ""), "amount": amt})

            sel_df = pd.DataFrame(sel_rows)
            if sel_df.empty:
                st.info("No funds selected for track record.")
            else:
                sel_df["fn_norm"] = sel_df["fund_name"].astype(str).str.strip().str.lower()

                # 2) Prepare fund_info safely (no boolean DF usage)
                fi = fund_info.copy() if fund_info is not None else pd.DataFrame()
                if "fund_name" not in fi.columns: fi["fund_name"] = ""
                if "fund_code" not in fi.columns: fi["fund_code"] = None
                if "amc" not in fi.columns: fi["amc"] = ""
                if "category" not in fi.columns: fi["category"] = ""
                fi["fn_norm"] = fi["fund_name"].astype(str).str.strip().str.lower()

                # 3) Merge to get fund_code (and canonical AMC/category if present)
                m = sel_df.merge(
                    fi[["fn_norm", "fund_code", "amc", "category"]],
                    on="fn_norm", how="left"
                )
                total = float(m["amount"].sum())
                m["weight"] = 0.0 if total == 0 else m["amount"] / total

                # 4) Fetch API performance (cached) per fund
                rows = []
                for _, r in m.iterrows():
                    code = r.get("fund_code", None)
                    perf = get_fund_performance(int(code)) if pd.notna(code) else None
                    rets = (perf or {}).get("returns", {}) or {}
                    rows.append({
                        "AMC": r.get("amc") or r.get("amc_x") or "",
                        "Fund": r.get("fund_name", ""),
                        "Category": r.get("category", ""),
                        "Amount (₹)": _fmt_inr(r.get("amount", 0.0)),
                        "Weight": f"{float(r.get('weight', 0.0))*100:.1f}%",
                        "NAV": f"{(perf or {}).get('nav', float('nan')):.2f}" if perf else "—",
                        "1M":  f"{rets.get('1M', float('nan'))*100:.1f}%"  if pd.notna(rets.get('1M',  float('nan'))) else "—",
                        "3M":  f"{rets.get('3M', float('nan'))*100:.1f}%"  if pd.notna(rets.get('3M',  float('nan'))) else "—",
                        "6M":  f"{rets.get('6M', float('nan'))*100:.1f}%"  if pd.notna(rets.get('6M',  float('nan'))) else "—",
                        "1Y":  f"{rets.get('1Y', float('nan'))*100:.1f}%"  if pd.notna(rets.get('1Y',  float('nan'))) else "—",
                        "3Y":  f"{rets.get('3Y', float('nan'))*100:.1f}%"  if pd.notna(rets.get('3Y',  float('nan'))) else "—",
                        "5Y":  f"{rets.get('5Y', float('nan'))*100:.1f}%"  if pd.notna(rets.get('5Y',  float('nan'))) else "—",
                        "10Y": f"{rets.get('10Y',float('nan'))*100:.1f}%"  if pd.notna(rets.get('10Y', float('nan'))) else "—",
                        "SI":  f"{rets.get('SI', float('nan'))*100:.1f}%"  if pd.notna(rets.get('SI',  float('nan'))) else "—",
                        "As of": (perf or {}).get("as_of", "—"),
                        "_w": float(r.get("weight", 0.0)),
                        "_rets": rets,
                    })

                perf_df = pd.DataFrame(rows)

                # 5) Weighted KPIs (only across horizons that exist)
                def wavg(key: str):
                    if perf_df.empty: return None
                    acc = 0.0; seen = False
                    for _, rr in perf_df.iterrows():
                        v = (rr.get("_rets") or {}).get(key, float("nan"))
                        if pd.notna(v):
                            acc += float(v) * float(rr.get("_w", 0.0))
                            seen = True
                    return acc if seen else None

                summary = {k: wavg(k) for k in ["1Y", "3Y", "5Y", "10Y", "SI"]}

                # 6) Display table (clean columns)
                show_cols = ["AMC","Fund","Category","Amount (₹)","Weight","NAV","1M","3M","6M","1Y","3Y","5Y","10Y","SI","As of"]
                st.dataframe(perf_df[show_cols], hide_index=True, use_container_width=True)

                k1, k2, k3, k4, k5 = st.columns(5)
                for col, key in zip([k1, k2, k3, k4, k5], ["1Y", "3Y", "5Y", "10Y", "SI"]):
                    with col:
                        val = summary.get(key, None)
                        st.metric(
                            label=f"Weighted {key} CAGR",
                            value=(f"{val*100:.1f}%" if isinstance(val, float) and pd.notna(val) else "—"),
                        )
    

    with tabs[0]:
        st.caption("Deep dive into the underlying composition of your mutual fund investments.")

        mf_rows = st.session_state.get("mf_portfolio", [])
        if not mf_rows:
            st.info("Add mutual funds in Step 1 to see your personalized breakdown.")
        elif holdings_info is None or holdings_info.empty:
            st.info("No holdings data available. Ensure holdings_info is loaded.")
        else:
            EPS = 1e-8

            def _z(v: float) -> float:
                """Clamp tiny magnitudes to 0.0 to avoid '-0.0' in display text."""
                return 0.0 if abs(float(v)) < EPS else float(v)

            def donut(labels, values, title, subtitle, height=360):
                vals = [ _z(v) for v in values ]
                fig = go.Figure(
                    go.Pie(
                        labels=labels, values=vals, hole=0.45,
                        textinfo="percent",
                        hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=height,
                    margin=dict(l=16, r=16, t=40, b=16),  # ⬅️ more top padding
                    showlegend=True,
                    legend=dict(orientation="v", x=1.02, y=0.5),
                    title={
                        "text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                        "y": 0.995, "x": 0.02, "xanchor": "left", "yanchor": "top",
                    },
                )
                return fig

            def hbar(x, y, title, subtitle, height=360):
                xx = [ _z(v) for v in x ]
                total = max(float(np.array(xx).sum()), 1e-9)
                fig = go.Figure()
                fig.add_bar(
                    x=xx, y=y, orientation="h",
                    text=[f"{(float(v)/total*100):.1f}%" for v in xx], textposition="auto",
                    hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>",
                )
                fig.update_layout(
                    height=height,
                    margin=dict(l=16, r=16, t=80, b=16),  # ⬅️ more top padding
                    xaxis_title="Rupees", yaxis_title="",
                    title={
                        "text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                        "y": 0.995, "x": 0.02, "xanchor": "left", "yanchor": "top",
                    },
                )
                return fig


            # 1) Asset mix
            alloc_df, meta = mf_asset_allocation_by_fund_name(mf_rows, holdings_info)
            total_mf = float(meta.get("total_mf", 0.0))
            df_show = alloc_df.copy()
            df_show["value"] = df_show["value"].apply(_z)  # ⬅️ kills “-0.0”
            df_show["% of MF"] = df_show["pct"].map(lambda v: f"{float(v):.1f}%")
            df_show["Value (₹)"] = df_show["value"].map(lambda v: f"₹{float(v):,.0f}")

            if meta.get("display_as_bar", False):
                plot_df = df_show.sort_values("value", ascending=True)
                st.plotly_chart(
                    hbar(plot_df["value"], plot_df["asset_class"], "Your Investment’s Underlying Distribution",
                        "Net Equity = Equity + Derivatives | Net Cash rule: Axis/Motilal = Cash; others = Cash − Derivatives."),
                    use_container_width=True
                )
            else:
                st.plotly_chart(
                    donut(df_show["asset_class"], df_show["value"], "Your Investment’s Underlying Distribution",
                        "Net Equity = Equity + Derivatives | Net Cash rule: Axis/Motilal = Cash; others = Cash − Derivatives."),
                    use_container_width=True
                )
            with st.expander("Details", expanded=False):
                st.dataframe(df_show[["asset_class","Value (₹)","% of MF"]], hide_index=True, use_container_width=True)

            st.markdown("---")

            # 2) Sector pie (industry_rating)
            sm = mf_sector_and_mcap_exposure_by_fund_name(mf_rows, holdings_info, equities_info, small_slice_threshold=0.03)
            sector_df = sm["sector"]
            if sector_df.empty:
                st.info("No positive net equity exposure available for sector distribution.")
            else:
                sd = sector_df.copy()
                sd["% of Equity"] = sd["pct"].map(lambda v: f"{float(v):.1f}%")
                sd["Value (₹)"] = sd["value"].map(lambda v: f"₹{float(v):,.0f}")
                st.plotly_chart(
                    donut(sd["label"], sd["value"], "Underlying Net Sector Distribution",
                        "User-weighted net equity (positives only). Small slices grouped as Other."),
                    use_container_width=True
                )
                with st.expander("Details", expanded=False):
                    st.dataframe(sd[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Sector"}),
                                hide_index=True, use_container_width=True)

            st.markdown("---")

            # 3) Market-cap pie
            mcap_df = sm["market_cap"]
            if mcap_df.empty:
                st.info("No positive net equity exposure available for market-cap distribution.")
            else:
                mc = mcap_df.copy()
                mc["% of Equity"] = mc["pct"].map(lambda v: f"{float(v):.1f}%")
                mc["Value (₹)"] = mc["value"].map(lambda v: f"₹{float(v):,.0f}")
                st.plotly_chart(
                    donut(mc["label"], mc["value"], "Underlying Net Market-Cap Distribution",
                        "User-weighted net equity (positives only). Small slices grouped as Other."),
                    use_container_width=True
                )
                with st.expander("Details", expanded=False):
                    st.dataframe(mc[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Market Cap"}),
                                hide_index=True, use_container_width=True)

            st.markdown("---")

            # 4) Top 10 underlying net stocks
            df_top = mf_top_net_holdings_by_fund_name(mf_rows, holdings_info, equities_info, top_n=10)
            if df_top.empty:
                st.info("No positive net equity exposure available to rank.")
            else:
                tt = df_top.copy()
                tt["% of Equity"] = tt["pct_of_equity"].map(lambda v: f"{float(v):.1f}%")
                st.plotly_chart(
                    hbar(tt["value"], tt["company_name"], "Top 10 Underlying Net Stock Holdings",
                        "User-weighted net equity (positives only). Hover for rupee values; labels show % of equity."),
                    use_container_width=True
                )
                with st.expander("Details", expanded=False):
                    show = tt.assign(**{"Value (₹)": tt["value"].map(lambda v: f"₹{float(v):,.0f}")})[
                        ["company_name","isin","Value (₹)","% of Equity"]
                    ].rename(columns={"company_name":"Company"})
                    st.dataframe(show, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Fund Track Record")

            # Build a clean list of selected funds with codes & weights
            mf_rows = st.session_state.get("mf_portfolio", []) or []
            if not mf_rows or fund_info is None or fund_info.empty:
                st.info("No funds to show. Add funds in Step 1.")
            else:
                sel = [r for r in mf_rows if float(r.get("amount", 0) or 0) > 0 and (r.get("fund_name") or "").strip()]
                if not sel:
                    st.info("No funds to show. Add funds in Step 1.")
                else:
                    total_amt = sum(float(r["amount"]) for r in sel)
                    fi = fund_info.copy()
                    for c in ["fund_name","fund_code"]:
                        if c not in fi.columns: fi[c] = None
                    fi["fund_name_norm"] = fi["fund_name"].astype(str).str.strip().str.lower()

                    rows = []
                    for r in sel:
                        fn = (r["fund_name"] or "").strip()
                        amt = float(r["amount"])
                        code = None
                        match = fi.loc[fi["fund_name_norm"] == fn.lower()]
                        if not match.empty and pd.notna(match.iloc[0].get("fund_code")):
                            code = match.iloc[0]["fund_code"]

                        perf = None
                        try:
                            if code not in (None, "", 0):
                                perf = get_fund_performance(code)
                        except Exception:
                            perf = None

                        row = {
                            "Fund": fn,
                            "Amount (₹)": amt,
                            "Weight %": (amt / total_amt * 100.0) if total_amt > 0 else 0.0,
                            "NAV": None,
                            "As of": None,
                            "1M": None, "3M": None, "6M": None,
                            "1Y": None, "3Y": None, "5Y": None, "10Y": None, "SI": None,
                        }
                        if perf:
                            row["NAV"] = float(perf.get("nav", None))
                            row["As of"] = perf.get("as_of", None)
                            rets = perf.get("returns", {}) or {}
                            for k in ["1M","3M","6M","1Y","3Y","5Y","10Y","SI"]:
                                v = rets.get(k, None)
                                row[k] = float(v) if v is not None else None
                        rows.append(row)

                    df_tr = pd.DataFrame(rows)

                    # Weighted composite for horizons (only where data exists)
                    weights = (df_tr["Amount (₹)"] / df_tr["Amount (₹)"].sum()) if df_tr["Amount (₹)"].sum() > 0 else 0.0
                    summary = {"Fund": "Weighted (by amount)", "Amount (₹)": df_tr["Amount (₹)"].sum(), "Weight %": 100.0, "NAV": None, "As of": None}
                    for k in ["1M","3M","6M","1Y","3Y","5Y","10Y","SI"]:
                        if k in df_tr:
                            valid = df_tr[k].where(df_tr[k].notna(), None)
                            if valid is not None and valid.notna().any():
                                summary[k] = float((df_tr[k].fillna(0.0) * weights).sum())
                            else:
                                summary[k] = None

                    # Display formatting
                    show = df_tr.copy()
                    show["Amount (₹)"] = show["Amount (₹)"].map(lambda v: f"₹{_z(v):,.0f}")
                    show["Weight %"] = show["Weight %"].map(lambda v: f"{float(v):.1f}%")
                    if "NAV" in show:
                        show["NAV"] = show["NAV"].map(lambda v: "" if pd.isna(v) else f"{float(v):,.2f}")
                    for k in ["1M","3M","6M","1Y","3Y","5Y","10Y","SI"]:
                        if k in show:
                            show[k] = show[k].map(lambda v: "" if pd.isna(v) else f"{float(v)*100:.1f}%")

                    st.dataframe(
                        show[["Fund","Amount (₹)","Weight %","NAV","As of","1M","3M","6M","1Y","3Y","5Y","10Y","SI"]],
                        hide_index=True, use_container_width=True
                    )

                    # Weighted summary (optional row beneath)
                    with st.expander("Weighted summary (CAGR)"):
                        sum_df = pd.DataFrame([summary])
                        for k in ["1M","3M","6M","1Y","3Y","5Y","10Y","SI"]:
                            if k in sum_df:
                                sum_df[k] = sum_df[k].map(lambda v: "" if v is None else f"{float(v)*100:.1f}%")
                        sum_df["Amount (₹)"] = sum_df["Amount (₹)"].map(lambda v: f"₹{_z(v):,.0f}")
                        st.dataframe(sum_df[["Fund","Amount (₹)","Weight %","1M","3M","6M","1Y","3Y","5Y","10Y","SI"]],
                                    hide_index=True, use_container_width=True)
    '''

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
                st.plotly_chart(hbar(plot_df["value"], plot_df["asset_class"]), use_container_width=True, config=PLOT_CFG)
            else:
                st.plotly_chart(donut(df_show["asset_class"], df_show["value"]), use_container_width=True, config=PLOT_CFG)

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
                st.plotly_chart(donut(sd["label"], sd["value"]), use_container_width=True, config=PLOT_CFG)
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
                st.plotly_chart(donut(mc["label"], mc["value"]), use_container_width=True, config=PLOT_CFG)
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
                    "Fund": "Weighted (by amount)",
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
                        st.plotly_chart(donut(sd["label"], sd["value"]), use_container_width=True, config=PLOT_CFG)
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
                        st.plotly_chart(donut(mc["label"], mc["value"]), use_container_width=True, config=PLOT_CFG)
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
    # TAB 3: COMBINED (placeholder for next steps)
    # ===========================================================
# ===========================================================
# TAB 3: COMBINED (MF Net Equity + Direct)
# ===========================================================
    '''
    with tabs[2]:
        st.caption("Unified view: your MF net equity added to direct equity.")

        # helpers (reuse if already defined once)
        def donut(labels, values, title, subtitle, height=320):
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45, textinfo="percent",
                                hovertemplate="%{label}: ₹%{value:,.0f} (%{percent})<extra></extra>"))
            fig.update_layout(height=height, margin=dict(l=10, r=10, t=64, b=10),
                            showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5),
                            title={"text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                                    "y":0.98,"x":0.02,"xanchor":"left","yanchor":"top"})
            return fig
        def hbar(x, y, title, subtitle, height=300):
            total = max(float(x.sum()), 1e-9)
            fig = go.Figure()
            fig.add_bar(x=x, y=y, orientation="h",
                        text=[f"{(v/total*100):.1f}%" for v in x], textposition="auto",
                        hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<extra></extra>")
            fig.update_layout(height=height, margin=dict(l=10, r=10, t=64, b=10),
                            xaxis_title="Rupees", yaxis_title="",
                            title={"text": f"<b>{title}</b><br><span style='font-size:12px;color:#A0A0A0'>{subtitle}</span>",
                                    "y":0.98,"x":0.02,"xanchor":"left","yanchor":"top"})
            return fig

        mf_rows = st.session_state.get("mf_portfolio", [])
        s_mf = mf_net_equity_series_by_fund_name(mf_rows, holdings_info)  # can include negatives
        s_direct = stocks_equity_by_isin(st.session_state.get("stock_portfolio", []), equities_info)

        # Split (positives only)
        split_df = combined_composition_split(s_mf, s_direct)
        total_pos = float(split_df["value"].sum())
        if total_pos <= 0:
            st.info("No positive equity exposures to display.")
        else:
            st.plotly_chart(
                donut(split_df["label"], split_df["value"],
                    "Portfolio Composition", "Positive exposures only: MF net equity vs direct equity."),
                use_container_width=True
            )

            # Combined series for sector/mcap and top-10
            s_combined = combined_equity_by_isin(s_mf, s_direct)
            # pies
            pies = sector_mcap_from_series(s_combined[s_combined > 0], equities_info, 0.03)
            c1, c2 = st.columns(2)
            with c1:
                sd = pies["sector"]
                st.plotly_chart(donut(sd["label"], sd["value"], "Combined Sector Distribution",
                                    "MF net + Direct; positives only. Small slices grouped as Other."),
                                use_container_width=True) if not sd.empty else st.info("No sector data.")
            with c2:
                mc = pies["market_cap"]
                st.plotly_chart(donut(mc["label"], mc["value"], "Combined Market-Cap Distribution",
                                    "MF net + Direct; positives only. Small slices grouped as Other."),
                                use_container_width=True) if not mc.empty else st.info("No market-cap data.")

            # Top 10 combined with MF/Direct breakdown in hover
            top = top_holdings_from_series(s_combined, equities_info, top_n=10)
            if top.empty:
                st.info("No top holdings to display.")
            else:
                # Build custom hover with source mix
                mf_vals = [float(max(s_mf.get(isin, 0.0), 0.0)) for isin in top["isin"]]
                dx_vals = [float(max(s_direct.get(isin, 0.0), 0.0)) for isin in top["isin"]]
                hover = [f"MF: ₹{mf_vals[i]:,.0f} | Direct: ₹{dx_vals[i]:,.0f}" for i in range(len(mf_vals))]
                fig = go.Figure()
                fig.add_bar(
                    x=top["value"], y=top["company_name"], orientation="h",
                    text=[f"{p:.1f}%" for p in top["pct_of_equity"]],
                    textposition="auto",
                    hovertemplate="%{y}: ₹%{x:,.0f} (%{text})<br>%{customdata}<extra></extra>",
                    customdata=hover,
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=64, b=10),
                                xaxis_title="Rupees", yaxis_title="",
                                title={"text":"<b>Top 10 Combined Holdings</b><br><span style='font-size:12px;color:#A0A0A0'>MF net + Direct; hover shows source mix.</span>",
                                        "y":0.98,"x":0.02,"xanchor":"left","yanchor":"top"})
                st.plotly_chart(fig, use_container_width=True)

            # Overlap with your direct stocks (quick table)
            inter = set((s_direct or pd.Series(dtype=float)).index) & set((s_mf or pd.Series(dtype=float)).index)
            if inter:
                rows = []
                for isin in inter:
                    d = float(max(s_direct.get(isin, 0.0), 0.0))
                    m = float(max(s_mf.get(isin, 0.0), 0.0))
                    tot = d + m
                    rows.append({"isin": isin, "company_name": _map_isin_to_attrs(pd.Series([isin],[isin]), equities_info)["company_name"].iloc[0]
                                if isin in (s_direct.index.union(s_mf.index)) else isin,
                                "Direct (₹)": d, "MF net (₹)": m, "Total (₹)": tot, "% Duplicate": (m / tot * 100.0 if tot > 0 else 0.0)})
                dup_df = pd.DataFrame(rows).sort_values("% Duplicate", ascending=False)
                dup_df[["Direct (₹)","MF net (₹)","Total (₹)"]] = dup_df[["Direct (₹)","MF net (₹)","Total (₹)"]].applymap(lambda v: f"₹{v:,.0f}")
                dup_df["% Duplicate"] = dup_df["% Duplicate"].map(lambda v: f"{v:.1f}%")
                with st.expander("Overlap with your Direct Stocks", expanded=False):
                    st.dataframe(dup_df[["company_name","isin","Direct (₹)","MF net (₹)","Total (₹)","% Duplicate"]],
                                hide_index=True, use_container_width=True)
    '''

# --- COMBINED PORTFOLIO TAB ---------------------------------------------------
    '''
    with tabs[2]:
        import re

        try:
            # If available, use the core helper
            from core.transforms import _mf_user_scaled_net_equity_series_by_isin as _mf_series_by_isin  # type: ignore
        except Exception:
            _mf_series_by_isin = None  # we'll fallback below

        try:
            from core.transforms import stocks_equity_by_isin as stocks_equity_by_isin
        except Exception:
            stocks_equity_by_isin = None
        # ── helpers (same look/feel as other tabs) ─────────────────────────────────
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
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5),
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
                height=height,
                margin=dict(l=16, r=16, t=16, b=16),
                xaxis_title="Rupees", yaxis_title="",
            )
            return fig

        def _fold_small_slices(df: pd.DataFrame, label_col: str, value_col: str, pct_col: str, thr: float = 0.03) -> pd.DataFrame:
            if df.empty:
                return df
            # thr is fraction (e.g., 0.03 -> 3%)
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
            # renormalize pct to 100
            ps = float(out[pct_col].sum())
            if ps not in (0.0, 100.0):
                out[pct_col] = out[pct_col] * (100.0 / ps)
            return out

        def _is_axis_or_motilal(amc: str) -> bool:
            s = (str(amc) or "").strip().lower()
            return ("axis" in s) or ("motilal oswal" in s)

        def _build_mf_series_fallback(mf_rows: list[dict], holdings_df: pd.DataFrame) -> pd.Series:
            """If core helper isn't available, compute user-scaled net equity per ISIN here."""
            if holdings_df is None or holdings_df.empty or not mf_rows:
                return pd.Series(dtype=float)
            df = holdings_df.copy()
            for c in ["fund_name","category","isin","company_name","market_value","amc"]:
                if c not in df.columns:
                    df[c] = "" if c not in ("market_value",) else 0.0
            df["fund_name_norm"] = df["fund_name"].astype(str).str.strip().str.lower()
            df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

            result: dict[str, float] = {}
            for r in mf_rows:
                fn = (r.get("fund_name") or "").strip()
                amt = float(r.get("amount", 0) or 0)
                amc = r.get("amc_name","")
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

                # per-ISIN net equity (equity + matched derivatives), then scale
                eq = sub[sub["category"] == "Equity"]
                der = sub[sub["category"] == "Derivatives"]
                der_map = der.groupby("isin")["market_value"].sum().to_dict()
                for _, row in eq.iterrows():
                    isin = str(row.get("isin",""))
                    base = float(row.get("market_value", 0.0))
                    adj  = base + float(der_map.get(isin, 0.0))
                    if adj <= 0:
                        continue  # pies/top-10 use positives only
                    result[isin] = result.get(isin, 0.0) + (adj * scale)

            return pd.Series(result, dtype=float)

        def _combined_equity_series(mf_rows: list[dict], holdings_df: pd.DataFrame, s_direct: pd.Series) -> pd.Series:
            """MF net (scaled) + Direct; positives only for pies/top-10."""
            s_mf = pd.Series(dtype=float)
            if _mf_series_by_isin:
                try:
                    s_mf = _mf_series_by_isin(mf_rows, holdings_df)
                except Exception:
                    s_mf = _build_mf_series_fallback(mf_rows, holdings_df)
            else:
                s_mf = _build_mf_series_fallback(mf_rows, holdings_df)
            s_mf = s_mf[s_mf > 0] if isinstance(s_mf, pd.Series) and not s_mf.empty else pd.Series(dtype=float)
            s_dir = s_direct[s_direct > 0] if isinstance(s_direct, pd.Series) and not s_direct.empty else pd.Series(dtype=float)
            # align and sum
            combined = s_mf.add(s_dir, fill_value=0.0)
            return combined[combined > 0] if not combined.empty else pd.Series(dtype=float)

        def _pies_from_series(s: pd.Series, equities_info: pd.DataFrame, label_col: str, small_thr: float = 0.03) -> pd.DataFrame:
            """Build a donut data df with columns [label,value,pct] using a label from equities_info."""
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
        st.caption("Unified view: Mutual Funds + Direct Stocks.")

        mf_rows = st.session_state.get("mf_portfolio", []) or []
        stock_rows = st.session_state.get("stock_portfolio", []) or []

        # Direct series from your existing helper (already computed earlier tab)
        try:
            # Reuse same helper you used in Direct tab
            from core.transforms import stocks_equity_by_isin
            s_direct = stocks_equity_by_isin(stock_rows, equities_info)
        except Exception:
            s_direct = pd.Series(dtype=float)

        total_direct_investment = float(s_direct[s_direct > 0].sum()) if isinstance(s_direct, pd.Series) else 0.0
        total_mf_investment = sum(_parse_amount(r.get("amount")) for r in mf_rows if (r.get("fund_name") or "").strip())
        total_combined_investment = total_mf_investment + total_direct_investment

        if (total_combined_investment <= 0) or (not mf_rows and (s_direct is None or s_direct.empty)):
            st.info("Add mutual funds (Step 1) and/or direct stocks (Step 2) to see the combined view.")
        else:
            # 1) Combined Asset Allocation (Equity/Debt/Cash)
            alloc_df, meta = mf_asset_allocation_by_fund_name(mf_rows, holdings_info)
            mf_equity = float(alloc_df.loc[alloc_df["asset_class"] == "Equity", "value"].sum()) if not alloc_df.empty else 0.0
            mf_debt   = float(alloc_df.loc[alloc_df["asset_class"] == "Debt",   "value"].sum()) if not alloc_df.empty else 0.0
            mf_cash   = float(alloc_df.loc[alloc_df["asset_class"] == "Cash",   "value"].sum()) if not alloc_df.empty else 0.0

            direct_equity = float(s_direct.sum()) if isinstance(s_direct, pd.Series) and not s_direct.empty else 0.0

            comb_equity = _z(mf_equity + direct_equity)
            comb_debt   = _z(mf_debt)
            comb_cash   = _z(mf_cash)

            # Title & chart
            chart_heading(
                "Combined Asset Allocation",
                "MF (scaled net buckets) + Direct stocks. If Equity is negative, we switch to a bar."
            )
            df_mix = pd.DataFrame([
                {"asset_class":"Equity","value": comb_equity},
                {"asset_class":"Debt",  "value": comb_debt},
                {"asset_class":"Cash",  "value": comb_cash},
            ])
            # Percent vs total combined *investment* (MF invested + direct rupees)
            denom = max(total_combined_investment, 1e-9)
            df_mix["pct"] = df_mix["value"] / denom * 100.0
            df_mix["Value (₹)"] = df_mix["value"].map(_fmt_inr)
            df_mix["% of Total"] = df_mix["pct"].map(lambda v: f"{float(v):.1f}%")

            display_as_bar = (comb_equity < -EPS) or (df_mix["value"] < -EPS).any()
            if display_as_bar:
                plot_df = df_mix.sort_values("value", ascending=True)
                st.plotly_chart(
                    hbar_with_text(plot_df["value"], plot_df["asset_class"], plot_df["% of Total"]),
                    use_container_width=True, config=PLOT_CFG
                )
            else:
                st.plotly_chart(
                    donut(df_mix["asset_class"], df_mix["value"]),
                    use_container_width=True, config=PLOT_CFG
                )
            with st.expander("Details", expanded=False):
                st.dataframe(df_mix[["asset_class","Value (₹)","% of Total"]], hide_index=True, use_container_width=True)

            st.markdown("---")

            # 2) Equity Source Split (MF vs Direct) — within Equity only
            mf_eq = _z(mf_equity)
            dir_eq = _z(direct_equity)
            chart_heading(
                "Equity Source Split",
                "Share of your combined equity coming from Mutual Funds vs Direct stocks."
            )
            if mf_eq < -EPS:
                # show bar if MF equity is truly negative
                df_src = pd.DataFrame([
                    {"source":"Direct Equity","value": dir_eq},
                    {"source":"MF Equity","value": mf_eq},
                ])
                total_eq = max(abs(df_src["value"]).sum(), 1e-9)
                df_src["% of Equity"] = df_src["value"] / total_eq * 100.0
                st.plotly_chart(
                    hbar_with_text(df_src["value"], df_src["source"], df_src["% of Equity"].map(lambda v: f"{float(v):.1f}%")),
                    use_container_width=True, config=PLOT_CFG
                )
            else:
                vals = [dir_eq, mf_eq]
                labels = ["Direct Equity","MF Equity"]
                st.plotly_chart(donut(labels, vals), use_container_width=True, config=PLOT_CFG)

            st.markdown("---")

            # 3 & 4) Combined pies (industry_rating + market_cap)
            comb_series = _combined_equity_series(mf_rows, holdings_info, s_direct)
            if comb_series is None or comb_series.empty:
                st.info("No positive combined equity exposure available to analyze.")
            else:
                c1, c2 = st.columns(2)

                with c1:
                    chart_heading(
                        "Combined Net Industry Distribution",
                        "User-weighted (MF) + Direct; positives only. Small slices grouped as Other."
                    )
                    sector_df = _pies_from_series(comb_series, equities_info, label_col="industry_rating", small_thr=0.03)
                    if sector_df.empty:
                        st.info("No industry data.")
                    else:
                        st.plotly_chart(donut(sector_df["label"], sector_df["value"]), use_container_width=True, config=PLOT_CFG)
                        with st.expander("Details", expanded=False):
                            show = sector_df.copy()
                            show["Value (₹)"] = show["value"].map(_fmt_inr)
                            show["% of Equity"] = show["pct"].map(lambda v: f"{float(v):.1f}%")
                            st.dataframe(show[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Industry (rating)"}),
                                        hide_index=True, use_container_width=True)

                with c2:
                    chart_heading(
                        "Combined Net Market-Cap Distribution",
                        "User-weighted (MF) + Direct; positives only. Small slices grouped as Other."
                    )
                    mcap_df = _pies_from_series(comb_series, equities_info, label_col="market_cap", small_thr=0.03)
                    if mcap_df.empty:
                        st.info("No market-cap data.")
                    else:
                        st.plotly_chart(donut(mcap_df["label"], mcap_df["value"]), use_container_width=True, config=PLOT_CFG)
                        with st.expander("Details", expanded=False):
                            show = mcap_df.copy()
                            show["Value (₹)"] = show["value"].map(_fmt_inr)
                            show["% of Equity"] = show["pct"].map(lambda v: f"{float(v):.1f}%")
                            st.dataframe(show[["label","Value (₹)","% of Equity"]].rename(columns={"label":"Market Cap"}),
                                        hide_index=True, use_container_width=True)

            st.markdown("---")

            # 5) Top-10 Combined Underlying Stock Holdings
            if comb_series is None or comb_series.empty:
                st.info("No positive combined equity exposure available to rank.")
            else:
                top_n = 10
                s_sorted = comb_series.sort_values(ascending=False).head(top_n)
                # % label should be % of TOTAL COMBINED INVESTMENT (MF invested + Direct invested)
                denom = max(total_combined_investment, 1e-9)
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
                    use_container_width=True, config=PLOT_CFG
                )
                with st.expander("Details", expanded=False):
                    show = df_top.copy()
                    show["Value (₹)"] = show["value"].map(_fmt_inr)
                    show["% of Total Investment"] = show["pct_of_total_investment"].map(lambda v: f"{float(v):.1f}%")
                    st.dataframe(
                        show[["company_name","isin","Value (₹)","% of Total Investment"]].rename(columns={"company_name":"Company"}),
                        hide_index=True, use_container_width=True
                    )
    '''

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
                xaxis_title="Rupees", yaxis_title="",
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
                {"asset_class":"Cash",  "value": comb_cash},
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
                    use_container_width=True, config=PLOT_CFG
                )
            else:
                st.plotly_chart(
                    donut(df_mix["asset_class"], df_mix["value"]),
                    use_container_width=True, config=PLOT_CFG
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
                    use_container_width=True, config=PLOT_CFG
                )
            else:
                st.plotly_chart(donut(["Direct Equity","MF Equity"], [dir_eq, mf_eq]), use_container_width=True, config=PLOT_CFG)

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
                        st.plotly_chart(donut(sector_df["label"], sector_df["value"]), use_container_width=True, config=PLOT_CFG)
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
                        st.plotly_chart(donut(mcap_df["label"], mcap_df["value"]), use_container_width=True, config=PLOT_CFG)
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
                    use_container_width=True, config=PLOT_CFG
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
        st.caption("How similar your selected funds are to each other. "
                "Overlap is computed on **positive** net equity exposures and scaled to your invested amounts.")

        # Ensure these imports exist at top of file:
        # from core.transforms import fund_net_equity_series_map, fund_overlap_matrix

        mf_rows = st.session_state.get("mf_portfolio", [])
        series_map = fund_net_equity_series_map(mf_rows, holdings_info)  # per-fund ISIN→₹ (positives only)

        if not series_map:
            st.info("Add mutual funds in Step 1 to see the overlap matrix.")
        else:
            # Build overlap matrix (values in [0,1], diagonal == 1)
            mat = fund_overlap_matrix(series_map)

            # Heatmap
            names = mat.index.tolist()
            z = mat.values.astype(float)

            heat = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=names,
                    y=names,
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    hovertemplate="%{y} vs %{x}: %{z:.1%}<extra></extra>",
                    showscale=True,
                )
            )
            heat.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=64, b=10),
                title={
                    "text": (
                        "<b>Fund–Fund Overlap</b>"
                        "<br><span style='font-size:12px;color:#A0A0A0'>"
                        "Definition: Overlap(i, j) = shared positive rupee exposure ÷ smaller fund’s positive equity."
                        "</span>"
                    ),
                    "y": 0.98, "x": 0.02, "xanchor": "left", "yanchor": "top",
                },
            )
            st.plotly_chart(heat, use_container_width=True)

            st.markdown("—")

            # Top overlapping pairs (upper triangle only)
            pairs = []
            n = len(names)
            for i in range(n):
                for j in range(i + 1, n):
                    v = float(mat.iat[i, j])
                    pairs.append({
                        "Fund A": names[i],
                        "Fund B": names[j],
                        "Overlap % of smaller": v * 100.0,
                    })

            if pairs:
                top_df = (
                    pd.DataFrame(pairs)
                    .sort_values("Overlap % of smaller", ascending=False, ignore_index=True)
                )
                top_df["Overlap % of smaller"] = top_df["Overlap % of smaller"].map(lambda x: f"{x:.1f}%")
                st.markdown("#### Top Overlapping Fund Pairs")
                st.dataframe(
                    top_df.head(15),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No meaningful overlap detected between the selected funds.")
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
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

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
