from __future__ import annotations
import streamlit as st
import pandas as pd
from core.ui import page_head, sticky_footer

def _inr(x) -> str:
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹0.00"

def render(fund_info: pd.DataFrame):
    st.session_state.setdefault("mf_portfolio", [])
    rows = st.session_state.mf_portfolio

    # --- Header tile (no expander, no weird bar) ---
    page_head(
        title="Step 1 : Build your Mutual Fund Portfolio",
        subtitle="Add your mutual fund schemes and market values",
        icon="🧭",
    )

    # --- Form row (AMC, Fund, Amount, Add) ---
    amcs = sorted(set(fund_info["amc_name"].dropna().tolist())) if fund_info is not None else []
    c1, c2, c3, c4 = st.columns([1.1, 2.4, .9, .6], vertical_alignment="bottom")

    with c1:
        amc = st.selectbox("AMC", ["Select AMC"] + amcs, index=0, key="mf_amc")

    with c2:
        if amc != "Select AMC":
            opts = fund_info.loc[fund_info["amc_name"] == amc, "fund_name"].tolist()
            fund = st.selectbox("Fund", ["Select Fund"] + opts, index=0, key="mf_fund")
        else:
            fund = "Select Fund"

    with c3:
        amt = st.number_input("Market Value (₹)", min_value=0.0, step=1000.0, value=0.0, key="mf_amt")

    with c4:
        st.write("")  # align baseline
        if st.button("Add", type="primary", key="mf_add_btn"):
            if amc == "Select AMC" or fund == "Select Fund" or amt <= 0:
                st.toast("Pick an AMC & Fund and enter a positive amount.", icon="⚠️")
            else:
                # merge duplicate (amc, fund)
                merged = False
                for r in rows:
                    if r["amc_name"] == amc and r["fund_name"] == fund:
                        r["amount"] = float(r.get("amount", 0)) + float(amt)
                        merged = True
                        break
                if not merged:
                    rows.append({"amc_name": amc, "fund_name": fund, "amount": float(amt)})
                st.toast(f"Added {fund} ({_inr(amt)}).", icon="✅")

    # subtle separator under form
    st.markdown('<div class="skw-row-sep"></div>', unsafe_allow_html=True)

    # --- List header (use the same column ratios as rows) ---
    h1, h2, h3, _ = st.columns([1.1, 2.4, .9, .3])
    h1.caption("AMC"); h2.caption("Fund"); h3.caption("Market Value")

    # --- Rows (perfect alignment using the same columns layout) ---
    if not rows:
        st.info("No funds added yet.")
    else:
        for i, r in enumerate(rows):
            c1, c2, c3, c4 = st.columns([1.1, 2.4, .9, .3], gap="small")
            with c1:
                st.markdown(f'<span class="skw-pill">{r["amc_name"]}</span>', unsafe_allow_html=True)
            with c2:
                st.write(r["fund_name"])
            with c3:
                st.markdown(f'<div class="skw-amount">{_inr(r.get("amount", 0))}</div>', unsafe_allow_html=True)
            with c4:
                if st.button("🗑️", key=f"del_{i}", help="Remove"):
                    del st.session_state.mf_portfolio[i]
                    st.rerun()
            # light divider between rows
            st.markdown('<div class="skw-row-sep" style="margin:8px 0;"></div>', unsafe_allow_html=True)

    # --- Total strip (brand blues) ---
    total_val = sum(float(r.get("amount", 0) or 0) for r in rows)
    num_funds = len(rows)
    st.markdown(
        f"""
        <div class="skw-total">
        <div class="kpis">
            <div>
            <div class="label">Total MF Value</div>
            <div class="val">{_inr(total_val)}</div>
            </div>
            <div>
            <div class="label">Number of Funds</div>
            <div class="val">{num_funds}</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Nav: Back left, Next right ---
    st.markdown('<div class="skw-nav">', unsafe_allow_html=True)
    nb, _, nn = st.columns([1, 6, 1])
    with nb:
        if st.button("← Back", key="mf_back"):
            st.session_state.page_idx = 0
            st.rerun()
    with nn:
        disabled = len(rows) == 0
        if st.button("Next • Add Stocks", type="primary", disabled=disabled, key="mf_next"):
            st.session_state.page_idx = 2
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer disclaimer ---
    sticky_footer("Mutual fund investments are subject to market risks. Read all scheme related documents.")
