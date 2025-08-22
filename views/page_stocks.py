# views/page_stocks.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from core.ui import page_head, sticky_footer

def _inr(x) -> str:
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹0.00"

def render(equities_info: pd.DataFrame):
    """
    equities_info: DataFrame with at least column 'company_name'.
                   (Extra cols ignored; we don't show sector/mcap here.)
    """
    # --- state ---
    st.session_state.setdefault("stock_portfolio", [])
    rows = st.session_state.stock_portfolio

    # --- Header tile (Linear-style, same as MF) ---
    page_head(
        title="Step 2 : Build your Stock Portfolio",
        subtitle="Add your direct stocks and market values",
        icon="📈",
    )

    # --- Form row: Company, Amount, Add ---
    companies = sorted(set(equities_info["company_name"].dropna().tolist())) if equities_info is not None else []
    c1, c2, c3 = st.columns([2.8, 1.0, .6], vertical_alignment="bottom")

    with c1:
        company = st.selectbox("Company", ["Select Company"] + companies, index=0, key="stk_company")

    with c2:
        amount = st.number_input("Market Value (₹)", min_value=0.0, step=1000.0, value=0.0, key="stk_amount")

    with c3:
        st.write("")  # align baseline
        if st.button("Add", type="primary", key="stk_add_btn"):
            if company == "Select Company" or amount <= 0:
                st.toast("Pick a company and enter a positive amount.", icon="⚠️")
            else:
                # merge duplicate company rows by accumulating amount
                merged = False
                for r in rows:
                    if r["company_name"] == company:
                        r["amount"] = float(r.get("amount", 0)) + float(amount)
                        merged = True
                        break
                if not merged:
                    rows.append({"company_name": company, "amount": float(amount)})
                st.toast(f"Added {company} ({_inr(amount)}).", icon="✅")

    # subtle separator
    st.markdown('<div class="skw-row-sep"></div>', unsafe_allow_html=True)

    # --- List header (aligned to rows) ---
    h1, h2, _ = st.columns([2.8, 1.0, .3])
    h1.caption("Company"); h2.caption("Market Value")

    # --- Rows (no inline edit, only delete) ---
    if not rows:
        st.info("No stock positions added yet.")
    else:
        for i, r in enumerate(rows):
            c1, c2, c3 = st.columns([2.8, 1.0, .3], gap="small")
            with c1:
                st.write(r["company_name"])
            with c2:
                st.markdown(f'<div class="skw-amount">{_inr(r.get("amount", 0))}</div>', unsafe_allow_html=True)
            with c3:
                if st.button("🗑️", key=f"stk_del_{i}", help="Remove"):
                    del st.session_state.stock_portfolio[i]
                    st.rerun()
            st.markdown('<div class="skw-row-sep" style="margin:8px 0;"></div>', unsafe_allow_html=True)

    # --- Summary bar: two KPIs (Total Stocks Value, Number of Positions) ---
    total_val = sum(float(r.get("amount", 0) or 0) for r in rows)
    count = len(rows)
    st.markdown(
        f"""
        <div class="skw-total">
          <div class="kpis">
            <div>
              <div class="label">Total Stocks Value</div>
              <div class="val">{_inr(total_val)}</div>
            </div>
            <div>
              <div class="label">Number of Positions</div>
              <div class="val">{count}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Nav: Back (to MF) left, Next (to Analysis) right ---
    st.markdown('<div class="skw-nav">', unsafe_allow_html=True)
    back_col, spacer, next_col = st.columns([1, 6, 1])
    with back_col:
        if st.button("← Back", key="stk_back"):
            st.session_state.page_idx = 1  # back to Mutual Funds
            st.rerun()
    with next_col:
        disabled = False
        if st.button("Next • Analysis", type="primary", disabled=disabled, key="stk_next"):
            st.session_state.page_idx = 3  # go to Analysis
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Disclaimer footer ---
    sticky_footer("Mutual fund investments are subject to market risks. Read all scheme related documents.")
