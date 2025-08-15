# page_landing.py
from __future__ import annotations
import streamlit as st
from core.ui import hero_linear, sticky_footer

def _tile(title: str, body: str, icon: str = "📊", disabled: bool = False):
    dim = " skw-disabled" if disabled else ""
    st.markdown(
        f"""
        <div class="skw-tile{dim}">
          <div class="skw-icon">{icon}</div>
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# page_landing.py (only the render() body shown)

def render():
    hero_linear(
        title="Want to understand your investments better?",
        subtitle="Turn complex details about your investment portfolio into clean, personalized insights - underlying companies, market cap allocation, sectoral allocation, as well as historical performance of mutual funds",
        badge="EssKay Wealth",
        hero_image="assets/logo.jpeg",   # optional
    )

    st.markdown('<div id="tiles" class="skw-tiles">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        _tile("Portfolio Analytics",
              "Add MFs and direct stocks, then see underlying portfolio analytics",
              "🧭", disabled=False)
        # ↓ spacing wrapper so the gap is between card container and button container
        st.markdown('<div class="tile-cta">', unsafe_allow_html=True)
        if st.button("Start Analysis", key="start_analysis"):
            st.session_state.page_idx = 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        _tile("Benchmarking",
              "Compare your portfolio performance against indices and our recommended model portfolios",
              "📐", disabled=True)
        st.markdown('<div class="tile-cta">', unsafe_allow_html=True)
        st.button("Coming Soon", key="bench_cs", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    sticky_footer("Copyright © EssKay Wealth 2025. Mutual fund investments are subject to market risks. Read all scheme related documents.")
