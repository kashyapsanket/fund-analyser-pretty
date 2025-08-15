# app.py
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st

# ---- tolerant imports (package or flat) ----


from core.ui import setup_theme, inject_css, header_logo  # type: ignore



from views.page_landing import render as render_landing  # type: ignore
from views.page_mutual_funds import render as render_mf  # type: ignore
from views.page_stocks import render as render_stocks  # type: ignore
from views.page_analysis import render as render_analysis  # type: ignore


from core.data_access import get_conn, load_fund_info, load_equities_info, load_holdings_for_funds  # type: ignore

st.set_page_config(page_title="EssKay Wealth — Portfolio Analyzer", layout="wide")

# ---- theme + css + header ----
setup_theme()
inject_css()

def _asset_logo() -> str | None:
    here = Path(__file__).parent
    for p in ["assets/logo.jpeg", "assets/logo.png", "assets/logo.jpg"]:
        fp = here / p
        if fp.exists():
            return str(fp)
    return None

logo_path = _asset_logo()
# show header on all pages EXCEPT landing
# if st.session_state.get("page_idx", 0) != 0 and st.session_state.get("page_idx", 0) != 1:
#    header_logo(_asset_logo())


# ---- session state ----
ss = st.session_state
ss.setdefault("page_idx", 0)                  # 0 landing, 1 mf, 2 stocks, 3 analysis
ss.setdefault("mf_portfolio", [])             # [{amc_name,fund_name,amount}]
ss.setdefault("stock_portfolio", [])          # [{company_name, amount}]

# ---- db path resolution (no assumptions) ----
def _resolve_db_path() -> str:
    try:
        if "db_path" in st.secrets:
            return str(st.secrets["db_path"])
    except Exception:
        pass
    env = os.environ.get("ESSKAY_DB_PATH")
    if env:
        return env
    # common fallbacks
    for cand in [
        "data/fund_analysis.db", "data/funds.db", "data/app.db",
        "funds.db", "portfolio_analytics.db"
    ]:
        if Path(cand).exists():
            return cand
    Path("data").mkdir(parents=True, exist_ok=True)
    return "/Users/sanket/Documents/WealthyMonitor/portfolio_analytics.db"

# ---- data boot ----
conn = None
fund_info = None
equities_info = None
try:
    db_path = _resolve_db_path()
    if Path(db_path).exists():
        conn = get_conn(db_path)
        fund_info = load_fund_info(conn)
        equities_info = load_equities_info(conn)
        holdings_info = load_holdings_for_funds(conn)
    else:
        st.warning(f"Database not found at **{db_path}**. Set `st.secrets['db_path']` or `ESSKAY_DB_PATH`.")
except Exception as e:
    st.error(f"DB init error: {e}")

# ---- routing ----
idx = int(ss.page_idx)
if idx == 0:
    render_landing()
elif idx == 1:
    if fund_info is None:
        st.warning("Fund data not available.")
    else:
        render_mf(fund_info)
elif idx == 2:
    render_stocks(equities_info)
elif idx == 3:
    if conn is None or equities_info is None:
        st.warning("Analysis requires database and equity metadata.")
    else:
        render_analysis(equities_info, fund_info, holdings_info)
else:
    st.info("Page under construction.")
