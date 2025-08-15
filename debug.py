import sqlite3, pandas as pd, os
from pathlib import Path
import streamlit as st

def _db_path():
    # Adjust if you store the DB elsewhere
    return "portfolio_analytics.db"

db = _db_path()
# st.write("DB path:", str(db), "| exists:", db.exists(), "| size(bytes):", db.stat().st_size if db.exists() else 0)

with sqlite3.connect(str(db)) as con:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", con)
    print("Tables:", tables)
    # Find a likely equities table
    guess = pd.read_sql_query("""
      SELECT name FROM sqlite_master
      WHERE type='table' AND (name LIKE '%equ%' OR name LIKE '%stock%');
    """, con)
    print("Equity-like tables:", guess)

    # If equities_info exists, show row count + first rows
    if "equities_info" in tables["name"].tolist():
        cnt = pd.read_sql_query("SELECT COUNT(*) AS n FROM equities_info;", con)["n"].iat[0]
        print("equities_info rows:", cnt)
        if cnt > 0:
            print(pd.read_sql_query("SELECT * FROM equities_info LIMIT 5;", con))
