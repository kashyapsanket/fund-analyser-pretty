# ui.py
from __future__ import annotations
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Iterable

# ---------------- Plotly theme ----------------
def setup_theme():
    colorway = ["#1DA7E1", "#59C7F2", "#7EC8E3", "#3D7EA6", "#98DDF7", "#0E5A84"]
    pio.templates["skw_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA", family="Inter, -apple-system, Segoe UI, Roboto, sans-serif"),
            colorway=colorway,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
    )
    pio.templates.default = "skw_dark"

# ---------------- CSS (Linear-inspired) ----------------

def inject_css_old():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          :root{
            --skw-brand:#1DA7E1;
            --skw-brand-2:#59C7F2;
            --skw-bg:#0E1117;
            --skw-card:#1C1F26;
            --skw-border:#2A2D35;
            --skw-text:#FAFAFA;
            --skw-muted:#A0A0A0;
            --skw-radius:20px;
          }

          .block-container{ max-width:1180px; padding-bottom:100px; }

          html, body, [class*="css"] { font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; }
          .skw-h1{ font-weight:800; font-size: clamp(34px, 5.2vw, 56px); line-height:1.02; letter-spacing:-0.02em; color:var(--skw-text); }
          .skw-sub{ color:var(--skw-muted); font-size:15px; line-height:1.5; max-width:48ch; }

          .skw-hero-wrap{
            display:grid; grid-template-columns: minmax(0,1.4fr) minmax(0,1fr);
            gap:36px; align-items:end; margin-top:6px; margin-bottom:24px;
          }
          .skw-hero-badge{ display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border-radius:999px;
                           background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                           border:1px solid var(--skw-border); color:#CFEFFF; font-weight:600; font-size:12px; }
          .skw-hero-art{
            height:160px; border-radius: var(--skw-radius);
            background:
              radial-gradient(120px 80px at 70% 30%, rgba(29,167,225,.25), transparent 60%),
              radial-gradient(160px 100px at 30% 70%, rgba(89,199,242,.18), transparent 60%),
              #13161B;
            border:1px solid var(--skw-border);
            background-size: cover; background-position:center; /* will apply if we inject an image */
          }

          /* Tiles */
          .skw-tiles{ display:grid; grid-template-columns: 1fr 1fr; gap:32px; }  /* more spacing */
          .skw-tile{
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00)), var(--skw-card);
            border:1px solid var(--skw-border); border-radius: var(--skw-radius);
            padding:18px 18px 14px 18px; min-height:120px;
            transition: transform .08s ease, border-color .12s ease, background .2s ease;
            position:relative; overflow:hidden;
          }
          .skw-tile:hover{ transform: translateY(-1px); border-color:#3a3f49; }
          .skw-tile h3{ margin:0 0 6px 0; font-size:18px; font-weight:700; }
          .skw-tile p{ margin:0; color:var(--skw-muted); font-size:14px; line-height:1.45; }
          .skw-tile .skw-icon{
            width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
            background:linear-gradient(180deg, rgba(29,167,225,.18), rgba(29,167,225,.06)); border:1px solid #2b3a44; color:#E8F8FF;
            position:absolute; top:14px; right:14px;
          }
          .skw-disabled{ opacity:.55; filter:grayscale(.8); }

          /* Compact brand buttons */
          .stButton > button{
            background:var(--skw-brand) !important; color:#04121C !important;
            font-weight:600; border:0; border-radius:12px; min-height:38px !important; padding:8px 14px !important; font-size:14px !important;
          }
          .stButton > button:hover{ filter:brightness(1.06) }
          .stButton > button:active{ filter:brightness(.95) }
          .stButton > button:focus{ outline:2px solid var(--skw-brand) !important; box-shadow:0 0 0 4px rgba(29,167,225,.25) !important; }
          .stButton > button:disabled{ background:#3a3a3a !important; color:#bdbdbd !important; }

          /* Landing-only: ensure each CTA sits nicely under its tile */
          #tiles .stButton{ display:block; margin-top:10px; }  /* prevents “bleed” */
          
          /* Sticky footer */
          .skw-footer-fixed{ position:fixed; left:0; right:0; bottom:0; z-index:9999; background:#0E1117;
                             border-top:1px solid var(--skw-border); padding:10px 24px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_css():
    st.markdown(
        """
        <style>
          /* Modern, quieter font */
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

          :root{
            --skw-brand:#1DA7E1;
            --skw-brand-2:#59C7F2;
            --skw-bg:#0E1117;
            --skw-card:#1C1F26;
            --skw-border:#2A2D35;
            --skw-text:#FAFAFA;
            --skw-muted:#A0A0A0;
            --skw-radius:20px;
          }

          /* container + footer padding */
          .block-container{ max-width:1160px; padding-bottom:100px; }

          /* Global typography */
          html, body, [class*="css"] { font-family: "Plus Jakarta Sans", Inter, -apple-system, Segoe UI, Roboto, sans-serif; }

          /* Hero */
          .skw-h1{
            font-weight:700;                            /* slightly lighter than before */
            font-size:clamp(32px,5vw,52px);            /* a hair smaller = calmer */
            line-height:1.04; letter-spacing:-0.01em; color:var(--skw-text);
          }
          .skw-sub{ color:var(--skw-muted); font-size:15px; line-height:1.55; max-width:48ch; }
          .skw-hero-wrap{
            display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);
            gap:36px; align-items:end; margin-top:6px; margin-bottom:24px;
          }
          .skw-hero-badge{
            display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border-radius:999px;
            background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border:1px solid var(--skw-border); color:#CFEFFF; font-weight:600; font-size:12px;
          }
          .skw-hero-art{
            height:180px; border-radius:var(--skw-radius);
            background:
              radial-gradient(120px 80px at 70% 30%, rgba(29,167,225,.25), transparent 60%),
              radial-gradient(160px 100px at 30% 70%, rgba(89,199,242,.18), transparent 60%),
              #13161B;
            background-size:cover; background-position:center;
            border:1px solid var(--skw-border);
          }

          /* Tiles / cards */
          .skw-tiles{ display:grid; grid-template-columns:1fr 1fr; gap:40px; }  /* more space between cards */
          .skw-tile{
            background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00)), var(--skw-card);
            border:1px solid var(--skw-border); border-radius:var(--skw-radius);
            padding:18px 18px 16px 18px; min-height:120px; position:relative; overflow:hidden;
            transition:transform .08s ease, border-color .12s ease, background .2s ease;
          }
          .skw-tile:hover{ transform:translateY(-1px); border-color:#3a3f49; }
          .skw-tile h3{ margin:0 0 6px 0; font-size:18px; font-weight:600; }    /* lighter title weight */
          .skw-tile p{  margin:0; color:var(--skw-muted); font-size:14px; line-height:1.45; }
          .skw-tile .skw-icon{
            width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
            background:linear-gradient(180deg, rgba(29,167,225,.18), rgba(29,167,225,.06));
            border:1px solid #2b3a44; color:#E8F8FF; position:absolute; top:14px; right:14px;
          }
          .skw-disabled{ opacity:.55; filter:grayscale(.8); }

          /* Buttons: compact + on-brand focus */
          .stButton > button{
            background:var(--skw-brand) !important; color:#04121C !important;
            font-weight:600; border:0; border-radius:12px;
            min-height:38px !important; padding:8px 14px !important; font-size:14px !important;
          }
          .stButton > button:hover{ filter:brightness(1.06) }
          .stButton > button:active{ filter:brightness(.95) }
          .stButton > button:focus{ outline:2px solid var(--skw-brand) !important; box-shadow:0 0 0 4px rgba(29,167,225,.25) !important; }
          .stButton > button:disabled{ background:#3a3a3a !important; color:#bdbdbd !important; }

          /* >>> Landing page CTA spacing <<< */
          #tiles .stButton{ display:block; margin-top:20px !important; }        /* clear space under card copy */
          #tiles .stButton > button{ width:fit-content !important; }           /* no full-width bleed */

          /* Sticky footer */
          .skw-footer-fixed{
            position:fixed; left:0; right:0; bottom:0; z-index:9999;
            background:#0E1117; border-top:1px solid var(--skw-border); padding:10px 24px;
        
           /* CTA wrapper so spacing applies outside Streamlit's widget container */
           .tile-cta{ margin-top:16px !important; }

          }
          .skw-h1{ font-weight:600; font-size:clamp(30px, 4.6vw, 48px); }

          /* --- Step page: panel & list (Linear-ish) --- */
        .skw-panel{
        background:var(--skw-card);
        border:1px solid var(--skw-border);
        border-radius:24px;
        padding:22px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        .skw-panel h2{
        margin:0 0 4px 0; font-weight:700; font-size:24px; letter-spacing:-0.01em;
        }
        .skw-panel .muted{ color:var(--skw-muted); font-size:14px; margin-bottom:12px; }

        .skw-pill{
        display:inline-flex; padding:4px 10px; gap:8px; align-items:center;
        background:rgba(255,255,255,0.04);
        border:1px solid var(--skw-border);
        border-radius:999px; font-size:12px; color:#DDE6EE;
        }

        .skw-list{ border-top:1px solid var(--skw-border); margin-top:8px; }
        .skw-row{
        display:grid; grid-template-columns: 1.1fr 2.4fr .9fr 40px;
        align-items:center; gap:12px;
        padding:12px 6px; border-bottom:1px solid var(--skw-border);
        }
        .skw-row:hover{ background:rgba(255,255,255,0.02); }
        .skw-amount{ text-align:right; font-variant-numeric: tabular-nums; }

        .skw-icon-btn{
        display:inline-flex; align-items:center; justify-content:center;
        width:34px; height:34px; border-radius:10px;
        background:#2B2F36; border:1px solid var(--skw-border); color:#B5C6D6;
        }
        .skw-icon-btn:hover{ filter:brightness(1.05); border-color:#3a3f49; }

        /* Total strip */
        .skw-total{
        margin-top:16px; border-radius:16px; padding:18px;
        background:linear-gradient(90deg, #6CA9FF 0%, #E96FD6 100%);
        color:#0d1016; text-align:center; font-weight:700;
        box-shadow: inset 0 0 1px rgba(255,255,255,0.4);
        }
        .skw-total .sub{ display:block; font-weight:600; opacity:.75; margin-bottom:2px; }

        /* Step pages */
        .skw-panel{ background:var(--skw-card); border:1px solid var(--skw-border); border-radius:24px; padding:22px; box-shadow:inset 0 1px 0 rgba(255,255,255,.02); }
        .skw-panel h2{ margin:0 0 4px 0; font-weight:700; font-size:24px; letter-spacing:-.01em; }
        .skw-panel .muted{ color:var(--skw-muted); font-size:14px; margin-bottom:12px; }

        .skw-pill{ display:inline-flex; padding:4px 10px; background:rgba(255,255,255,.04); border:1px solid var(--skw-border); border-radius:999px; font-size:12px; color:#DDE6EE; }

        /* Use one grid for both header and rows */
        .skw-grid{ display:grid; grid-template-columns: 1.1fr 2.4fr .9fr 40px; gap:12px; align-items:center; }
        .skw-list{ border-top:1px solid var(--skw-border); margin-top:10px; }
        .skw-head{ padding:6px 6px; }
        .skw-row{ padding:12px 6px; border-bottom:1px solid var(--skw-border); }
        .skw-row:hover{ background:rgba(255,255,255,.02); }
        .skw-amount{ text-align:right; font-variant-numeric: tabular-nums; }

        /* Small icon button */
        .skw-icon-btn{ display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:10px; background:#2B2F36; border:1px solid var(--skw-border); color:#B5C6D6; }
        .skw-icon-btn:hover{ filter:brightness(1.05); border-color:#3a3f49; }

        /* Brand-blue total strip */
        .skw-total{ margin-top:16px; border-radius:16px; padding:18px; background:linear-gradient(90deg,#0F79C9 0%,#1DA7E1 45%,#59C7F2 100%); color:#071019; text-align:center; }
        .skw-total .sub{ display:block; color:#DCEFFB; opacity:.9; margin-bottom:2px; font-weight:600; }
        .skw-total .val{ font-weight:800; font-size:20px; }

        /* Compact nav buttons row */
        .skw-nav{ display:flex; gap:16px; justify-content:space-between; margin-top:14px; }
        .skw-nav .stButton > button{ width:auto !important; padding:8px 14px !important; min-height:38px !important; }

        /* Step page panel */
        .skw-panel{ background:var(--skw-card); border:1px solid var(--skw-border); border-radius:24px; padding:22px; box-shadow:inset 0 1px 0 rgba(255,255,255,.02); }
        .skw-panel h2{ margin:0 0 4px 0; font-weight:700; font-size:24px; letter-spacing:-.01em; }
        .skw-panel .muted{ color:var(--skw-muted); font-size:14px; margin-bottom:12px; }

        /* List uses one grid for header + rows so numbers align perfectly */
        .skw-grid{ display:grid; grid-template-columns: 1.1fr 2.4fr .9fr 34px; gap:12px; align-items:center; }
        .skw-list{ border-top:1px solid var(--skw-border); margin-top:10px; }
        .skw-head{ padding:6px 6px; }
        .skw-row{ padding:12px 6px; border-bottom:1px solid var(--skw-border); }
        .skw-row:hover{ background:rgba(255,255,255,.02); }
        .skw-amount{ text-align:right; font-variant-numeric: tabular-nums; }

        .skw-pill{ display:inline-flex; padding:4px 10px; background:rgba(255,255,255,.04); border:1px solid var(--skw-border); border-radius:999px; font-size:12px; color:#DDE6EE; }

        /* compact icon button */
        .skw-icon-btn{ width:34px; height:34px; border-radius:10px; background:#2B2F36; border:1px solid var(--skw-border); color:#B5C6D6; display:flex; align-items:center; justify-content:center; }
        .skw-icon-btn:hover{ filter:brightness(1.05); border-color:#3a3f49; }

        /* Brand-blue total strip */
        .skw-total{ margin-top:16px; border-radius:16px; padding:18px; background:linear-gradient(90deg,#0F79C9 0%,#1DA7E1 45%,#59C7F2 100%); color:#071019; text-align:center; }
        .skw-total .sub{ display:block; color:#DCEFFB; opacity:.9; margin-bottom:2px; font-weight:600; }
        .skw-total .val{ font-weight:800; font-size:20px; }

        /* Compact nav */
        .skw-nav{ display:flex; gap:12px; justify-content:space-between; margin-top:14px; }
        .skw-nav .stButton > button{ width:auto !important; padding:8px 14px !important; min-height:38px !important; }

        /* Utility: removes mysterious bar if any empty hr/div sneaks in */
        .skw-hide{ display:none !important; }

        /* Page header tile (matches landing vibe) */
        .skw-pagehead{
        display:flex; align-items:flex-start; gap:14px;
        background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
        border:1px solid var(--skw-border);
        border-radius:20px; padding:16px 18px; margin:6px 0 10px 0;
        }
        .skw-pagehead .icon{
        width: 45px; height:45px; border-radius:12px;
        background:linear-gradient(180deg, rgba(29,167,225,.18), rgba(29,167,225,.06));
        border:1px solid #2b3a44; color:#E8F8FF; display:flex; align-items:center; justify-content:center;
        font-weight:700;
        }
        .skw-pagehead .title{ font-weight:700; font-size:30px; letter-spacing:-0.01em; }
        .skw-pagehead .sub{ color:var(--skw-muted); font-size: 20px; margin-top:2px; }

        /* MF list: one consistent grid via columns; helpers */
        .skw-pill{ display:inline-flex; padding:4px 10px; background:rgba(255,255,255,.04); border:1px solid var(--skw-border); border-radius:999px; font-size:12px; color:#DDE6EE; }
        .skw-row-sep{ height:1px; background:var(--skw-border); margin:2px 0 10px 0; }

        /* Total strip in brand blues */
        .skw-total{ margin-top:16px; border-radius:16px; padding:18px; background:linear-gradient(90deg,#0F79C9 0%,#1DA7E1 45%,#59C7F2 100%); color:#071019; text-align:center; }
        .skw-total .sub{ display:block; color:#DCEFFB; opacity:.9; margin-bottom:2px; font-weight:600; }
        .skw-total .val{ font-weight:800; font-size:20px; }

        /* Compact nav row */
        .skw-nav{ margin-top:14px; }
        .skw-nav .stButton > button{ width:auto !important; padding:8px 14px !important; min-height:38px !important; }

        /* Amount column: left aligned (was right) */
        .skw-amount{ text-align:left; font-variant-numeric: tabular-nums; }

        /* Summary bar: translucent blue that blends with bg + 2-KPI grid */
        .skw-total{
        margin-top:16px; border-radius:16px; padding:18px;
        background:linear-gradient(180deg, rgba(29,167,225,.16), rgba(89,199,242,.12));
        border:1px solid var(--skw-border);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.03);
        color:#EAF6FF;
        }
        .skw-total .kpis{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; align-items:center; }
        .skw-total .label{ font-size:13px; font-weight:600; opacity:.9; margin-bottom:4px; color:#CFE9F8; }
        .skw-total .val{ font-weight:800; font-size:20px; color:#F5FBFF; }

        
        /* Tabs: compact, brand-accent, calm */
        .stTabs [role="tablist"]{
        gap:10px;
        border-bottom:1px solid var(--skw-border);
        margin-bottom:8px;
        }
        .stTabs [role="tab"]{
        padding:8px 14px;
        border:1px solid var(--skw-border);
        border-bottom:none;
        border-radius:12px 12px 0 0;
        background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
        color:#E5EEF6;
        font-weight:600;
        }
        .stTabs [role="tab"]:hover{
        filter:brightness(1.06);
        border-color:#3a3f49;
        }
        .stTabs [role="tab"][aria-selected="true"]{
        background:linear-gradient(180deg, rgba(29,167,225,.12), rgba(29,167,225,.06));
        color:#F5FBFF;
        border-color:#3a3f49;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
        }

        /* Tab panel surface aligns with our cards */
        .stTabs [role="tabpanel"]{
        background:var(--skw-card);
        border:1px solid var(--skw-border);
        border-top:none;
        border-radius:0 12px 12px 12px;
        padding:16px;
        }


        /* Analysis actions (always visible above tabs) */
        .skw-actions{
        display:flex; gap:10px; align-items:center; margin:6px 0 10px 0;
        }
        .skw-actions .stButton{ display:inline-block; }             /* make buttons sit inline */
        .skw-actions .stButton > button{
        width:auto !important; min-height:36px !important;         /* compact */
        padding:8px 12px !important; border-radius:10px;
        }
        /* Ghost/secondary style for Reset so it doesn't overpower primary CTAs */
        .skw-actions .reset-btn > button{
        background:rgba(255,255,255,0.02) !important;
        color:#E5EEF6 !important;
        border:1px solid var(--skw-border) !important;
        }
        .skw-actions .reset-btn > button:hover{
        filter:brightness(1.06); border-color:#3a3f49 !important;
        }

        /* Use the same compact nav row as other pages */
        .skw-nav{ margin-top:14px; }
        .skw-nav .stButton > button{
        width:auto !important; padding:8px 14px !important; min-height:38px !important; border-radius:10px;
        }
        /* Reset should be a ghost/secondary button on the right */
        .skw-nav .reset-btn > button{
        background:rgba(255,255,255,.02) !important;
        color:#E5EEF6 !important;
        border:1px solid var(--skw-border) !important;
        }
        .skw-nav .reset-btn > button:hover{
        filter:brightness(1.06); border-color:#3a3f49 !important;
}

        </style>
        """,
        unsafe_allow_html=True,
    )



# ---------------- Header (for non-landing pages) ----------------
def header_logo(path: str | None):
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:4px 0 10px 0">', unsafe_allow_html=True)
    if path and Path(path).exists():
        st.image(path, width=112)
    else:
        st.markdown("### EssKay Wealth")
    st.markdown("</div><hr style='border:0;height:1px;background:#2A2D35;margin:8px 0 12px 0'/>", unsafe_allow_html=True)

# ---------------- Linear-like hero ----------------
def hero_linear(title: str, subtitle: str, badge: str = "EssKay • Modern portfolio analytics"):
    st.markdown(f"""
      <div class="skw-hero-wrap">
        <div>
          <div class="skw-hero-badge">{badge}</div>
          <div class="skw-h1" style="margin-top:10px;">{title}</div>
          <div class="skw-sub" style="margin-top:10px;">{subtitle}</div>
        </div>
        <div class="skw-hero-art"></div>
      </div>
    """, unsafe_allow_html=True)

# ---------------- Sticky footer ----------------
def sticky_footer(text: str):
    st.markdown(f'<div class="skw-footer-fixed"><span style="color:#A0A0A0">{text}</span></div>', unsafe_allow_html=True)

# ---------------- Helpers you already used ----------------
def card_start(title: str | None = None):
    st.markdown('<div class="skw-tile">', unsafe_allow_html=True)
    if title: st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def sleek_table(df: pd.DataFrame, height: int | None = 360, config: dict | None = None):
    return st.dataframe(df, use_container_width=True, hide_index=True, height=height, column_config=config or {})

def pie_grid(figs: Iterable[go.Figure]):
    figs = [f for f in figs if f is not None]
    if not figs: return
    if len(figs) == 1:
        st.plotly_chart(figs[0], use_container_width=True); return
    cols = st.columns(len(figs))
    for c, f in zip(cols, figs):
        with c: st.plotly_chart(f, use_container_width=True)

# icons (used only in markup as text right now)
def svg_analytics(fill="#FAFAFA"): return "📊"
def svg_benchmark(fill="#FAFAFA"): return "📈"

import base64, os

def hero_linear(title: str, subtitle: str, badge: str = "EssKay • Modern portfolio analytics", hero_image: str | None = None):
    """
    If hero_image is provided and exists (e.g., 'assets/hero.png' or '.svg'),
    we inline it as a CSS data URL so Streamlit can render it reliably.
    """
    css_img = ""
    if hero_image and os.path.exists(hero_image):
        ext = os.path.splitext(hero_image)[1].lower()
        mime = "image/svg+xml" if ext == ".svg" else f"image/{ext.strip('.')}"
        with open(hero_image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        css_img = f"""
        <style>
          .skw-hero-art {{
            background-image: url('data:{mime};base64,{b64}');
          }}
        </style>
        """

    st.markdown(f"""
      {css_img}
      <div class="skw-hero-wrap">
        <div>
          <div class="skw-hero-badge">{badge}</div>
          <div class="skw-h1" style="margin-top:10px;">{title}</div>
          <div class="skw-sub" style="margin-top:10px;">{subtitle}</div>
        </div>
        <div class="skw-hero-art"></div>
      </div>
    """, unsafe_allow_html=True)


def page_head(title: str, subtitle: str, icon: str = "📊"):
    st.markdown(
        f"""
        <div class="skw-pagehead">
          <div class="icon">{icon}</div>
          <div>
            <div class="title">{title}</div>
            <div class="sub">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )