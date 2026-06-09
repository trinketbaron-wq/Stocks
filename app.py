"""
ALPHAWIRE — interactive multi-stock screener (Streamlit)
========================================================
© 2026 7562609 Manitoba Inc. All rights reserved.
Up to 5 tickers -> one composite AlphaRank score each (0-100) that maps to a
BUY / HOLD / SELL verdict (green / amber / red). Every indicator casts points
into that score, including the VIX. A dense color-coded matrix compares all
tickers; tap any indicator row to blow up a detail chart. Each stock also gets
a price chart with the historical BUY/SELL signals marked over time.

Run:     streamlit run app.py        (after: pip install -r requirements.txt)
Deploy:  push app.py + requirements.txt (+ .streamlit/config.toml) to GitHub
         -> share.streamlit.io

Scoring note: the HISTORICAL signal line uses the technical indicators + VIX
(both have daily history). News sentiment has no clean per-day history, so it
only nudges the LIVE verdict, not the backtested markers. This is analysis of
current/!past conditions — descriptive, not a prediction, not advice.
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import json as _json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================================================
# THEME
# ==========================================================================
BG="#0a0e14"; PANEL="#121a26"; GRID="rgba(255,255,255,0.05)"
TXT="#e6edf3"; MUTE="#a7b2c0"
GREEN="#16c784"; RED="#ea3943"; AMBER="#f5a623"; CYAN="#3ec1d3"
TAB_PALETTE=["#16c784","#4d8bf0","#f5a623","#ea3943","#a78bfa"]  # fixed distinct color per tab position

st.set_page_config(page_title="AlphaWire", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp {{ background:
    radial-gradient(1200px 600px at 80% -10%, rgba(62,193,211,0.08), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(22,199,132,0.07), transparent 55%),
    {BG}; color:{TXT}; }}
/* ---- SIDEBAR forced dark (Streamlit defaults it to light → invisible labels) ---- */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{{ background:{BG} !important; }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{{ color:{TXT} !important; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{{ color:#b8c2cf !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"]{{
  background:#0d1622 !important; border:1px solid rgba(255,255,255,0.12) !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{{ color:{TXT} !important; }}
html,body,[class*="css"] {{ font-family:'IBM Plex Mono',monospace; }}
h1,h2,h3,h4,.deck-tkr {{ font-family:'Chakra Petch',sans-serif; letter-spacing:.02em; }}
/* hero */
.hero {{ position:relative; padding:22px 26px; border-radius:16px; overflow:hidden;
    border:1px solid rgba(255,255,255,0.08);
    background:linear-gradient(120deg,#0d1420 0%,#101b2b 50%,#0c1622 100%); margin-bottom:6px; }}
.hero:before {{ content:""; position:absolute; inset:0;
    background:repeating-linear-gradient(0deg,transparent 0 22px,rgba(255,255,255,0.025) 22px 23px);
    pointer-events:none; }}
.hero h1 {{ margin:0; font-size:38px; font-weight:700;
    background:linear-gradient(90deg,{GREEN},{CYAN} 60%,{AMBER});
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }}
.hero .blip {{ display:inline-block; width:10px; height:10px; border-radius:50%;
    background:{GREEN}; box-shadow:0 0 12px {GREEN}; margin-right:10px;
    animation:pulse 1.6s infinite ease-in-out; vertical-align:middle; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(.7)}} }}
.hero p {{ color:{MUTE}; margin:6px 0 0; font-size:13px; }}
/* score cards */
.cardwrap {{ display:flex; gap:14px; flex-wrap:wrap; margin:10px 0 4px; }}
.card {{ flex:1; min-width:150px; background:{PANEL}; border-radius:14px; padding:16px 16px 14px;
    border:1px solid rgba(255,255,255,0.07); position:relative;
    box-shadow:0 10px 30px rgba(0,0,0,0.35); animation:rise .5s ease both; }}
@keyframes rise {{ from{{opacity:0;transform:translateY(12px)}} to{{opacity:1;transform:none}} }}
.card .deck-tkr {{ font-size:22px; font-weight:700; }}
.card .verdict {{ float:right; font-family:'Chakra Petch'; font-weight:700; font-size:13px;
    padding:3px 10px; border-radius:999px; letter-spacing:.08em; }}
.card .num {{ font-size:40px; font-weight:600; line-height:1.1; margin-top:6px; }}
.card .lab {{ color:{MUTE}; font-size:11px; text-transform:uppercase; letter-spacing:.12em; }}
.gauge {{ height:9px; border-radius:6px; margin-top:12px; position:relative;
    background:linear-gradient(90deg,{RED} 0%,{AMBER} 50%,{GREEN} 100%); opacity:.85; }}
.gauge .tick {{ position:absolute; top:-4px; width:3px; height:17px; border-radius:2px;
    background:#fff; box-shadow:0 0 8px rgba(255,255,255,0.8); }}
.subnote {{ color:{MUTE}; font-size:11px; margin-top:8px; }}
.cardsub {{ color:{MUTE}; font-size:11px; margin-top:9px; line-height:1.5; }}
.fgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(116px,1fr)); gap:10px; margin:4px 0 14px; }}
.fstat {{ background:{PANEL}; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:9px 12px; }}
.fstat .k {{ color:{MUTE}; font-size:10px; letter-spacing:.1em; text-transform:uppercase; }}
.fstat .v {{ font-family:'Chakra Petch',sans-serif; font-size:18px; color:{TXT}; margin-top:3px; }}
hr {{ border-color:rgba(255,255,255,0.08); }}
[data-testid="stMetricValue"] {{ font-family:'Chakra Petch'; }}
</style>
""", unsafe_allow_html=True)

# Force dark, readable form controls even if the dark theme config isn't applied
st.markdown("""
<style>
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="input"], [data-baseweb="base-input"]{ background:#0d1622 !important; }
[data-baseweb="input"] button, [data-baseweb="input"] [role="button"]{ background:#0d1622 !important; }
[data-baseweb="input"] button svg{ fill:#9aa4b2 !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input {
  background:#0d1622 !important; color:#e6edf3 !important; -webkit-text-fill-color:#e6edf3 !important;
  border:1px solid rgba(255,255,255,0.12) !important;
}
[data-baseweb="select"] > div, [data-baseweb="select"] div[role="button"] {
  background:#0d1622 !important; color:#e6edf3 !important; border-color:rgba(255,255,255,0.12) !important;
}
[data-baseweb="select"] svg { fill:#9aa4b2 !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], ul[role="listbox"] {
  background:#101b2b !important; color:#e6edf3 !important;
}
[role="option"], li[role="option"] { background:#101b2b !important; color:#e6edf3 !important; }
[role="option"]:hover, li[role="option"]:hover { background:#16233a !important; }
[data-baseweb="tag"] { background:rgba(22,199,132,0.22) !important; color:#e6edf3 !important; }
[data-baseweb="tag"] svg { fill:#e6edf3 !important; }
input::placeholder, textarea::placeholder { color:#7d8694 !important; }
/* ---- buttons: primary stays red; secondary forced dark & readable ---- */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:#ea3943 !important; color:#fff !important; border:0 !important;
}
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"]{
  background:#0d1622 !important; color:#e6edf3 !important;
  border:1px solid rgba(255,255,255,0.18) !important;
}
.stButton > button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover{
  border-color:#3ec1d3 !important; color:#fff !important;
}
/* ---- selection pills/chips: dark bg, light text; selected = green ---- */
[data-testid="stPills"] button, .stPills button, [data-baseweb="pill"],
[data-testid="stButtonGroup"] button, [role="group"] button[kind="pills"],
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"],
button[kind="pills"], button[kind="pillsActive"]{
  background:#0d1622 !important; color:#e6edf3 !important;
  border:1px solid rgba(255,255,255,0.22) !important;
}
[data-testid="stPills"] button *, [data-testid="stButtonGroup"] button *,
[data-testid="stBaseButton-pills"] *{ color:#e6edf3 !important; }
[data-testid="stPills"] button[aria-selected="true"],
[data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stBaseButton-pillsActive"], button[kind="pillsActive"]{
  background:rgba(234,57,67,0.20) !important; border-color:#ea3943 !important;
}
[data-testid="stBaseButton-pillsActive"], [data-testid="stBaseButton-pillsActive"] *,
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"] *,
button[kind="pillsActive"], button[kind="pillsActive"] *{ color:#ff6b73 !important; }
/* ---- in-sidebar CLOSE arrow: keep light, it sits on the dark sidebar ---- */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarHeader"] button, [data-testid="stSidebarHeader"] button *,
[data-testid="baseButton-headerNoPadding"], [data-testid="baseButton-headerNoPadding"] *,
button[kind="headerNoPadding"], button[kind="headerNoPadding"] *{
  color:#e6edf3 !important; fill:#e6edf3 !important;
  opacity:1 !important; visibility:visible !important;
}
[data-testid="stSidebarCollapseButton"]{ background:rgba(255,255,255,0.08) !important; border-radius:8px !important; }
/* ---- OPEN-SETTINGS control (shown when sidebar is collapsed): BLACK arrow on a light pill, with a label ---- */
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"]{
  background:#e6edf3 !important; border-radius:10px !important; padding:6px 12px 6px 10px !important;
  display:flex !important; align-items:center !important; gap:8px !important; width:auto !important;
  box-shadow:0 4px 14px rgba(0,0,0,0.35) !important; border:1px solid rgba(0,0,0,0.15) !important;
}
[data-testid="collapsedControl"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="stExpandSidebarButton"] *{
  color:#0a0e14 !important; fill:#0a0e14 !important; opacity:1 !important; visibility:visible !important;
}
[data-testid="collapsedControl"]::after, [data-testid="stSidebarCollapsedControl"]::after, [data-testid="stExpandSidebarButton"]::after{
  content:"AlphaWire analysis settings"; color:#0a0e14; font-family:'Chakra Petch',sans-serif;
  font-weight:700; font-size:13px; letter-spacing:.02em; white-space:nowrap;
}
/* ---- screener modal: Streamlit's dialog uses a LIGHT background, so its light-themed labels/captions
       were invisible (white-on-white). Force them dark. Buttons + dark widgets are left alone. ---- */
div[data-testid="stDialog"] [data-testid="stCaptionContainer"],
div[data-testid="stDialog"] [data-testid="stCaptionContainer"] *,
div[data-testid="stDialog"] [data-testid="stWidgetLabel"],
div[data-testid="stDialog"] [data-testid="stWidgetLabel"] *,
div[data-testid="stDialog"] summary, div[data-testid="stDialog"] summary *,
div[data-testid="stDialog"] [data-testid="stThumbValue"],
div[data-testid="stDialog"] [data-testid="stTickBar"], div[data-testid="stDialog"] [data-testid="stTickBar"] *{
  color:#15202e !important;
}
/* ---- per-ticker TABS: large, bold, readable (default is tiny & dim) ---- */
[data-baseweb="tab-list"]{ gap:4px !important; flex-wrap:wrap !important; }
button[data-baseweb="tab"]{ padding:8px 16px !important; }
button[data-baseweb="tab"] p, button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p,
[data-baseweb="tab"] [data-testid="stMarkdownContainer"]{
  font-size:20px !important; font-weight:700 !important;
  font-family:'Chakra Petch',sans-serif !important; color:#c7d0db !important; letter-spacing:.02em !important;
}
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] div[data-testid="stMarkdownContainer"] p{
  color:#16c784 !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ background:#16c784 !important; height:3px !important; }
/* ---- universal readable text: labels, captions, body all near-white ---- */
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
.stMarkdown, .stMarkdown p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stRadio label, .stCheckbox label, .stToggle label, [data-baseweb="form-control-label"]{
  color:#e6edf3 !important;
}
small, .stCaption, [data-testid="stCaptionContainer"]{ color:#b8c2cf !important; }
/* ---- radio: dark, readable; SELECTED option turns green ---- */
.stRadio [role="radiogroup"] label{ color:#e6edf3 !important; }
.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label[aria-checked="true"]{ color:#16c784 !important; font-weight:600 !important; }
[data-baseweb="radio"] div:first-child{ border-color:#7f8a98 !important; }
[data-baseweb="radio"][aria-checked="true"] div:first-child,
[data-baseweb="radio"][data-checked="true"] div:first-child{
  border-color:#16c784 !important; background:#16c784 !important;
}
/* ---- toggle: readable label; track turns green when ON ---- */
.stToggle label, [data-testid="stToggle"] label{ color:#e6edf3 !important; font-weight:600 !important; }
[data-baseweb="checkbox"] [role="switch"]{ background:#39424f !important; }
[data-baseweb="checkbox"] [role="switch"][aria-checked="true"]{ background:#16c784 !important; }
[data-baseweb="checkbox"][aria-checked="true"] div[role="switch"]{ background:#16c784 !important; }
/* ---- selectbox / multiselect closed control: dark with light text ---- */
.stSelectbox div[data-baseweb="select"] *, .stMultiSelect div[data-baseweb="select"] *{ color:#e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_FONT = dict(family="IBM Plex Mono, monospace", color=TXT, size=12)
def dark(fig, h=420):
    fig.update_layout(height=h, template="plotly_dark", font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8,r=8,t=34,b=8), legend=dict(orientation="h",y=1.08,font=dict(size=11)),
        hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

# modebar config that turns on drawing tools (trend lines, boxes, freehand, erase)
PLOTLY_DRAW={"displaylogo":False,"scrollZoom":True,
    "modeBarButtonsToAdd":["drawline","drawopenpath","drawclosedpath","drawrect","drawcircle","eraseshape"]}
# INDICATOR MATH
# ==========================================================================
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _sign(s): return np.sign(s).fillna(0)
def _g(x, scale):
    """Graded vote in (-1,1): tanh saturates, so ±1 is the built-in per-indicator strength cap."""
    return np.tanh(np.asarray(x, dtype=float)/scale)

def enrich(df):
    o=df.copy(); c,h,l,v=o.Close,o.High,o.Low,o.Volume
    o["EMA50"],o["EMA200"]=ema(c,50),ema(c,200)
    d=c.diff(); g=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    ls=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    o["RSI"]=100-100/(1+g/ls.replace(0,np.nan))
    macd=ema(c,12)-ema(c,26); o["MACD"]=macd; o["MACD_signal"]=ema(macd,9); o["MACD_hist"]=macd-o["MACD_signal"]
    lo,hi=l.rolling(14).min(),h.rolling(14).max()
    o["Stoch_K"]=100*(c-lo)/(hi-lo).replace(0,np.nan); o["Stoch_D"]=o["Stoch_K"].rolling(3).mean()
    m=c.rolling(20).mean(); sd=c.rolling(20).std(); o["BB_up"],o["BB_mid"],o["BB_low"]=m+2*sd,m,m-2*sd
    o["PctB"]=(c-o["BB_low"])/(o["BB_up"]-o["BB_low"]).replace(0,np.nan)
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/14,adjust=False).mean()
    up,dn=h.diff(),-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=o.index)
    mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=o.index)
    pdi=100*pdm.ewm(alpha=1/14,adjust=False).mean()/a; mdi=100*mdm.ewm(alpha=1/14,adjust=False).mean()/a
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    o["ADX"]=dx.ewm(alpha=1/14,adjust=False).mean(); o["plus_DI"],o["minus_DI"]=pdi,mdi
    o["OBV"]=(np.sign(c.diff()).fillna(0)*v).cumsum()
    tp=(h+l+c)/3; rmf=tp*v
    pos=rmf.where(tp>tp.shift(),0.0); neg=rmf.where(tp<tp.shift(),0.0)
    o["MFI"]=100-100/(1+pos.rolling(14).sum()/neg.rolling(14).sum().replace(0,np.nan))
    o["ATR"]=a; o["ATRpct"]=a/c*100
    o["VolAvg20"]=v.rolling(20).mean()
    # --- expanded indicator set (graded into the composite) ---
    o["EMA20"]=ema(c,20)
    o["ROC20"]=(c/c.shift(20)-1)*100.0                         # momentum (20-day % change)
    hh,ll=h.rolling(14).max(),l.rolling(14).min()
    o["WilliamsR"]=-100*(hh-c)/(hh-ll).replace(0,np.nan)       # Williams %R (-100..0)
    tpc=(h+l+c)/3.0; smatp=tpc.rolling(20).mean()
    mdv=(tpc-smatp).abs().rolling(20).mean()
    o["CCI"]=(tpc-smatp)/(0.015*mdv.replace(0,np.nan))         # CCI (20)
    nar=25
    o["AroonUp"]=h.rolling(nar+1).apply(lambda x:100.0*np.argmax(x)/nar,raw=True)
    o["AroonDn"]=l.rolling(nar+1).apply(lambda x:100.0*np.argmin(x)/nar,raw=True)
    o["AroonOsc"]=o["AroonUp"]-o["AroonDn"]                    # Aroon oscillator (-100..100)
    o["Hi252"]=c.rolling(252,min_periods=20).max(); o["Lo252"]=c.rolling(252,min_periods=20).min()
    o["RangePos"]=(c-o["Lo252"])/(o["Hi252"]-o["Lo252"]).replace(0,np.nan)  # 52w range position 0..1
    return o

# ---- composite weighting (tunable: how much each indicator moves the score) ----
# ---- composite weighting: each indicator contributes a GRADED vote in [-1,1], scaled by its weight (= its cap) ----
WEIGHTS={"trend20":1.0,"trend50":1.25,"trend200":1.5,"cross":1.5,"ema50_slope":1.25,
         "macd_zero":1.0,"macd":1.25,"di":1.25,"aroon":1.0,"roc":1.25,
         "rsi":1.0,"stoch":1.0,"mfi":1.0,"williams":0.75,"cci":0.75,"boll":1.0,
         "obv":1.0,"rs":1.5,"range52":1.0,"vix":1.0,"news":1.0}
BUY_TH=0.18   # net bullish fraction (of max weight) needed to flip BUY / SELL (softer graded inputs)

def signal_frame(o, vix_aligned=None, spy_aligned=None):
    """Per-bar weighted votes -> composite, 0-100 strength, BUY/HOLD/SELL state.
    News is NOT included here (no daily history); it's added to the LIVE score
    in the main loop. Returns (composite, strength, state, votes_df, max_weight)."""
    c=o.Close
    V=pd.DataFrame(index=o.index)
    # trend / structure (trend-following, graded by distance)
    V["trend20"]=_g(c/o.EMA20-1, 0.04)
    V["trend50"]=_g(c/o.EMA50-1, 0.06)
    V["trend200"]=_g(c/o.EMA200-1, 0.12)
    V["cross"]=_g(o.EMA50/o.EMA200-1, 0.04)
    V["ema50_slope"]=_g(o.EMA50/o.EMA50.shift(20)-1, 0.05)
    # MACD
    V["macd_zero"]=_g(o.MACD/c, 0.012)                 # MACD line vs zero
    V["macd"]=_g(o.MACD_hist/c, 0.004)                 # MACD histogram momentum
    # directional / momentum
    V["di"]=_g(o.plus_DI-o.minus_DI, 18)
    V["aroon"]=_g(o.AroonOsc, 45)
    V["roc"]=_g(o.ROC20, 8)
    # oscillators (MEAN-REVERSION: overbought = caution/bearish, oversold = bullish)
    V["rsi"]=_g(50-o.RSI, 16)
    V["stoch"]=_g(50-o.Stoch_K, 22)
    V["mfi"]=_g(50-o.MFI, 20)
    V["williams"]=_g(-(o.WilliamsR+50), 22)
    V["cci"]=_g(-o.CCI, 110)
    V["boll"]=_g(0.5-o.PctB, 0.32)
    V["range52"]=_g(o.RangePos-0.5, 0.30)
    # volume
    obv_sl=o.OBV-o.OBV.shift(20)
    obv_dn=(o.OBV.diff().abs().rolling(60).mean()*20).replace(0,np.nan)
    V["obv"]=_g(obv_sl/obv_dn, 1.5)
    # relative strength vs market
    if spy_aligned is not None:
        ratio=c/spy_aligned.reindex(o.index).ffill()
        V["rs"]=_g(ratio/ratio.shift(21)-1, 0.05)
    else: V["rs"]=0.0
    # VIX — smooth risk tilt (calm = tailwind, fear scales the headwind)
    if vix_aligned is not None:
        lvl=vix_aligned.reindex(o.index).ffill()
        V["vix"]=_g(18.0-lvl.values, 8.0)
    else: V["vix"]=0.0
    V=V.fillna(0)
    hk=[k for k in WEIGHTS if k!="news"]
    w=pd.Series({k:WEIGHTS[k] for k in hk})
    composite=(V[hk]*w).sum(axis=1)
    max_w=float(w.abs().sum())
    r=composite/max_w
    strength=((r+1)/2*100).round()
    state=pd.Series(np.select([r>=BUY_TH,r<=-BUY_TH],["BUY","SELL"],"HOLD"),index=o.index)
    return composite,strength,state,V,max_w

def alternating_signals(state):
    """Buy fires only after a prior sell (and vice-versa) -> clean markers."""
    pos=0; buys=[]; sells=[]
    for i,s in enumerate(state.values):
        if s=="BUY" and pos!=1: buys.append(i); pos=1
        elif s=="SELL" and pos!=-1: sells.append(i); pos=-1
    return buys,sells

def trade_markers(pos):
    """Actual entries (flat->long, 0->1) and exits (long->flat, 1->0) of a backtest position series."""
    p=pos.fillna(0.0).values
    entries=[i for i in range(1,len(p)) if p[i]>0.5 and p[i-1]<=0.5]
    exits=[i for i in range(1,len(p)) if p[i]<=0.5 and p[i-1]>0.5]
    if len(p) and p[0]>0.5: entries=[0]+entries
    return entries,exits

def long_spans(pos):
    """Contiguous (start,end) positional ranges where the position is long, for shading."""
    p=pos.fillna(0.0).values; spans=[]; s=None
    for i,x in enumerate(p):
        if x>0.5 and s is None: s=i
        elif x<=0.5 and s is not None: spans.append((s,i-1)); s=None
    if s is not None: spans.append((s,len(p)-1))
    return spans

# ==========================================================================
# SENTIMENT + NEWS
# ==========================================================================
FIN={"beats":2.5,"beat":2,"crushes":3,"crushed":2.5,"tops":2,"topped":2,"surge":2.5,
 "surges":2.5,"soars":3,"soar":2.5,"soared":2.5,"rally":2,"rallies":2,"upgrade":2.5,
 "upgraded":2.5,"outperform":2.5,"bullish":3,"record":1.5,"jumps":2,"jump":1.5,"gains":1.5,
 "rebound":2,"raises":1.5,"raised":1.5,"approval":2,"misses":-2.5,"miss":-2,"missed":-2,
 "plunge":-3,"plunges":-3,"slumps":-2.5,"tumbles":-2.5,"downgrade":-2.5,"downgraded":-2.5,
 "lawsuit":-2,"probe":-1.5,"recall":-2,"bankruptcy":-3.5,"default":-2.5,"bearish":-3,
 "warns":-2,"warning":-1.5,"sinks":-2.5,"slides":-1.5,"drops":-1.5,"cuts":-1.5,"halts":-2}

@st.cache_resource
def analyzer():
    a=SentimentIntensityAnalyzer(); a.lexicon.update(FIN); return a
def vader(t): return analyzer().polarity_scores(t)["compound"]

def llm_sentiment(headlines, api_key):
    try:
        import anthropic, json, re
        cl=anthropic.Anthropic(api_key=api_key)
        numbered="\n".join(f"{i}. {h}" for i,h in enumerate(headlines))
        msg=cl.messages.create(model="claude-haiku-4-5",max_tokens=500,
            messages=[{"role":"user","content":
            "Score each financial headline for sentiment toward the company's stock "
            "from -1 (very negative) to 1 (very positive). Reply ONLY a JSON array of "
            "numbers in order.\n\n"+numbered}])
        txt="".join(b.text for b in msg.content if getattr(b,"type","")=="text")
        return [float(x) for x in json.loads(re.search(r"\[.*\]",txt,re.S).group())][:len(headlines)]
    except Exception:
        return [vader(h) for h in headlines]

def parse_news(item):
    c=item.get("content",item)
    title=c.get("title") or item.get("title") or ""
    pub=(c.get("provider",{}) or {}).get("displayName") or item.get("publisher") or ""
    url=(c.get("canonicalUrl",{}) or {}).get("url") or item.get("link") or ""
    when=c.get("pubDate") or item.get("providerPublishTime") or ""
    if isinstance(when,(int,float)): when=dt.datetime.fromtimestamp(when).strftime("%Y-%m-%d")
    elif isinstance(when,str) and "T" in when: when=when.split("T")[0]
    return {"title":title,"publisher":pub,"url":url,"when":when}

# ==========================================================================
# DATA (cached)
# ==========================================================================
@st.cache_data(ttl=900,show_spinner=False)
def get_hist(t,period):
    df=yf.Ticker(t).history(period=period,auto_adjust=True)
    return df[["Open","High","Low","Close","Volume"]] if not df.empty else None
@st.cache_data(ttl=900,show_spinner=False)
def get_news(t,n=8):
    try: return [parse_news(x) for x in (yf.Ticker(t).news or [])[:n]]
    except Exception: return []

def _finnhub_key():
    """Locate a Finnhub key from Streamlit secrets (top-level OR a nested [section]) or an
    environment variable, tolerant of common name variants and stray quotes/whitespace.
    Returns a clean key string, or '' if none is found."""
    NAMES={"FINNHUB_API_KEY","FINNHUB_KEY","FINNHUB_TOKEN","FINNHUB"}
    def _clean(v):
        try: return str(v).strip().strip('"').strip("'").strip()
        except Exception: return ""
    try:
        sec=st.secrets
        for k in list(sec.keys()):                 # top-level, case-insensitive
            if str(k).upper() in NAMES:
                c=_clean(sec[k])
                if c: return c
        for k in list(sec.keys()):                 # one level of nesting ([section] tables)
            v=sec[k]
            if hasattr(v,"keys"):
                for kk in list(v.keys()):
                    if str(kk).upper() in NAMES:
                        c=_clean(v[kk])
                        if c: return c
    except Exception:
        pass
    try:
        import os
        for n in NAMES:
            v=os.environ.get(n) or os.environ.get(n.lower())
            if v:
                c=_clean(v)
                if c: return c
    except Exception:
        pass
    return ""

@st.cache_data(ttl=900,show_spinner=False)
def get_finnhub_news(symbol, days=365):
    """Dated, company-specific news from Finnhub (needs FINNHUB_API_KEY in Secrets)."""
    key=_finnhub_key()
    if not key: return None
    import requests, datetime as _dt
    to=_dt.date.today(); frm=to-_dt.timedelta(days=days)
    try:
        r=requests.get("https://finnhub.io/api/v1/company-news",
            params={"symbol":symbol,"from":str(frm),"to":str(to),"token":key},timeout=12)
        raw=r.json()
    except Exception:
        return None
    if not isinstance(raw,list) or not raw: return None
    out=[]
    for a in raw:
        ts=a.get("datetime")
        when=_dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
        if a.get("headline"):
            out.append({"title":a["headline"],"source":a.get("source","") or "",
                "url":a.get("url","") or "","when":when,"dt":ts or 0,"summary":a.get("summary","") or ""})
    out.sort(key=lambda x:x["dt"],reverse=True)
    return out[:40] or None

def get_company_news(t, period_years=1):
    """Deep, staggered Finnhub history if a key is set (back `period_years`, ≥1yr); else Yahoo."""
    deep=get_finnhub_news_range(t, years=max(period_years,1))
    if deep: return deep,"Finnhub"
    yh=get_news(t)
    norm=[{"title":n.get("title",""),"source":n.get("publisher",""),"url":n.get("url",""),
           "when":n.get("when",""),"dt":0,"summary":""} for n in yh]
    return norm,"Yahoo"

def _naive_idx(o):
    idx=pd.to_datetime(o.index)
    return idx.tz_localize(None) if idx.tz is not None else idx

def price_move_after(o, when, horizon=3):
    """% price change over `horizon` trading days after a news date (None if out of range)."""
    if not when: return None
    idx=_naive_idx(o)
    pos=int(idx.searchsorted(pd.Timestamp(when)))
    if pos>=len(idx)-1: return None
    end=min(pos+horizon,len(idx)-1)
    return (float(o.Close.iloc[end])/float(o.Close.iloc[pos])-1)*100

# ================= MATERIAL-NEWS ENGINE (multi-year) =================
# Pull years of company news, keep only MATERIAL catalysts (A-filter), find the stock's biggest
# price moves (B-spine), then line the two up: when the stock jumped, was there real news?

PERIOD_YEARS={"1mo":0.1,"3mo":0.3,"6mo":0.5,"1y":1,"2y":2,"5y":5,"10y":10,"max":10}

# ordered most-important first; first match wins
MATERIAL_RULES=[
 ("Earnings",   ["earnings"," eps ","beats","beat estimates","misses","missed estimates","tops estimates","top estimates",
                 "quarter","quarterly","fiscal","reports ","results","revenue","record revenue","net income","profit",
                 "loss per share","earnings call","q1 2","q2 2","q3 2","q4 2","third-quarter","fourth-quarter",
                 "first-quarter","second-quarter"]),
 ("Guidance",   ["guidance","outlook","forecast","guide","raises fy","cuts fy","raises full","cuts full","expects","sees ",
                 "projects","warns","lowers","raised its","cut its","growth forecast","reaffirms guidance","cuts estimate"]),
 ("M&A",        ["acqui","merger","to buy","to purchase","buys ","buyout","takeover","to acquire","agrees to buy",
                 "stake in","combine with","tender offer","deal to"]),
 ("Analyst",    ["upgrade","downgrade","price target","raises target","cuts target","initiates","initiated","coverage",
                 "overweight","underweight","outperform","buy rating","sell rating","hold rating","reiterates","reaffirms",
                 "maintains","raised pt","cut pt"," pt ","analyst","estimates raised","estimates cut","raised to","cut to"]),
 ("Regulatory", ["fda","approval","approved","approves","clearance","granted","lawsuit","settlement","investigat",
                 "antitrust","probe"," fine ","penalty","recall","sec charges","subpoena","ruling","sues"]),
 ("Capital",    ["buyback","repurchase","authoriz","dividend","stock split","offering","raises $","convertible",
                 "debt offering","secondary","declares"]),
 ("Product",    ["launch","unveil","introduc","announces","partnership","collaborat","contract","awarded","wins ","secures",
                 "deal with","supply","design win","selected","expands","new chip","new product"]),
 ("Management", ["ceo","cfo","president","chief executive","resign","steps down","appoints","named ceo","executive","board",
                 "departure","interim","hires"]),
]

def classify_news(title, summary=""):
    text=" "+(str(title)+" "+str(summary)).lower()+" "
    for cat,kws in MATERIAL_RULES:
        if any(k in text for k in kws): return cat
    return None

def material_news(items):
    """Keep only items that match a material category; tag each with its category."""
    out=[]
    for it in items or []:
        cat=classify_news(it.get("title",""), it.get("summary",""))
        if cat:
            it=dict(it); it["category"]=cat; out.append(it)
    return out

@st.cache_data(ttl=3600,show_spinner=False)
def get_finnhub_news_range(symbol, years=5, per_call_days=120, cap=1500):
    """Paginate Finnhub company-news in ~quarterly windows back `years`, staggered across time.
    Free tier only serves ~1yr of history, so we stop after consecutive empty windows
    (the API has run dry) — that bounds calls and respects the 60/min rate limit."""
    key=_finnhub_key()
    if not key: return None
    import requests, datetime as _dt, time as _time
    end=_dt.date.today(); start_limit=end-_dt.timedelta(days=int(max(years,0.1)*365))
    out={}; cur_to=end; calls=0; empty_streak=0
    while cur_to>start_limit and calls<26:
        cur_from=max(start_limit, cur_to-_dt.timedelta(days=per_call_days)); calls+=1
        got=0
        try:
            r=requests.get("https://finnhub.io/api/v1/company-news",
                params={"symbol":symbol,"from":str(cur_from),"to":str(cur_to),"token":key},timeout=15)
            raw=r.json() if r.status_code==200 else []
        except Exception:
            raw=[]
        if isinstance(raw,list):
            for a in raw:
                ts=a.get("datetime"); hl=a.get("headline")
                if not hl or not ts: continue
                when=_dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                k2=(when,hl[:60])
                if k2 in out: continue
                out[k2]={"title":hl,"source":a.get("source","") or "","url":a.get("url","") or "",
                         "when":when,"dt":ts,"summary":a.get("summary","") or ""}; got+=1
        empty_streak = empty_streak+1 if got==0 else 0
        if empty_streak>=2: break          # API has no more history this far back — stop
        cur_to=cur_from-_dt.timedelta(days=1)
        if len(out)>=cap: break
        _time.sleep(0.2)                     # gentle on the 60/min free-tier limit
    items=sorted(out.values(),key=lambda x:x["dt"],reverse=True)
    return items or None

def find_big_moves(o, window=3, top_n=14, min_pct=6.0, sep=8):
    """B-SPINE: the stock's largest ~window-day moves, de-duplicated, chronological."""
    idx=_naive_idx(o); c=o["Close"].values; n=len(c)
    cand=[]
    for i in range(window,n):
        prev=c[i-window]
        if prev>0: cand.append((abs(c[i]/prev-1)*100, i, (c[i]/prev-1)*100))
    cand.sort(reverse=True)
    used=[]; chosen=[]
    for ar,i,r in cand:
        if ar<min_pct: break
        if any(abs(i-j)<sep for j in used): continue
        used.append(i); chosen.append((idx[i],r,i))
        if len(chosen)>=top_n: break
    chosen.sort(key=lambda x:x[0])
    return chosen

def correlate_moves_news(moves, material, window=4):
    """For each big move, attach the nearest material headline within ±window days (if any)."""
    md=[]
    for m in material:
        if m.get("when"):
            try: md.append((pd.Timestamp(m["when"]),m))
            except Exception: pass
    rows=[]
    for ts,pct,ipos in moves:
        near=[(abs((d-ts).days),d,it) for d,it in md if abs((d-ts).days)<=window]
        near.sort(key=lambda x:x[0])
        best=near[0][2] if near else None
        rows.append(dict(date=str(ts)[:10],pct=pct,up=pct>=0,had=bool(near),
            category=(best.get("category") if best else None),
            headline=(best.get("title") if best else None),
            url=(best.get("url") if best else ""),n_near=len(near)))
    return rows

def news_keyword_stats(o, material, fwd=5):
    """For each material category, average price move over `fwd` days AFTER the news fired."""
    from collections import defaultdict
    agg=defaultdict(list)
    for it in material:
        cat=it.get("category")
        if not cat or not it.get("when"): continue
        mv=price_move_after(o,it["when"],horizon=fwd)
        if mv is not None: agg[cat].append(mv)
    stats=[]
    for cat,vals in agg.items():
        if not vals: continue
        arr=np.array(vals)
        stats.append(dict(category=cat,n=len(vals),avg=float(arr.mean()),
            pos=float((arr>0).mean()*100),best=float(arr.max()),worst=float(arr.min())))
    stats.sort(key=lambda s:abs(s["avg"]),reverse=True)
    return stats

def big_moves_table_html(rows, fwd_label="move"):
    import html as _h
    TD=f"padding:6px 9px;border-bottom:1px solid {GRID};vertical-align:top"
    TH=f"padding:7px 9px;color:{MUTE};font-size:10px;letter-spacing:.04em;text-align:left;border-bottom:1px solid {GRID}"
    body=[]
    for r in rows[::-1]:   # most recent first
        mc=GREEN if r["up"] else RED; arrow="▲" if r["up"] else "▼"
        if r["headline"]:
            cat=f"<span style='color:{CYAN};font-size:10px'>{r['category']}</span>"
            hl=_h.escape(r["headline"][:120])
            link=f"<a href='{r['url']}' target='_blank' style='color:{TXT};text-decoration:none'>{hl}</a>" if r["url"] else hl
            news=f"{cat}<br>{link}"+(f"<span style='color:{MUTE};font-size:10px'> · +{r['n_near']-1} more</span>" if r["n_near"]>1 else "")
        else:
            news=f"<span style='color:{MUTE}'>— no material news within ±4 days —</span>"
        body.append(f"<tr><td style='{TD};color:{MUTE};white-space:nowrap'>{r['date']}</td>"
                    f"<td style='{TD};color:{mc};font-weight:700;text-align:right;white-space:nowrap'>{arrow} {r['pct']:+.1f}%</td>"
                    f"<td style='{TD}'>{news}</td></tr>")
    head=(f"<tr><th style='{TH}'>Date</th><th style='{TH};text-align:right'>Big {fwd_label}</th>"
          f"<th style='{TH}'>Material news at that time</th></tr>")
    return (f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"Chakra Petch\",sans-serif;font-size:12px'>{head}{''.join(body)}</table></div>")

def keyword_stats_html(stats, fwd=5):
    TD=f"padding:6px 9px;border-bottom:1px solid {GRID};text-align:right"
    TH=f"padding:7px 9px;color:{MUTE};font-size:10px;text-align:right;border-bottom:1px solid {GRID}"
    body=[]
    for s in stats:
        ac=GREEN if s["avg"]>=0 else RED
        body.append(f"<tr><td style='{TD};text-align:left;color:{TXT}'>{s['category']}</td>"
                    f"<td style='{TD};color:{MUTE}'>{s['n']}</td>"
                    f"<td style='{TD};color:{ac};font-weight:700'>{s['avg']:+.1f}%</td>"
                    f"<td style='{TD};color:{MUTE}'>{s['pos']:.0f}%</td>"
                    f"<td style='{TD};color:{GREEN}'>{s['best']:+.0f}%</td>"
                    f"<td style='{TD};color:{RED}'>{s['worst']:+.0f}%</td></tr>")
    head=(f"<tr><th style='{TH};text-align:left'>Catalyst type</th><th style='{TH}'>Events</th>"
          f"<th style='{TH}'>Avg {fwd}d move after</th><th style='{TH}'>% up</th>"
          f"<th style='{TH}'>Best</th><th style='{TH}'>Worst</th></tr>")
    return (f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"Chakra Petch\",sans-serif;font-size:12.5px'>{head}{''.join(body)}</table></div>")

# ================= AWN — AlphaWire News indicator =================
# A NEWS-DRIVEN signal (like RSI/MACD, but built from catalysts). For each material event it
# fires an impulse = the CAUSAL historical average move that this stock made after PRIOR, already-
# resolved events of the same catalyst type (no look-ahead). Impulses then decay continuously
# (option B) with a tunable half-life, so a fresh beat reads stronger than a stale one.
def awn_series(o, material, half_life=5, fwd=5, scale=8.0):
    idx=_naive_idx(o); n=len(o); c=o["Close"].values
    evs=[]
    for it in material or []:
        if not it.get("when"): continue
        pos=int(idx.searchsorted(pd.Timestamp(it["when"])))
        if pos>=n: pos=n-1
        evs.append((pos,it.get("category"),it.get("title","")))
    evs.sort()
    def fwd_move(p):
        e=min(p+fwd,n-1)
        return (c[e]/c[p]-1)*100 if (c[p]>0 and e>p) else 0.0
    from collections import defaultdict
    hist=defaultdict(list)             # category -> list of (resolve_index, realized move)
    impulse=np.zeros(n)
    for p,cat,title in evs:
        prior=[m for (ri,m) in hist[cat] if ri<=p]      # only events whose fwd window CLOSED by p
        imp=float(np.mean(prior)) if prior else 0.3*vader(title)   # fallback before a track record
        impulse[p]+=imp
        hist[cat].append((p+fwd,fwd_move(p)))
    decay=0.5**(1.0/max(half_life,1)); acc=0.0; awn=np.zeros(n)
    for i in range(n):
        acc=acc*decay+impulse[i]; awn[i]=acc
    awn=pd.Series(awn,index=o.index)
    awn_long=(awn>0).astype(float)
    awn_score=pd.Series(100*np.tanh(awn/scale),index=o.index)
    return awn, awn_long, awn_score

def awn_latest(material):
    """Most recent material catalyst (for the card readout). Robust to dt=0 (Yahoo) via 'when'."""
    md=[m for m in (material or []) if (m.get("when") or m.get("dt"))]
    if not md: return None
    def _key(m):
        if m.get("dt"): return float(m["dt"])
        try: return pd.Timestamp(m["when"]).timestamp()
        except Exception: return 0.0
    return max(md,key=_key)

def news_table_html(items, scores, moves):
    import html as _h
    rows=[]
    for it,sc,mv in zip(items,scores,moves):
        tag=("🟢 positive" if sc>0.05 else "🔴 negative" if sc<-0.05 else "⚪ neutral")
        tcol=(GREEN if sc>0.05 else RED if sc<-0.05 else MUTE)
        if mv is None:
            react=f"<span style='color:{MUTE}'>price reaction n/a (too recent)</span>"
        else:
            mc=GREEN if mv>=0 else RED
            rel=("aligned with the news" if (sc>0.05 and mv>0) or (sc<-0.05 and mv<0)
                 else "moved against the news" if abs(sc)>0.05 and ((sc>0)!=(mv>0))
                 else "little directional link")
            react=f"<span style='color:{mc}'>{mv:+.1f}% over next 3 sessions</span> · {rel}"
        title=_h.escape(it["title"]); src=_h.escape(it.get("source") or "")
        meta=f"{src} · {it.get('when','')}"+(f" · <a href='{_h.escape(it['url'])}' style='color:{CYAN}'>open</a>" if it.get("url") else "")
        left=f"<div style='font-size:13px;color:{TXT};line-height:1.35'>{title}</div><div style='color:{MUTE};font-size:11px;margin-top:3px'>{meta}</div>"
        right=f"<div style='color:{tcol};font-size:12px;font-weight:600'>{tag}</div><div style='color:{MUTE};font-size:12px;margin-top:3px'>{react}</div>"
        bd="border-bottom:1px solid rgba(255,255,255,0.06);padding:9px 10px;vertical-align:top"
        rows.append(f"<tr><td style='{bd};width:56%'>{left}</td><td style='{bd}'>{right}</td></tr>")
    head=(f"<th style='text-align:left;color:{MUTE};font-size:11px;letter-spacing:.08em;padding:6px 10px'>HEADLINE</th>"
          f"<th style='text-align:left;color:{MUTE};font-size:11px;letter-spacing:.08em;padding:6px 10px'>AI SENTIMENT &amp; PRICE REACTION</th>")
    return f"<table style='width:100%;border-collapse:collapse'><tr>{head}</tr>{''.join(rows)}</table>"

def news_reaction_summary(scores, moves):
    pairs=[(s,m) for s,m in zip(scores,moves) if m is not None]
    if len(pairs)<3: return None
    pos=[m for s,m in pairs if s>0.05]; neg=[m for s,m in pairs if s<-0.05]
    parts=[]
    if pos: parts.append(f"positive headlines were followed by an average **{np.mean(pos):+.1f}%** over the next 3 sessions (n={len(pos)})")
    if neg: parts.append(f"negative headlines by **{np.mean(neg):+.1f}%** (n={len(neg)})")
    if not parts: return None
    body="Across the available window, "+"; ".join(parts)+"."
    if pos and neg:
        if abs(np.mean(neg))>abs(np.mean(pos))*1.3: body+=" This stock has reacted more sharply to **negative** news."
        elif abs(np.mean(pos))>abs(np.mean(neg))*1.3: body+=" **Positive** news has tended to move it more."
    return body
@st.cache_data(ttl=900,show_spinner=False)
def get_vix_hist(period):
    try:
        v=yf.Ticker("^VIX").history(period=period,auto_adjust=True)
        return v["Close"] if not v.empty else None
    except Exception: return None
@st.cache_data(ttl=900,show_spinner=False)
def get_spy(period):
    try:
        s=yf.Ticker("SPY").history(period=period,auto_adjust=True)
        return s["Close"] if not s.empty else None
    except Exception: return None
@st.cache_data(ttl=3600,show_spinner=False)
def get_fundamentals(t):
    info={}; tk=None
    try: tk=yf.Ticker(t)
    except Exception: tk=None
    try: info=(tk.info if tk else {}) or {}
    except Exception: info={}
    fi={}
    if tk is not None:
        try:
            fobj=tk.fast_info
            for kk in ["market_cap","year_high","year_low","last_volume",
                       "ten_day_average_volume","three_month_average_volume"]:
                try: fi[kk]=fobj[kk]
                except Exception:
                    try: fi[kk]=getattr(fobj,kk)
                    except Exception: pass
        except Exception: pass
    def g(*keys):
        for k in keys:
            x=info.get(k)
            if x not in (None,"",0): return x
        return None
    return dict(name=g("shortName","longName") or t, sector=g("sector"),
        industry=g("industry"),
        market_cap=g("marketCap") or fi.get("market_cap"),
        pe=g("trailingPE"), fpe=g("forwardPE"), eps=g("trailingEps"),
        div=g("dividendYield"), beta=g("beta"), pb=g("priceToBook"),
        margin=g("profitMargins"),
        hi52=g("fiftyTwoWeekHigh") or fi.get("year_high"),
        lo52=g("fiftyTwoWeekLow") or fi.get("year_low"),
        avg_vol=g("averageVolume","averageDailyVolume10Day")
                or fi.get("ten_day_average_volume") or fi.get("three_month_average_volume"),
        volume=g("volume","regularMarketVolume") or fi.get("last_volume"))

def _extract_fin(stmt):
    """Pull revenue + net income series out of a yfinance income statement frame."""
    if stmt is None or getattr(stmt,"empty",True): return None
    def row(*names):
        for n in names:
            if n in stmt.index: return stmt.loc[n]
        return None
    rev=row("Total Revenue","TotalRevenue")
    ni=row("Net Income","NetIncome","Net Income Common Stockholders")
    if rev is None and ni is None: return None
    cols=list(stmt.columns)
    def val(s,c): return float(s[c]) if (s is not None and pd.notna(s[c])) else np.nan
    df=pd.DataFrame({"period":[pd.Timestamp(c).strftime("%Y-%m") for c in cols],
        "revenue":[val(rev,c) for c in cols],"net_income":[val(ni,c) for c in cols]})
    return df.iloc[::-1].reset_index(drop=True)   # oldest -> newest

@st.cache_data(ttl=3600,show_spinner=False)
def get_financials(t):
    out={"annual":None,"quarterly":None}
    try:
        tk=yf.Ticker(t)
        out["annual"]=_extract_fin(tk.income_stmt)
        out["quarterly"]=_extract_fin(tk.quarterly_income_stmt)
    except Exception: pass
    return out

def human(n):
    if n is None: return "—"
    n=float(n)
    for u,dv in (("T",1e12),("B",1e9),("M",1e6),("K",1e3)):
        if abs(n)>=dv: return f"{n/dv:.2f}{u}"
    return f"{n:.0f}"
def money(n): return "—" if n is None else "$"+human(n)
def fnum(x,suf=""): return "—" if x is None else f"{float(x):.2f}{suf}"
def fpct(x):
    if x is None: return "—"
    x=float(x); x=x*100 if abs(x)<1 else x   # yfinance sometimes fraction, sometimes %
    return f"{x:.2f}%"

def fundamentals_grid(f, accent=CYAN):
    cells=[("Market Cap",money(f["market_cap"])),("P/E (TTM)",fnum(f["pe"])),
           ("Fwd P/E",fnum(f["fpe"])),("EPS",fnum(f["eps"])),
           ("Div Yield",fpct(f["div"])),("Beta",fnum(f["beta"])),
           ("P/B",fnum(f["pb"])),("Profit Margin",fpct(f["margin"])),
           ("52W High",fnum(f["hi52"])),("52W Low",fnum(f["lo52"])),
           ("Avg Vol",human(f["avg_vol"])),("Volume",human(f["volume"]))]
    inner="".join(f'<div class="fstat" style="border-left:3px solid {accent}">'
                  f'<div class="k">{k}</div><div class="v">{v}</div></div>' for k,v in cells)
    return f'<div class="fgrid">{inner}</div>'

# ==========================================================================
# MARKET MOVERS  (curated large-cap universes -> ranked by today's move)
# Note: full-exchange (e.g. all of NYSE) movers aren't freely available, so we
# use the major indices. Lists are curated large-caps; edit to taste.
# ==========================================================================

# ---- index universes for the movers picker (curated snapshots; misses fail gracefully) ----
DOW30=["AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON",
 "IBM","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","NVDA","PG","TRV","UNH","V","VZ","WMT"]
SP100=["AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC",
 "BK","BKNG","BLK","BMY","BRK-B","C","CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO",
 "CVS","CVX","DHR","DIS","DOW","DUK","EMR","F","FDX","GD","GE","GILD","GM","GOOG","GOOGL","GS",
 "HD","HON","IBM","INTC","INTU","ISRG","JNJ","JPM","KHC","KO","LIN","LLY","LMT","LOW","MA","MCD",
 "MDLZ","MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP",
 "PFE","PG","PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","T","TGT","TMO","TMUS","TSLA","TXN","UNH",
 "UNP","UPS","USB","V","VZ","WFC","WMT","XOM"]
TSX60=["RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO","ENB.TO","TRP.TO","CNQ.TO","SU.TO","CVE.TO",
 "IMO.TO","CNR.TO","CP.TO","SHOP.TO","ATD.TO","BCE.TO","T.TO","RCI-B.TO","MFC.TO","SLF.TO","GWO.TO",
 "IFC.TO","TRI.TO","BN.TO","BAM.TO","WCN.TO","NTR.TO","AEM.TO","FNV.TO","WPM.TO","ABX.TO","CCO.TO",
 "L.TO","MRU.TO","FTS.TO","EMA.TO","H.TO","POW.TO","PPL.TO","KEY.TO","QSR.TO","DOL.TO","CSU.TO",
 "OTEX.TO","WSP.TO","STN.TO","MG.TO","TECK-B.TO","FM.TO","BIP-UN.TO","BEP-UN.TO"]
NASDAQ100=["AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","AVGO","TSLA","COST","NFLX",
 "TMUS","CSCO","PEP","AMD","ADBE","TXN","QCOM","INTU","AMGN","AMAT","ISRG","BKNG","HON",
 "VRTX","ADP","REGN","MU","LRCX","PANW","ADI","KLAC","SBUX","GILD","MDLZ","SNPS","CDNS",
 "MELI","CRWD","MAR","CTAS","ORLY","ABNB","MRVL","PYPL","FTNT","DASH","ASML","ADSK","NXPI",
 "PCAR","ROP","MNST","CPRT","WDAY","AEP","PAYX","ROST","KDP","ODFL","FAST","CHTR","CCEP",
 "FANG","EXC","VRSK","EA","KHC","CSGP","DDOG","XEL","GEHC","CTSH","TTWO","IDXX","ANSS",
 "DXCM","BKR","ON","CSX","BIIB","GFS","MCHP","CDW","TTD","WBD","ZS","ARM","MDB","TEAM",
 "SMCI","LULU","PDD","INTC","AZN"]
UNIVERSES={"Dow Jones 30":DOW30,"S&P 100":SP100,"Nasdaq 100":NASDAQ100,"TSX 60":TSX60}

@st.cache_data(ttl=600,show_spinner=False)
def fetch_movers(universe_name):
    """Rank an index's constituents by their latest 1-day % move. Returns DataFrame."""
    tickers=UNIVERSES[universe_name]
    try:
        df=yf.download(tickers,period="5d",auto_adjust=True,progress=False,group_by="column",threads=True)
    except Exception:
        return None
    if df is None or df.empty: return None
    close=df["Close"] if "Close" in df.columns.get_level_values(0) else df
    if isinstance(close,pd.Series): close=close.to_frame()
    close=close.dropna(how="all")
    if len(close)<2: return None
    chg=(close.iloc[-1]/close.iloc[-2]-1)*100
    vol=None
    if "Volume" in df.columns.get_level_values(0):
        v=df["Volume"]
        if isinstance(v,pd.Series): v=v.to_frame()
        vol=v.reindex(columns=close.columns).iloc[-1]
    out=pd.DataFrame({"ticker":chg.index,"chg":chg.values,"price":close.iloc[-1].values,
        "volume":(vol.values if vol is not None else np.nan)}).dropna(subset=["chg","price"])
    out=out.reindex(out["chg"].abs().sort_values(ascending=False).index)  # biggest movers first
    return out.reset_index(drop=True)

# ---- FULL SCREENER: score every constituent with AlphaRank + an indicator snapshot ----
# Tier-1 scan = technical composite only (no per-name news / fundamentals calls — those run
# later, only on the handful you load into the deck). Keeps a 100-name scan inside the data budget.
def _scan_row(t, sub, vix=None, spy=None):
    """Score one ticker's OHLCV frame into a screener row (or None if too little history)."""
    try:
        sub=sub.dropna(how="all")
        if not all(col in sub.columns for col in ("Open","High","Low","Close","Volume")): return None
        if len(sub)<60: return None
        o=enrich(sub); comp,strength,state,V,mw=signal_frame(o, vix, spy)
        c=o["Close"]; last=o.iloc[-1]
        chg=float((c.iloc[-1]/c.iloc[-2]-1)*100) if len(c)>=2 else np.nan
        e200=last.get("EMA200"); e50=last.get("EMA50")
        va=last.get("VolAvg20"); vn=last.get("Volume"); rp=last.get("RangePos")
        hi20=float(sub["High"].rolling(20).max().iloc[-1])
        def _f(x): return float(x) if pd.notna(x) else np.nan
        return dict(ticker=t, alpharank=int(strength.iloc[-1]), state=str(state.iloc[-1]),
            price=float(c.iloc[-1]), chg=chg, rsi=_f(last.get("RSI")), adx=_f(last.get("ADX")),
            roc20=_f(last.get("ROC20")),
            above200=(bool(c.iloc[-1]>e200) if pd.notna(e200) else None),
            golden=(bool(e50>e200) if (pd.notna(e50) and pd.notna(e200)) else None),
            macd_pos=(bool(last.get("MACD")>0) if pd.notna(last.get("MACD")) else None),
            rangepos=_f(rp),
            vol_surge=(float(vn/va) if (pd.notna(vn) and pd.notna(va) and va>0) else np.nan),
            new20high=(bool(c.iloc[-1]>=hi20*0.999) if pd.notna(hi20) else None))
    except Exception:
        return None

def _extract_ohlcv(df, t):
    """Pull one ticker's OHLCV out of a yf.download frame regardless of column orientation."""
    if not isinstance(df.columns, pd.MultiIndex): return df          # single ticker
    lvl0=set(df.columns.get_level_values(0))
    if t in lvl0:                                                    # group_by="ticker": (ticker, field)
        try: return df[t]
        except Exception: return None
    if {"Open","High","Low","Close"} & lvl0:                         # group_by="column": (field, ticker)
        try: return df.xs(t, axis=1, level=1)
        except Exception: return None
    return None

def _scan_universe_build(universe_name, period):
    """Chunked download (batches of 20, one retry each) so Yahoo throttling on one batch
    doesn't kill the whole scan. Partial results are kept. Returns DataFrame or None."""
    tickers=UNIVERSES[universe_name]
    vix=spy=None                                                     # shared market context (best-effort)
    try:
        _v=yf.download("^VIX",period=period,auto_adjust=True,progress=False)
        if _v is not None and not _v.empty and "Close" in _v: vix=_v["Close"]
    except Exception: pass
    try:
        _s=yf.download("^GSPC",period=period,auto_adjust=True,progress=False)
        if _s is not None and not _s.empty and "Close" in _s: spy=_s["Close"]
    except Exception: pass
    rows=[]; CH=20
    for i in range(0,len(tickers),CH):
        batch=tickers[i:i+CH]; df=None
        for _ in range(2):                                           # one retry per batch
            try:
                df=yf.download(batch,period=period,auto_adjust=True,progress=False,group_by="ticker",threads=True)
                if df is not None and not df.empty: break
            except Exception:
                df=None
        if df is None or df.empty: continue
        for t in batch:
            sub=_extract_ohlcv(df,t)
            if sub is None: continue
            r=_scan_row(t, sub.copy(), vix, spy)
            if r: rows.append(r)
    return pd.DataFrame(rows) if rows else None

@st.cache_data(ttl=900,show_spinner=False)
def _scan_universe_cached(universe_name, period):
    out=_scan_universe_build(universe_name, period)
    if out is None or out.empty:
        raise RuntimeError("empty scan")                            # exceptions aren't cached -> a retry actually retries
    return out

def scan_universe(universe_name, period="1y"):
    """Fault-tolerant universe scan -> AlphaRank DataFrame (or None). Successes cached 15 min; failures are not."""
    try:
        return _scan_universe_cached(universe_name, period)
    except Exception:
        return None

SORT_OPTS=["AlphaRank (high first)","AlphaRank (low first)","Biggest % move","Smallest % move",
 "RSI (low first)","RSI (high first)","ADX (high first)","Volume surge","52-wk position (high first)"]
_SORT_MAP={"AlphaRank (high first)":("alpharank",False),"AlphaRank (low first)":("alpharank",True),
 "Biggest % move":("chg",False),"Smallest % move":("chg",True),"RSI (low first)":("rsi",True),
 "RSI (high first)":("rsi",False),"ADX (high first)":("adx",False),"Volume surge":("vol_surge",False),
 "52-wk position (high first)":("rangepos",False)}

def apply_screen(df, f):
    """Filter + sort scan results by the filter dict f (pure, testable)."""
    if df is None or df.empty: return df
    d=df.copy()
    d=d[(d["alpharank"]>=f.get("ar_min",0))&(d["alpharank"]<=f.get("ar_max",100))]
    if f.get("verdict","Any")!="Any": d=d[d["state"]==f["verdict"]]
    d=d[(d["rsi"].fillna(50)>=f.get("rsi_min",0))&(d["rsi"].fillna(50)<=f.get("rsi_max",100))]
    tr=f.get("trend","Any")
    if tr=="Above 200-day MA": d=d[d["above200"]==True]
    elif tr=="Below 200-day MA": d=d[d["above200"]==False]
    if f.get("golden_only"): d=d[d["golden"]==True]
    if f.get("new_high"):    d=d[d["new20high"]==True]
    if f.get("macd_pos"):    d=d[d["macd_pos"]==True]
    if f.get("adx_min",0)>0: d=d[d["adx"].fillna(0)>=f["adx_min"]]
    if f.get("vol_min",1.0)>1.0: d=d[d["vol_surge"].fillna(0)>=f["vol_min"]]
    rp=f.get("range_pos","Any")
    if rp=="Near 52-week highs": d=d[d["rangepos"].fillna(0)>=0.90]
    elif rp=="Near 52-week lows": d=d[d["rangepos"].fillna(1)<=0.10]
    col,asc=_SORT_MAP.get(f.get("sort","AlphaRank (high first)"),("alpharank",False))
    return d.sort_values(col,ascending=asc,na_position="last").reset_index(drop=True)

# screener widget defaults (keyed by their Streamlit widget key) + one-tap preset screens
WIDGET_DEFAULTS={"scr_ar_range":(0,100),"scr_verdict":"Any","scr_rsi_range":(0,100),
 "scr_trend":"Any","scr_range_pos":"Any","scr_sort":"AlphaRank (high first)",
 "scr_adx_min":0,"scr_vol_min":1.0,"scr_golden_only":False,"scr_new_high":False,"scr_macd_pos":False}
SCREEN_PRESETS={
 "— none —":{},
 "Bullish leaders":{"scr_ar_range":(65,100),"scr_verdict":"BUY","scr_trend":"Above 200-day MA","scr_sort":"AlphaRank (high first)"},
 "Oversold bounce":{"scr_rsi_range":(0,35),"scr_sort":"RSI (low first)"},
 "Momentum breakouts":{"scr_new_high":True,"scr_vol_min":1.5,"scr_adx_min":20,"scr_sort":"Biggest % move"},
 "Golden cross + bullish":{"scr_golden_only":True,"scr_ar_range":(55,100),"scr_sort":"AlphaRank (high first)"},
 "Near 52-week highs":{"scr_range_pos":"Near 52-week highs","scr_ar_range":(55,100),"scr_sort":"52-wk position (high first)"},
 "Bearish / weak":{"scr_ar_range":(0,35),"scr_verdict":"SELL","scr_sort":"AlphaRank (low first)"}}

# ==========================================================================
# COMPONENTS / VIEWS
# ==========================================================================
def verdict_color(state): return {"BUY":GREEN,"SELL":RED}.get(state,AMBER)

def score_card(tkr,strength,state,funda,bespoke=None):
    vc=funda.get("vol_chg")
    voltxt="—" if vc is None else f"{'▲' if vc>=0 else '▼'} {abs(vc):.0f}%"
    volcol=GREEN if (vc is not None and vc>=0) else (RED if vc is not None else MUTE)
    sub=(f"MCAP {money(funda.get('market_cap'))} · P/E {fnum(funda.get('pe'))} · "
         f"DIV {fpct(funda.get('div'))} · VOL <span style='color:{volcol}'>{voltxt}</span>")
    # day price + change (from OHLCV — always available)
    pr=funda.get("price"); chg=funda.get("chg")
    chgcol=GREEN if (chg is not None and chg>=0) else (RED if chg is not None else MUTE)
    priceline=""
    if pr is not None:
        chgtxt="" if chg is None else (f"<span style='color:{chgcol};font-size:14px;font-weight:600'> "
                                       f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span>")
        priceline=(f"<div class='cardsub' style='font-family:\"IBM Plex Mono\",monospace;"
                   f"font-size:18px;color:{TXT};margin:2px 0 4px'>${pr:,.2f}{chgtxt}</div>")
    col=verdict_color(state)
    if bespoke:
        poscol=GREEN if bespoke["long"] else AMBER
        posbadge="LONG ▲" if bespoke["long"] else "CASH ■"
        incol=GREEN if bespoke["in_beat"] else RED
        oocol=GREEN if bespoke["beat"] else RED
        in_tag=("beats B&amp;H" if bespoke["in_beat"] else "ties/▼ B&amp;H")
        oo_tag=("beat B&amp;H out-of-sample" if bespoke["beat"] else "did NOT beat B&amp;H out-of-sample")
        return f"""<div class="card" style="border-color:{AMBER}66">
      <span class="deck-tkr">{tkr}</span>
      <span class="verdict" style="background:{col}22;color:{col};border:1px solid {col}66">{state}</span>
      {priceline}
      <div class="lab">AlphaRank</div>
      <div class="num" style="color:{col}">{int(strength)}<span style="font-size:16px;color:{MUTE}">/100</span></div>
      <div class="gauge"><span class="tick" style="left:calc({strength}% - 1.5px)"></span></div>
      <div class="cardsub">{sub}</div>
      <hr style="border:none;border-top:1px solid {AMBER}55;margin:11px 0 8px"/>
      <div class="lab" style="color:{AMBER};letter-spacing:.04em;text-transform:none">αAlphawire · {bespoke['rule']}
        <span style="margin-left:6px;padding:1px 7px;border-radius:6px;font-size:11px;background:{poscol}22;color:{poscol};border:1px solid {poscol}66">{posbadge}</span></div>
      <div style="font-family:'Chakra Petch',sans-serif;font-size:23px;font-weight:700;color:{incol};margin:3px 0 1px">{bespoke['in_ret']:+.0f}%<span style="font-size:12px;color:{MUTE};font-weight:400"> historical vs B&amp;H {bespoke['in_bh']:+.0f}% · {in_tag}</span></div>
      <div class="cardsub" style="color:{oocol};font-size:12px">▶ out-of-sample (unseen): <b>{bespoke['oos']:+.0f}%</b> vs B&amp;H {bespoke['bh']:+.0f}% — {oo_tag}</div>
    </div>"""
    return f"""<div class="card">
      <span class="deck-tkr">{tkr}</span>
      <span class="verdict" style="background:{col}22;color:{col};border:1px solid {col}66">{state}</span>
      {priceline}
      <div class="lab">AlphaRank</div>
      <div class="num" style="color:{col}">{int(strength)}<span style="font-size:16px;color:{MUTE}">/100</span></div>
      <div class="gauge"><span class="tick" style="left:calc({strength}% - 1.5px)"></span></div>
      <div class="cardsub">{sub}</div>
    </div>"""

INDICATORS=["VERDICT","ALPHARANK","TREND","EMA 50/200","RSI","MACD","STOCH %K",
            "BOLLINGER","MFI","REL STR","OBV","ADX","ATR %","VIX","NEWS"]

def matrix(data):
    """data: {tkr: dict(...)} -> (disp_df, color_df) using the latest votes."""
    disp={}; color={}
    for t,d in data.items():
        o=d["o"].iloc[-1]; v=d["votes"].iloc[-1]; stg=d["strength"]; stt=d["state"]
        news=d["news_avg"]; nv=d["news_vote"]; vv=float(v.get("vix",0))
        def cell(val,s): return val,(GREEN if s>0 else RED if s<0 else MUTE)
        rows={}
        rows["VERDICT"]=(stt,verdict_color(stt))
        rows["ALPHARANK"]=(f"{int(stg)}", GREEN if stg>=60 else RED if stg<=40 else AMBER)
        tr=v["trend50"]+v["trend200"]
        rows["TREND"]=("▲ up" if tr>0 else "▼ down" if tr<0 else "~ mixed",
                       GREEN if tr>0 else RED if tr<0 else MUTE)
        rows["EMA 50/200"]=cell("golden" if v["cross"]>0 else "death", v["cross"])
        rows["RSI"]=cell(f"{o.RSI:.0f}", v["rsi"])
        rows["MACD"]=cell(f"{'▲' if v['macd']>0 else '▼'} {o.MACD_hist:+.2f}", v["macd"])
        rows["STOCH %K"]=cell(f"{o.Stoch_K:.0f}", v["stoch"])
        rows["BOLLINGER"]=cell("upper" if o.Close>o.BB_up else "lower" if o.Close<o.BB_low else
                               ("mid+" if v["boll"]>0 else "mid-"), v["boll"])
        rows["MFI"]=cell(f"{o.MFI:.0f}", v["mfi"])
        rows["REL STR"]=cell("▲ outperf" if v["rs"]>0 else "▼ underperf" if v["rs"]<0 else "~ inline", v["rs"])
        rows["OBV"]=cell("▲ rising" if v["obv"]>0 else "▼ falling", v["obv"])
        rows["ADX"]=(f"{o.ADX:.0f} {'trend' if o.ADX>25 else 'chop'}", CYAN if o.ADX>25 else MUTE)
        rows["ATR %"]=(f"{o.ATRpct:.1f}%", CYAN)   # volatility context, not a vote
        rows["VIX"]=(f"{d['vix_now']:.0f}" if d['vix_now'] else "n/a",
                     GREEN if vv>0 else RED if vv<0 else MUTE)
        rows["NEWS"]=cell(f"{news:+.2f}", nv)
        disp[t]={k:val[0] for k,val in rows.items()}
        color[t]={k:val[1] for k,val in rows.items()}
    dd=pd.DataFrame(disp).reindex(INDICATORS)
    cc=pd.DataFrame(color).reindex(INDICATORS)
    return dd,cc

def style_matrix(dd,cc):
    css=pd.DataFrame("",index=dd.index,columns=dd.columns)
    for r in dd.index:
        for c in dd.columns:
            col=cc.loc[r,c]
            strong = r in ("VERDICT","ALPHARANK")
            css.loc[r,c]=(f"background-color:{col}{'33' if strong else '1f'};"
                          f"color:{col};font-weight:{700 if strong else 500};"
                          f"text-align:center;border:1px solid {col}44")
    return dd.style.apply(lambda _:css,axis=None)

def matrix_html(dd, cc):
    """Dark HTML table (st.dataframe ignores injected CSS and follows the Streamlit theme,
    which renders white without a dark config.toml — so we build the table ourselves)."""
    th=(f"<th style='position:sticky;left:0;z-index:2;background:{PANEL};color:{MUTE};"
        "text-align:left;padding:8px 11px;font-size:11px;letter-spacing:.07em'>INDICATOR</th>")
    for c in dd.columns:
        th+=(f"<th style='background:{PANEL};color:{TXT};text-align:center;padding:8px 11px;"
             f"font-family:\"Chakra Petch\",sans-serif;font-size:14px;border-bottom:1px solid {GRID}'>{c}</th>")
    body=""
    for r in dd.index:
        body+=(f"<tr><td style='position:sticky;left:0;z-index:1;background:{PANEL};color:{MUTE};"
               f"padding:7px 11px;font-size:11px;white-space:nowrap;border-bottom:1px solid {GRID}'>{r}</td>")
        for c in dd.columns:
            col=cc.loc[r,c]; val=dd.loc[r,c]
            if val is None or (isinstance(val,float) and pd.isna(val)): val=""
            strong = r in ("VERDICT","ALPHARANK")
            body+=(f"<td style='background:{col}{'33' if strong else '1f'};color:{col};text-align:center;"
                   f"padding:7px 11px;font-weight:{700 if strong else 500};font-size:12.5px;"
                   f"border:1px solid {col}44'>{val}</td>")
        body+="</tr>"
    return (f"<div style='overflow-x:auto;border-radius:12px;border:1px solid {GRID}'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"IBM Plex Mono\",monospace'>"
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>")

def overlay_indicator(name, data):
    fig=go.Figure()
    def add(col, fn):
        for t,d in data.items():
            s=fn(d["o"]).tail(180)
            fig.add_trace(go.Scatter(x=s.index,y=s,name=t,mode="lines",line=dict(width=1.6)))
    title=name
    if name in ("VERDICT","ALPHARANK"):
        for t,d in data.items():
            s=d["strength_series"].tail(180)
            fig.add_trace(go.Scatter(x=s.index,y=s,name=t,mode="lines",line=dict(width=1.8)))
        fig.add_hrect(y0=60,y1=100,fillcolor=GREEN,opacity=0.06,line_width=0)
        fig.add_hrect(y0=0,y1=40,fillcolor=RED,opacity=0.06,line_width=0)
        title="AlphaRank over time (0–100)"
    elif name=="RSI":
        add("RSI",lambda o:o.RSI); fig.add_hline(y=70,line=dict(dash="dash",width=.6,color=RED)); fig.add_hline(y=30,line=dict(dash="dash",width=.6,color=GREEN)); title="RSI"
    elif name=="MACD": add("MACD",lambda o:o.MACD); title="MACD line"
    elif name=="STOCH %K": add("K",lambda o:o.Stoch_K); fig.add_hline(y=80,line=dict(dash="dash",width=.6,color=RED)); fig.add_hline(y=20,line=dict(dash="dash",width=.6,color=GREEN)); title="Stochastic %K"
    elif name=="ADX": add("ADX",lambda o:o.ADX); fig.add_hline(y=25,line=dict(dash="dash",width=.6,color=CYAN)); title="ADX (trend strength)"
    elif name=="OBV": add("OBV",lambda o:o.OBV); title="On-Balance Volume"
    elif name=="MFI":
        add("MFI",lambda o:o.MFI); fig.add_hline(y=80,line=dict(dash="dash",width=.6,color=RED)); fig.add_hline(y=20,line=dict(dash="dash",width=.6,color=GREEN)); title="Money Flow Index (volume-weighted)"
    elif name=="ATR %": add("ATR%",lambda o:o.ATRpct); title="ATR % — volatility (higher = choppier)"
    elif name=="REL STR":
        spy=data[next(iter(data))]["spy_series"]
        for t,d in data.items():
            px=d["o"].Close.tail(180)
            if spy is not None:
                sp=spy.reindex(d["o"].index).ffill().tail(180)
                rel=(px/px.iloc[0])/(sp/sp.iloc[0])*100
            else:
                rel=px/px.iloc[0]*100
            fig.add_trace(go.Scatter(x=rel.index,y=rel,name=t,mode="lines",line=dict(width=1.7)))
        fig.add_hline(y=100,line=dict(dash="dash",width=.7,color=CYAN))
        title="Relative strength vs S&P 500 (>100 = outperforming)"
    elif name=="BOLLINGER": add("%B",lambda o:o.PctB); fig.add_hline(y=1,line=dict(dash="dash",width=.6,color=RED)); fig.add_hline(y=0,line=dict(dash="dash",width=.6,color=GREEN)); title="Bollinger %B (0=lower band, 1=upper)"
    elif name in ("TREND","EMA 50/200"):
        for t,d in data.items():
            s=d["o"].Close.tail(180); s=s/s.iloc[0]*100
            fig.add_trace(go.Scatter(x=s.index,y=s,name=t,mode="lines",line=dict(width=1.6)))
        title="Price rebased to 100 (relative trend)"
    elif name=="VIX":
        vs=data[next(iter(data))]["vix_series"]
        if vs is not None:
            vs=vs.tail(180); fig.add_trace(go.Scatter(x=vs.index,y=vs,name="VIX",line=dict(width=1.8,color=AMBER)))
        title="VIX — market fear"
    elif name=="NEWS":
        ts=list(data.keys()); vals=[data[t]["news_avg"] for t in ts]
        cols=[GREEN if v>0.05 else RED if v<-0.05 else MUTE for v in vals]
        fig.add_trace(go.Bar(x=ts,y=vals,marker_color=cols)); title="News sentiment (current)"
    fig.update_layout(title=title)
    return dark(fig, 380)

def price_signals(o, state_series, strength_series, tkr, big=False, news_marks=None, fib=False, style="candles", trade_pos=None):
    d=o if trade_pos is not None else o.tail(180)   # match the backtest window when showing trades
    if trade_pos is not None:
        buys,sells=trade_markers(trade_pos); idx=trade_pos.index   # actual backtested trades
    else:
        buys,sells=alternating_signals(state_series); idx=state_series.index
    bi=[i for i in buys if idx[i] in d.index]
    si=[i for i in sells if idx[i] in d.index]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.07,row_heights=[0.7,0.3],
                      subplot_titles=("","AlphaRank (0–100)"))
    if style=="line":
        fig.add_trace(go.Scatter(x=d.index,y=d.Close,name="price",mode="lines",
            line=dict(width=1.8,color=TXT)),row=1,col=1)
    else:
        fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,
            name="px",increasing_line_color=GREEN,decreasing_line_color=RED),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA50,name="EMA50",line=dict(width=1,color=CYAN)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA200,name="EMA200",line=dict(width=1,color=AMBER)),row=1,col=1)
    if trade_pos is not None:   # shade spans where the strategy actually HELD the stock (dark = in cash)
        tp=trade_pos.reindex(d.index).fillna(0.0)
        for a,b in long_spans(tp):
            fig.add_vrect(x0=d.index[a],x1=d.index[b],fillcolor=GREEN,opacity=0.07,
                          line_width=0,layer="below",row=1,col=1)
    if fib:
        hi=float(d.High.max()); lo=float(d.Low.min()); rng=hi-lo
        for lvl in (0,0.236,0.382,0.5,0.618,0.786,1.0):
            y=hi-rng*lvl
            fig.add_hline(y=y,line=dict(color="rgba(245,166,35,0.45)",width=1,dash="dot"),
                annotation_text=f"{lvl*100:.1f}%  {y:.2f}",annotation_position="right",
                annotation_font=dict(size=9,color=AMBER),row=1,col=1)
    if trade_pos is not None:   # connect each BUY to its SELL; green = trade won, red = trade LOST
        wx=[];wy=[];lx=[];ly=[]
        for tr in trade_log(o,trade_pos):
            seg_x=[tr["bi"],tr["si"],None]; seg_y=[tr["bp"],tr["sp"],None]
            if tr["sp"]>=tr["bp"]: wx+=seg_x; wy+=seg_y
            else: lx+=seg_x; ly+=seg_y
        if wx: fig.add_trace(go.Scatter(x=wx,y=wy,mode="lines",line=dict(color=GREEN,width=1.3),
            opacity=0.6,name="winning trade",hoverinfo="skip",showlegend=False),row=1,col=1)
        if lx: fig.add_trace(go.Scatter(x=lx,y=ly,mode="lines",line=dict(color=RED,width=2.4),
            opacity=0.95,name="losing trade (sold below buy)",hoverinfo="skip",showlegend=False),row=1,col=1)
    if bi:
        fig.add_trace(go.Scatter(x=[idx[i] for i in bi],
            y=[float(o.Close.loc[idx[i]]) for i in bi],mode="markers",name="BUY",
            marker=dict(symbol="triangle-up",size=11,color=GREEN,line=dict(width=1,color="#063"))),row=1,col=1)
    if si:
        fig.add_trace(go.Scatter(x=[idx[i] for i in si],
            y=[float(o.Close.loc[idx[i]]) for i in si],mode="markers",name="SELL",
            marker=dict(symbol="triangle-down",size=11,color=RED,line=dict(width=1,color="#600"))),row=1,col=1)
    if news_marks:
        ndt=_naive_idx(d); xs=[];ys=[];txt=[];col=[]
        for when,sc,title in news_marks:
            p=int(ndt.searchsorted(pd.Timestamp(when)))
            if 0<=p<len(d):
                xs.append(d.index[p]); ys.append(float(d.High.iloc[p])*1.03)
                txt.append((title[:80]+"…") if len(title)>80 else title)
                col.append(GREEN if sc>0.05 else RED if sc<-0.05 else MUTE)
        if xs:
            fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers",name="news",showlegend=False,
                marker=dict(symbol="diamond",size=8,color=col,line=dict(width=0.5,color="#000")),
                text=txt,hovertemplate="📰 %{text}<extra></extra>"),row=1,col=1)
    s4=strength_series.reindex(d.index)
    fig.add_trace(go.Scatter(x=s4.index,y=s4,name="AlphaRank",
        line=dict(width=1.4,color=CYAN),fill="tozeroy",fillcolor="rgba(62,193,211,0.08)"),row=2,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, dragmode="zoom",
        newshape=dict(line=dict(color=AMBER,width=2)))
    return dark(fig, 820 if big else 560)

def strategy_positions(d, strategy, buy_th=72, sell_th=42):
    """Build a raw long(1)/flat(0) position series for the chosen backtest strategy."""
    o=d["o"]
    if strategy=="strength":
        s=d["strength_series"]
        raw=np.where(s>=buy_th,1.0,np.where(s<=sell_th,0.0,np.nan))
        return pd.Series(raw,index=s.index).ffill().fillna(0.0)
    if strategy=="rsi":
        r=o["RSI"]                                   # long when oversold, exit when overbought
        raw=np.where(r<=buy_th,1.0,np.where(r>=sell_th,0.0,np.nan))
        return pd.Series(raw,index=o.index).ffill().fillna(0.0)
    if strategy=="macd":                             # long while MACD line is above its signal
        return (o["MACD"]>o["MACD_signal"]).astype(float)
    if strategy=="ema":                              # golden/death cross
        return (o["EMA50"]>=o["EMA200"]).astype(float)
    # default: the AlphaWire BUY/HOLD/SELL signal
    return d["state_series"].map({"BUY":1.0,"SELL":0.0,"HOLD":np.nan}).ffill().fillna(0.0)

# strategy -> (label, needs_thresholds, default_buy, default_sell, buy_caption, sell_caption)
BT_STRATEGIES={
 "signal":   ("AlphaWire signal (BUY / SELL)", False, 72,42,"",""),
 "strength": ("AlphaRank threshold",            True, 72,42,"Go long when AlphaRank ≥","Go to cash when AlphaRank ≤"),
 "rsi":      ("RSI threshold",                 True, 35,70,"Go long when RSI ≤ (oversold)","Go to cash when RSI ≥ (overbought)"),
 "macd":     ("MACD cross (line vs signal)",   False,72,42,"",""),
 "ema":      ("EMA 50/200 cross",              False,72,42,"",""),
}

def backtest(o, pos, cost=0.001):
    """Long/flat timing test of a position series vs buy & hold.
    NO LOOK-AHEAD: positions are decided at a day's close, so they only take
    effect the NEXT bar (pos.shift(1)). A cost is charged on every switch.
    """
    ret=o.Close.pct_change().fillna(0.0)
    pos=pos.reindex(ret.index).ffill().fillna(0.0)
    pos_eff=pos.shift(1).fillna(0.0)                 # act the day AFTER the signal
    switch=pos_eff.diff().abs().fillna(0.0)
    strat=pos_eff*ret - switch*cost                  # subtract trading cost on switches
    eq=(1+strat).cumprod(); bh=(1+ret).cumprod()
    def cagr(e):
        yrs=max((e.index[-1]-e.index[0]).days/365.25,1e-9)
        return e.iloc[-1]**(1/yrs)-1
    def mdd(e): return float((e/e.cummax()-1).min())
    def shp(r):
        sd=r.std(); return float(r.mean()/sd*np.sqrt(252)) if sd>0 else 0.0
    return dict(eq=eq, bh=bh,
        strat_ret=(eq.iloc[-1]-1)*100, bh_ret=(bh.iloc[-1]-1)*100,
        strat_cagr=cagr(eq)*100, bh_cagr=cagr(bh)*100,
        strat_mdd=mdd(eq)*100, bh_mdd=mdd(bh)*100,
        strat_sharpe=shp(strat), bh_sharpe=shp(ret),
        trades=int((switch>1e-9).sum()), exposure=float(pos_eff.mean())*100)

def trade_log(o, pos):
    """Every actual round-trip of a position series: buy at the entry-signal close,
    sell at the exit-signal close. Returns list of dicts. Matches the backtest exactly:
    the product of (1+ret) over these trades == the strategy's gross return."""
    p=pos.fillna(0.0).values; c=o.Close.values; idx=o.index; out=[]; e=None
    for i in range(len(p)):
        if p[i]>0.5 and e is None: e=i
        elif p[i]<=0.5 and e is not None:
            out.append(dict(bi=idx[e],bp=float(c[e]),si=idx[i],sp=float(c[i]),ret=c[i]/c[e]-1,open=False)); e=None
    if e is not None:
        out.append(dict(bi=idx[e],bp=float(c[e]),si=idx[-1],sp=float(c[-1]),ret=c[-1]/c[e]-1,open=True))
    return out

def trade_log_stats(trades):
    rets=[t["ret"] for t in trades]
    if not rets: return dict(n=0,wins=0,losses=0,avg=0.0,gross=0.0,best=0.0,worst=0.0)
    gross=float(np.prod([1+r for r in rets])-1)
    return dict(n=len(rets),wins=sum(r>0 for r in rets),losses=sum(r<0 for r in rets),
                avg=float(np.mean(rets)),gross=gross,best=max(rets),worst=min(rets))

def trade_log_html(trades):
    TD=f"padding:5px 9px;border-bottom:1px solid {GRID}"
    TH=f"padding:6px 9px;color:{MUTE};font-size:10.5px;letter-spacing:.06em;text-align:left;border-bottom:1px solid {GRID}"
    rows=[]; eq=100.0
    for k,t in enumerate(trades,1):
        eq*=(1+t["ret"])                      # reinvest each result -> compounding
        c=GREEN if t["ret"]>=0 else RED
        tag=" <span style='color:%s'>· open</span>"%MUTE if t.get("open") else ""
        rows.append(f"<tr><td style='{TD};color:{MUTE}'>{k}</td>"
            f"<td style='{TD};color:{TXT}'>{str(t['bi'])[:10]} @ {t['bp']:.2f}</td>"
            f"<td style='{TD};color:{TXT}'>{str(t['si'])[:10]} @ {t['sp']:.2f}{tag}</td>"
            f"<td style='{TD};color:{c};text-align:right;font-weight:600'>{t['ret']*100:+.1f}%</td>"
            f"<td style='{TD};color:{CYAN};text-align:right'>${eq:,.2f}</td></tr>")
    head=(f"<tr><th style='{TH}'>#</th><th style='{TH}'>Bought (close)</th>"
          f"<th style='{TH}'>Sold (close)</th><th style='{TH};text-align:right'>Return</th>"
          f"<th style='{TH};text-align:right'>Equity ($100 → compounding)</th></tr>")
    return (f"<div style='max-height:360px;overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"IBM Plex Mono\",monospace;font-size:12px'>{head}{''.join(rows)}</table></div>")

def equity_chart(bt, tkr, label="Strategy"):
    e=bt["eq"]/bt["eq"].iloc[0]*100; b=bt["bh"]/bt["bh"].iloc[0]*100
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=b.index,y=b,name="Buy & hold",line=dict(width=1.4,color=MUTE)))
    fig.add_trace(go.Scatter(x=e.index,y=e,name=label,line=dict(width=2,color=GREEN),
        fill="tonexty",fillcolor="rgba(22,199,132,0.06)"))
    fig.update_layout(title=f"{tkr} — $100 invested: {label} vs buy &amp; hold")
    return dark(fig,360)

# ---- bespoke per-stock optimizer: tune on the TRAIN half, score on the untouched TEST half ----
# Searching params on one stock's history finds the rule that best fit its PAST. The only honest
# question is whether that same rule survives on data it never saw -> the test half stands in for
# "the future". This is what separates a real edge from a curve-fit.
OPT_GRIDS={
 "rsi":      [(b,s) for b in range(10,55,5) for s in range(55,90,5)],
 "strength": [(b,s) for b in range(55,95,5) for s in range(15,55,5) if s<b-5],
 "signal":   [(72,42)],   # no tunable threshold
 "macd":     [(72,42)],
 "ema":      [(72,42)],
}

def _split_idx(n, frac): return max(40, min(n-40, int(round(n*frac))))

def optimize_for_ticker(d, strategies, split_frac=0.7, cost=0.001):
    """For each strategy, find the (buy,sell) that MAXIMISES return on the TRAIN half, then report
    that SAME rule's return on the untouched TEST half. Rows sorted by in-sample return (the
    seductive number) so the drop to out-of-sample is obvious at a glance."""
    o=d["o"]; n=len(o); k=_split_idx(n,split_frac)
    tr_o, te_o = o.iloc[:k], o.iloc[k:]
    out=[]
    for strat in strategies:
        if strat not in BT_STRATEGIES: continue
        label=BT_STRATEGIES[strat][0]; grid=OPT_GRIDS.get(strat,[(72,42)]); best=None
        for (b,s) in grid:
            pos=strategy_positions(d,strat,b,s)
            r=backtest(tr_o, pos.iloc[:k], cost=cost)["strat_ret"]
            if best is None or r>best[0]: best=(r,b,s,pos)
        _,bb,ss,pos=best
        bt_tr=backtest(tr_o, pos.iloc[:k], cost=cost)
        bt_te=backtest(te_o, pos.iloc[k:], cost=cost)   # SAME rule, unseen data, equity restarts at $1
        out.append(dict(strat=strat,label=label,buy=bb,sell=ss,tunable=len(grid)>1,
            train_ret=bt_tr["strat_ret"], test_ret=bt_te["strat_ret"],
            bh_train=bt_tr["bh_ret"], bh_test=bt_te["bh_ret"],
            test_trades=bt_te["trades"], test_exp=bt_te["exposure"],
            train_beat=bt_tr["strat_ret"]>bt_tr["bh_ret"],
            test_beat=bt_te["strat_ret"]>bt_te["bh_ret"]))
    out.sort(key=lambda r:r["train_ret"],reverse=True)
    return dict(rows=out,k=k,n=n,
        train_start=str(o.index[0])[:10],split_date=str(o.index[k])[:10],test_end=str(o.index[-1])[:10],
        survivors=sum(r["test_beat"] for r in out),total=len(out))

def _opt_params_str(r):
    return (f"RSI {r['buy']}/{r['sell']}" if r['strat']=="rsi" else
            f"≥{r['buy']} / ≤{r['sell']}" if r['strat']=="strength" else "—")

def optimize_table_html(res):
    TD=f"padding:6px 9px;border-bottom:1px solid {GRID}"
    TH=f"padding:7px 9px;color:{MUTE};font-size:10px;letter-spacing:.04em;text-align:left;border-bottom:1px solid {GRID};vertical-align:bottom"
    body=[]
    for r in res["rows"]:
        tc=GREEN if r["test_ret"]>=r["bh_test"] else RED
        verdict=(f"<span style='color:{GREEN};font-weight:700'>BEAT B&amp;H</span>" if r["test_beat"]
                 else f"<span style='color:{RED}'>missed</span>")
        body.append(
            f"<tr><td style='{TD};color:{TXT}'>{r['label']}</td>"
            f"<td style='{TD};color:{MUTE};font-family:\"IBM Plex Mono\",monospace'>{_opt_params_str(r)}</td>"
            f"<td style='{TD};color:{GREEN};text-align:right;font-weight:600'>{r['train_ret']:+.0f}%</td>"
            f"<td style='{TD};color:{tc};text-align:right;font-weight:700'>{r['test_ret']:+.0f}%</td>"
            f"<td style='{TD};color:{MUTE};text-align:right'>{r['bh_test']:+.0f}%</td>"
            f"<td style='{TD};text-align:right'>{verdict}</td></tr>")
    head=(f"<tr><th style='{TH}'>Strategy</th><th style='{TH}'>Best fit</th>"
          f"<th style='{TH};text-align:right'>In-sample<br><span style='font-size:9px;color:{GREEN}'>tuned · the fantasy</span></th>"
          f"<th style='{TH};text-align:right'>Out-of-sample<br><span style='font-size:9px;color:{AMBER}'>unseen · the truth</span></th>"
          f"<th style='{TH};text-align:right'>Buy &amp; hold<br><span style='font-size:9px'>test half</span></th>"
          f"<th style='{TH};text-align:right'>Verdict<br><span style='font-size:9px'>out-of-sample</span></th></tr>")
    return (f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"Chakra Petch\",sans-serif;font-size:12.5px'>{head}{''.join(body)}</table></div>")

# ================= COMBINATORIAL BESPOKE ENGINE =================
# A library of indicator "primitives" (each a long/flat 1-0 series), searched singly and in
# AND / OR / 3-way combinations to hunt for whatever interaction best fit a stock's history.
# Everything is chosen on the TRAIN half and reported on the held-out TEST half.

def _hyst(x, buy, sell):
    """Mean-reversion hysteresis: long when x<=buy (oversold), flat when x>=sell, carry between."""
    raw=np.where(x<=buy,1.0,np.where(x>=sell,0.0,np.nan))
    return pd.Series(raw,index=x.index).ffill().fillna(0.0)

def primitive_catalog():
    """Return list of primitive-ids. A pid is a tuple; _prim_series rebuilds its 1-0 series."""
    pids=[("ema_cross",),("px_ema50",),("px_ema200",),("stacked",),("ema50_slope",),
          ("macd",),("macd_hist_up",),("di",),("stoch_cross",),
          ("obv",),("obv_slope",),("px_bbmid",),("bb_break",),("rsi_mid",),
          ("adx_di",20),("adx_di",25),("adx_rising",)]
    for b,s in [(30,70),(25,75),(35,65),(40,60),(20,80),(45,55)]: pids.append(("rsi",b,s))
    for b,s in [(20,80),(30,70),(10,90)]: pids.append(("stoch",b,s))
    for b,s in [(20,80),(30,70),(25,75)]: pids.append(("mfi",b,s))
    for b,s in [(0.2,0.8),(0.1,0.9),(0.05,0.95)]: pids.append(("pctb",b,s))
    return pids

def _prim_series(o, pid, awn=None):
    k=pid[0]
    if k=="awn":       return awn if awn is not None else pd.Series(0.0,index=o.index)
    if k=="ema_cross": return (o["EMA50"]>=o["EMA200"]).astype(float)
    if k=="px_ema50":  return (o["Close"]>=o["EMA50"]).astype(float)
    if k=="px_ema200": return (o["Close"]>=o["EMA200"]).astype(float)
    if k=="stacked":   return ((o["Close"]>=o["EMA50"])&(o["EMA50"]>=o["EMA200"])).astype(float)
    if k=="ema50_slope": return (o["EMA50"]>=o["EMA50"].shift(5)).astype(float)
    if k=="macd":      return (o["MACD"]>=o["MACD_signal"]).astype(float)
    if k=="macd_hist_up": return (o["MACD_hist"]>=o["MACD_hist"].shift(3)).astype(float)
    if k=="di":        return (o["plus_DI"]>=o["minus_DI"]).astype(float)
    if k=="stoch_cross": return (o["Stoch_K"]>=o["Stoch_D"]).astype(float)
    if k=="obv":       return (o["OBV"]>=o["OBV"].rolling(20).mean()).astype(float)
    if k=="obv_slope": return (o["OBV"]>=o["OBV"].shift(10)).astype(float)
    if k=="px_bbmid":  return (o["Close"]>=o["BB_mid"]).astype(float)
    if k=="bb_break":  return (o["Close"]>=o["BB_up"]).astype(float)
    if k=="rsi_mid":   return (o["RSI"]>=50).astype(float)
    if k=="adx_di":    return (((o["ADX"]>=pid[1])&(o["plus_DI"]>=o["minus_DI"]))).astype(float)
    if k=="adx_rising":return (((o["ADX"]>=o["ADX"].shift(5))&(o["ADX"]>=20))).astype(float)
    if k=="rsi":       return _hyst(o["RSI"],pid[1],pid[2])
    if k=="stoch":     return _hyst(o["Stoch_K"],pid[1],pid[2])
    if k=="mfi":       return _hyst(o["MFI"],pid[1],pid[2])
    if k=="pctb":      return _hyst(o["PctB"],pid[1],pid[2])
    return pd.Series(0.0,index=o.index)

def _prim_label(pid):
    k=pid[0]
    if k=="awn": return "AWN>0"
    return {"ema_cross":"EMA50≥200","px_ema50":"Px≥EMA50","px_ema200":"Px≥EMA200","stacked":"Px≥EMA50≥200",
            "ema50_slope":"EMA50↑","macd":"MACD≥sig","macd_hist_up":"MACDhist↑","di":"+DI≥−DI",
            "stoch_cross":"StochK≥D","obv":"OBV>avg","obv_slope":"OBV↑","px_bbmid":"Px≥BBmid",
            "bb_break":"Px≥BBup","rsi_mid":"RSI≥50","adx_rising":"ADX↑"}.get(k) or (
            f"ADX≥{pid[1]}&+DI" if k=="adx_di" else
            f"RSI {pid[1]}/{pid[2]}" if k=="rsi" else
            f"Stoch {pid[1]}/{pid[2]}" if k=="stoch" else
            f"MFI {pid[1]}/{pid[2]}" if k=="mfi" else
            f"%B {pid[1]}/{pid[2]}" if k=="pctb" else str(pid))

def combo_series(o, spec, awn=None):
    """spec = {'op':'SINGLE'|'AND'|'OR','parts':[pid,...]} -> long/flat 1-0 series."""
    parts=[_prim_series(o,p,awn) for p in spec["parts"]]
    if not parts: return pd.Series(0.0,index=o.index)
    if spec["op"]=="OR":
        s=parts[0]
        for p in parts[1:]: s=((s+p)>0).astype(float)
        return s
    s=parts[0]                                   # SINGLE or AND
    for p in parts[1:]: s=(s*p)
    return s

def combo_label(spec):
    j={"AND":" & ","OR":" | ","SINGLE":""}.get(spec["op"]," & ")
    return j.join(_prim_label(p) for p in spec["parts"])

def optimize_combo_for_ticker(d, split_frac=0.7, cost=0.001, top_k=10, triples=True, awn=None):
    """Search singles + AND/OR pairs (+ 3-way ANDs of the top primitives). Pick the best by
    TRAIN return; report each candidate's TEST return. Returns ranked rows + the best spec."""
    o=d["o"]; n=len(o); k=_split_idx(n,split_frac)
    tr_o, te_o = o.iloc[:k], o.iloc[k:]
    cat=primitive_catalog()
    if awn is not None and float(awn.iloc[:k].std() or 0)>0:   # only search AWN if it actually varies
        cat=cat+[("awn",)]
    cache={}
    def full(pid):
        if pid not in cache: cache[pid]=_prim_series(o,pid,awn)
        return cache[pid]
    def evl(series):                              # train return for a full-length 1-0 series
        return backtest(tr_o, series.iloc[:k], cost=cost)["strat_ret"]
    # 1) singles
    singles=[]
    for pid in cat:
        s=full(pid); r=evl(s); singles.append((r,{"op":"SINGLE","parts":[pid]}))
    singles.sort(key=lambda x:x[0],reverse=True)
    cand=list(singles)
    topp=[sp["parts"][0] for _,sp in singles[:top_k]]
    # 2) AND / OR pairs among the top primitives
    for i in range(len(topp)):
        for j in range(i+1,len(topp)):
            a,b=topp[i],topp[j]
            sand=full(a)*full(b); cand.append((evl(sand),{"op":"AND","parts":[a,b]}))
            sor=((full(a)+full(b))>0).astype(float); cand.append((evl(sor),{"op":"OR","parts":[a,b]}))
    # 3) 3-way ANDs among the very top
    if triples:
        for i in range(min(5,len(topp))):
            for j in range(i+1,min(5,len(topp))):
                for m in range(j+1,min(5,len(topp))):
                    a,b,c=topp[i],topp[j],topp[m]
                    s3=full(a)*full(b)*full(c); cand.append((evl(s3),{"op":"AND","parts":[a,b,c]}))
    # build rows with train + test
    rows=[]
    for tr_ret,spec in cand:
        ser=combo_series(o,spec,awn)
        bt_tr=backtest(tr_o,ser.iloc[:k],cost=cost); bt_te=backtest(te_o,ser.iloc[k:],cost=cost)
        rows.append(dict(spec=spec,label=combo_label(spec),
            train_ret=bt_tr["strat_ret"],test_ret=bt_te["strat_ret"],
            bh_train=bt_tr["bh_ret"],bh_test=bt_te["bh_ret"],
            train_beat=bt_tr["strat_ret"]>bt_tr["bh_ret"],test_beat=bt_te["strat_ret"]>bt_te["bh_ret"],
            test_trades=bt_te["trades"],test_exp=bt_te["exposure"]))
    # de-dup identical labels keeping best train, sort by train (the historical fit)
    seen={}
    for r in rows:
        if r["label"] not in seen or r["train_ret"]>seen[r["label"]]["train_ret"]: seen[r["label"]]=r
    rows=sorted(seen.values(),key=lambda r:r["train_ret"],reverse=True)
    best=rows[0] if rows else None
    return dict(rows=rows,best=best,k=k,n=n,n_tested=len(cand),
        train_start=str(o.index[0])[:10],split_date=str(o.index[k])[:10],test_end=str(o.index[-1])[:10],
        survivors=sum(r["test_beat"] for r in rows),total=len(rows))

def combo_table_html(res, limit=12):
    TD=f"padding:6px 9px;border-bottom:1px solid {GRID}"
    TH=f"padding:7px 9px;color:{MUTE};font-size:10px;letter-spacing:.04em;text-align:left;border-bottom:1px solid {GRID};vertical-align:bottom"
    body=[]
    for r in res["rows"][:limit]:
        tc=GREEN if r["test_ret"]>=r["bh_test"] else RED
        verdict=(f"<span style='color:{GREEN};font-weight:700'>beats</span>" if r["test_beat"]
                 else f"<span style='color:{RED}'>misses</span>")
        body.append(
            f"<tr><td style='{TD};color:{TXT};font-family:\"IBM Plex Mono\",monospace;font-size:11px'>{r['label']}</td>"
            f"<td style='{TD};color:{GREEN};text-align:right;font-weight:600'>{r['train_ret']:+.0f}%</td>"
            f"<td style='{TD};color:{tc};text-align:right;font-weight:700'>{r['test_ret']:+.0f}%</td>"
            f"<td style='{TD};color:{MUTE};text-align:right'>{r['bh_test']:+.0f}%</td>"
            f"<td style='{TD};text-align:right'>{verdict}</td></tr>")
    head=(f"<tr><th style='{TH}'>Indicator combination</th>"
          f"<th style='{TH};text-align:right'>In-sample<br><span style='font-size:9px;color:{GREEN}'>fit · fantasy</span></th>"
          f"<th style='{TH};text-align:right'>Out-of-sample<br><span style='font-size:9px;color:{AMBER}'>unseen · truth</span></th>"
          f"<th style='{TH};text-align:right'>B&amp;H<br><span style='font-size:9px'>test</span></th>"
          f"<th style='{TH};text-align:right'>vs B&amp;H</th></tr>")
    return (f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"Chakra Petch\",sans-serif;font-size:12.5px'>{head}{''.join(body)}</table></div>")

# ---- forward PROJECTIONS (extrapolation of the historical compound rate — NOT a forecast) ----
def _cagr(total_pct, n_bars):
    yrs=max(n_bars,1)/252.0
    base=1+total_pct/100.0
    return (base**(1/yrs)-1)*100 if base>0 and yrs>0 else float("nan")

PROJ_HORIZONS=[("1M",1/12),("3M",0.25),("6M",0.5),("1Y",1),("5Y",5),("10Y",10)]

def projection_table_html(bh_total, gen_total, bsp_total, n_bars, has_bespoke):
    cb,cg,cs=_cagr(bh_total,n_bars),_cagr(gen_total,n_bars),(_cagr(bsp_total,n_bars) if has_bespoke else None)
    def proj(c,y):
        if c is None or c!=c: return "—"
        v=((1+c/100)**y-1)*100; col=GREEN if v>=0 else RED
        return f"<span style='color:{col}'>{v:+.0f}%</span>"
    TD=f"padding:5px 8px;border-bottom:1px solid {GRID};text-align:right"
    rows=[]
    for lbl,y in PROJ_HORIZONS:
        cells=f"<td style='{TD};color:{MUTE};text-align:left'>{lbl}</td><td style='{TD}'>{proj(cb,y)}</td><td style='{TD}'>{proj(cg,y)}</td>"
        if has_bespoke: cells+=f"<td style='{TD}'>{proj(cs,y)}</td>"
        rows.append(f"<tr>{cells}</tr>")
    TH=f"padding:6px 8px;color:{MUTE};font-size:10px;text-align:right;border-bottom:1px solid {GRID}"
    head=(f"<tr><th style='{TH};text-align:left'>Horizon</th><th style='{TH}'>Buy &amp; hold</th>"
          f"<th style='{TH}'>Generic signal</th>"+(f"<th style='{TH}'>αAlphawire</th>" if has_bespoke else "")+"</tr>")
    note=(f"Extrapolates each strategy's <b>historical compound annual rate</b> "
          f"(B&amp;H {cb:+.0f}%/yr, generic {cg:+.0f}%/yr"+(f", Alphawire {cs:+.0f}%/yr" if has_bespoke else "")+
          "). NOT a forecast — it assumes the past rate simply continues, which it won't exactly.")
    return (f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
            f"<table style='width:100%;border-collapse:collapse;background:{BG};"
            f"font-family:\"Chakra Petch\",sans-serif;font-size:12.5px'>{head}{''.join(rows)}</table></div>"
            f"<div style='color:{MUTE};font-size:10.5px;margin-top:5px'>{note}</div>")

SCORE_LABELS={"trend20":"Price vs EMA20","trend50":"Price vs EMA50","trend200":"Price vs EMA200",
 "cross":"EMA 50/200 cross","ema50_slope":"EMA50 slope","macd_zero":"MACD vs 0","macd":"MACD momentum",
 "di":"DI direction","aroon":"Aroon","roc":"Momentum (ROC)","rsi":"RSI","stoch":"Stochastic",
 "mfi":"MFI","williams":"Williams %R","cci":"CCI","boll":"Bollinger %B","range52":"52w range pos",
 "obv":"OBV","rs":"Rel strength","vix":"VIX","news":"News"}

def score_breakdown(d):
    """Per-indicator contribution (vote × weight) summing to the composite -> strength."""
    v=d["votes"].iloc[-1]
    contrib={k:(d["news_vote"] if k=="news" else float(v.get(k,0)))*WEIGHTS[k] for k in WEIGHTS}
    s=pd.Series(contrib).reindex(list(WEIGHTS.keys()))
    labels=[SCORE_LABELS[k] for k in s.index]
    colors=[GREEN if x>0 else RED if x<0 else MUTE for x in s.values]
    fig=go.Figure(go.Bar(x=s.values,y=labels,orientation="h",marker_color=colors,
        text=[f"{x:+.1f}" for x in s.values],textposition="outside",cliponaxis=False))
    fig.update_layout(title="Contribution to score (points = vote × weight)",
        xaxis_title="points",yaxis=dict(autorange="reversed"),dragmode=False)
    fig=dark(fig,420)
    fig.update_xaxes(fixedrange=True); fig.update_yaxes(fixedrange=True)
    comp=float(s.sum()); maxw=float(sum(abs(w) for w in WEIGHTS.values()))
    strength=round((comp/maxw+1)/2*100)
    return fig,comp,maxw,strength

def financials_chart(df, tkr, title):
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df["period"],y=df["revenue"]/1e9,name="Revenue ($B)",marker_color=CYAN))
    fig.add_trace(go.Bar(x=df["period"],y=df["net_income"]/1e9,name="Net income ($B)",marker_color=GREEN))
    fig.update_layout(title=title,barmode="group")
    return dark(fig,300)

def render_movers():
    st.caption("Scan an index, score every name with **AlphaRank**, then filter. **Tap rows** to select up to 5, "
               "then load them. (AlphaRank here is the technical composite — news is added once you load a name.)")
    for k,dv in WIDGET_DEFAULTS.items(): st.session_state.setdefault(k,dv)
    _pend=st.session_state.pop("_scr_load",None)   # apply a saved-screen load BEFORE widgets exist
    if _pend is not None:
        merged=dict(WIDGET_DEFAULTS); merged.update(_pend)
        for k,vv in merged.items(): st.session_state[k]=vv
        st.session_state["scr_preset"]="— none —"; st.session_state["_scr_preset_done"]="— none —"

    tcol=st.columns([2,2])
    src=tcol[0].selectbox("Universe",list(UNIVERSES.keys()),key="mv_src")
    preset=tcol[1].selectbox("Preset screen",list(SCREEN_PRESETS.keys()),key="scr_preset")
    if st.session_state.get("_scr_preset_done")!=preset:      # apply preset once per change
        st.session_state["_scr_preset_done"]=preset
        merged=dict(WIDGET_DEFAULTS); merged.update(SCREEN_PRESETS[preset])
        for k,vv in merged.items(): st.session_state[k]=vv
        st.rerun()

    with st.spinner(f"Scanning {src} — scoring names…"):
        sc=scan_universe(src,period="1y")
    if sc is None or sc.empty:
        st.error(f"Couldn't load **{src}** right now — Yahoo is likely throttling this shared server. "
                 f"Try a smaller universe (**Dow Jones 30** is the most reliable), wait ~30s and reopen, "
                 f"or just type tickers into the boxes manually."); return

    with st.expander("Filters", expanded=True):
        c1,c2=st.columns(2)
        ar=c1.slider("AlphaRank range",0,100,key="scr_ar_range")
        verdict=c2.selectbox("Verdict",["Any","BUY","HOLD","SELL"],key="scr_verdict")
        c3,c4=st.columns(2)
        rsi=c3.slider("RSI range",0,100,key="scr_rsi_range")
        trend=c4.selectbox("Trend (vs 200-day MA)",["Any","Above 200-day MA","Below 200-day MA"],key="scr_trend")
        c5,c6=st.columns(2)
        rangep=c5.selectbox("52-week position",["Any","Near 52-week highs","Near 52-week lows"],key="scr_range_pos")
        sort=c6.selectbox("Sort by",SORT_OPTS,key="scr_sort")
        c7,c8=st.columns(2)
        adx_min=c7.slider("Min ADX (trend strength; 0 = off)",0,50,key="scr_adx_min")
        vol_min=c8.slider("Min volume surge (x avg; 1.0 = off)",1.0,5.0,step=0.5,key="scr_vol_min")
        c9,c10,c11=st.columns(3)
        golden=c9.checkbox("Golden cross only",key="scr_golden_only")
        newhi=c10.checkbox("New 20-day high",key="scr_new_high")
        macdp=c11.checkbox("MACD > 0",key="scr_macd_pos")

    f=dict(ar_min=ar[0],ar_max=ar[1],rsi_min=rsi[0],rsi_max=rsi[1],verdict=verdict,trend=trend,
           range_pos=rangep,sort=sort,adx_min=adx_min,vol_min=vol_min,
           golden_only=golden,new_high=newhi,macd_pos=macdp)
    res=apply_screen(sc,f)

    with st.expander("Saved screens"):
        saved=st.session_state.setdefault("screens_saved",{})
        s1,s2=st.columns([2,1])
        nm=s1.text_input("Name this screen",key="scr_save_name",placeholder="e.g. My oversold value")
        if s2.button("💾 Save",use_container_width=True,disabled=not (nm or "").strip()):
            saved[nm.strip()]={k:st.session_state.get(k,WIDGET_DEFAULTS[k]) for k in WIDGET_DEFAULTS}
            st.toast(f"Saved screen: {nm.strip()}",icon="💾")
        if saved:
            l1,l2,l3=st.columns([2,1,1])
            pick=l1.selectbox("Load a saved screen",list(saved.keys()),key="scr_load_pick")
            if l2.button("Load",use_container_width=True):
                st.session_state["_scr_load"]=saved.get(pick,{}); st.rerun()
            if l3.button("Delete",use_container_width=True):
                saved.pop(pick,None); st.rerun()
        st.caption("Logged in, saved screens persist with your account; otherwise they last while the app is awake.")

    st.caption(f"**{len(res)}** of {len(sc)} names match.")
    if res.empty:
        st.info("No matches — loosen the filters."); return

    disp=res.head(60).reset_index(drop=True)
    show=pd.DataFrame({"Ticker":disp["ticker"],"AlphaRank":disp["alpharank"],"Verdict":disp["state"],
        "% move":disp["chg"].map(lambda x:f"{x:+.2f}%" if pd.notna(x) else "—"),
        "Price":disp["price"].map(lambda x:f"{x:,.2f}"),
        "RSI":disp["rsi"].map(lambda x:f"{x:.0f}" if pd.notna(x) else "—")})
    css=pd.DataFrame("",index=show.index,columns=show.columns)
    for i in show.index:
        vc=verdict_color(disp.loc[i,"state"])
        css.loc[i,"AlphaRank"]=f"color:{vc};font-weight:700"
        css.loc[i,"Verdict"]=f"color:{vc};font-weight:700"
        ch=disp.loc[i,"chg"]
        css.loc[i,"% move"]=f"color:{GREEN if (pd.notna(ch) and ch>=0) else RED};font-weight:600"
    styled=show.style.apply(lambda _:css,axis=None)

    picks=[]
    try:    # native: tap rows to select (Streamlit >=1.35)
        ev=st.dataframe(styled,use_container_width=True,height=440,hide_index=True,
                        on_select="rerun",selection_mode="multi-row",key="mv_table")
        rows=(ev.selection.rows if hasattr(ev,"selection") else ev["selection"]["rows"])
        picks=[disp.loc[i,"ticker"] for i in rows]
    except TypeError:   # older Streamlit: tappable chips
        st.dataframe(styled,use_container_width=True,height=320,hide_index=True)
        opts=disp["ticker"].tolist(); arm=dict(zip(disp["ticker"],disp["alpharank"]))
        if hasattr(st,"pills"):
            picks=st.pills("Tap to select",opts,selection_mode="multi",key="mv_pills",
                           format_func=lambda t:f"{t} · {arm[t]}") or []
        else:
            picks=st.multiselect("Pick up to 5",opts,max_selections=5,key="mv_pills")

    if len(picks)>5:
        st.warning("Max 5 — using your first five."); picks=picks[:5]
    st.caption(f"**Selected:** {', '.join(picks) if picks else 'tap rows above'}")
    if st.button("⚡ Load into AlphaWire",type="primary",disabled=not picks,use_container_width=True):
        st.session_state["_load_syms"]=list(picks)[:5]
        st.session_state.pop("mv_pills",None); st.session_state.pop("mv_table",None)
        st.rerun()

# real modal if available (Streamlit >=1.37), else inline fallback
_open_movers = st.dialog("🔎 AlphaWire Screener")(render_movers) if hasattr(st,"dialog") else render_movers

def _breakdown_body(d, t):
    bfig,comp,maxw,bstr=score_breakdown(d)
    st.markdown(f"**{t}** · AlphaRank **{int(d['strength'])}/100** · **{d['state']}**")
    st.plotly_chart(bfig,use_container_width=True,config={"displayModeBar":False})
    st.markdown(f"Sum of contributions = **{comp:+.1f}** of ±{maxw:.0f} possible "
                f"→ ( {comp:+.1f}/{maxw:.0f} + 1 ) ÷ 2 = **{bstr}/100**. "
                "Each bar is one indicator's **graded** vote — anywhere from −1 to +1 depending on how "
                "strong its signal is (capped at ±1) — multiplied by its weight. "
                "News only nudges the live score (it has no daily history); every other indicator, "
                "including VIX, is part of the historical signal too.")
if hasattr(st,"dialog"):
    @st.dialog("🔢 How this score is built")
    def show_breakdown(d, t): _breakdown_body(d, t)
else:
    def show_breakdown(d, t):
        with st.expander(f"🔢 How {t}'s score is built",expanded=True): _breakdown_body(d, t)

def pick_indicator():
    """Reliable tappable chips (falls back to a dropdown on older Streamlit)."""
    if hasattr(st,"pills"):
        ch=st.pills("Tap an indicator to expand its chart",INDICATORS,
                    default="ALPHARANK",selection_mode="single",key="indpick")
        return ch or "ALPHARANK"
    return st.selectbox("Expand indicator",INDICATORS,index=INDICATORS.index("ALPHARANK"),key="indpick")

# ==========================================================================
# UI
# ==========================================================================
st.markdown("""
<div class="hero">
<svg viewBox="0 0 460 124" style="width:100%;max-width:520px;height:auto;display:block" role="img" aria-label="AlphaWire">
  <defs>
    <linearGradient id="awg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#16c784"/>
      <stop offset="0.55" stop-color="#3ec1d3"/>
      <stop offset="1" stop-color="#f5a623"/>
    </linearGradient>
    <filter id="awglow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <line x1="20" y1="112" x2="440" y2="112" stroke="rgba(62,193,211,0.22)" stroke-width="1.4"/>
  <line x1="20" y1="112" x2="440" y2="112" stroke="url(#awg)" stroke-width="1.4"
        stroke-dasharray="6 10" class="aw-dash"/>
  <circle r="3.4" fill="#16c784" filter="url(#awglow)" cy="112">
    <animate attributeName="cx" values="20;440" dur="3.2s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="3.2s" repeatCount="indefinite"/>
  </circle>
  <g filter="url(#awglow)">
    <circle cx="54" cy="50" r="32" fill="#0d1622" stroke="url(#awg)" stroke-width="2.4"/>
  </g>
  <path d="M86 50 H112" stroke="url(#awg)" stroke-width="2" fill="none"/>
  <circle cx="112" cy="50" r="3" fill="#3ec1d3"/>
  <text x="54" y="51" font-family="'Chakra Petch',sans-serif" font-weight="700" font-size="40"
        fill="url(#awg)" text-anchor="middle" dominant-baseline="central">&#945;</text>
  <text x="128" y="52" font-family="'Chakra Petch',sans-serif" font-weight="700" font-size="44"
        fill="url(#awg)" dominant-baseline="middle">AlphaWire</text>
  <text x="130" y="86" font-family="'IBM Plex Mono',monospace" font-weight="500" font-size="12"
        fill="#7d8694" letter-spacing="1.6">COMPOSITE BUY / SELL SIGNAL ENGINE</text>
</svg>
</div>
<style>
.aw-dash{animation:awflow 1.1s linear infinite}
@keyframes awflow{to{stroke-dashoffset:-16}}
</style>
""", unsafe_allow_html=True)

api_key=st.secrets.get("ANTHROPIC_API_KEY",None)
_has_finnhub=bool(_finnhub_key())

# ================= ACCOUNTS (lightweight) =================
# Username + salted-hash password, stored in a JSON file. NOTE: on Streamlit Community Cloud the
# container filesystem is EPHEMERAL — this resets on reboot/redeploy. For permanent multi-user
# accounts, point ALPHAWIRE_DB at a persistent disk or swap _load_db/_save_db for a real database.
import json as _json, hashlib as _hl, os as _os
_USER_DB=_os.environ.get("ALPHAWIRE_DB","alphawire_users.json")

def _load_db():
    try:
        with open(_USER_DB,"r") as f: return _json.load(f)
    except Exception: return {}

def _save_db(db):
    try:
        with open(_USER_DB,"w") as f: _json.dump(db,f)
        return True
    except Exception: return False

def _pw_hash(pw,salt): return _hl.sha256((salt+":"+pw).encode("utf-8")).hexdigest()

def register_user(username,pw):
    username=(username or "").strip().lower()
    if len(username)<3: return False,"Username needs ≥3 characters."
    if len(pw or "")<6: return False,"Password needs ≥6 characters."
    db=_load_db()
    if username in db: return False,"That username is taken."
    salt=_hl.sha256(_os.urandom(16)).hexdigest()[:16]
    db[username]={"salt":salt,"pw":_pw_hash(pw,salt),"data":{"watchlist":[],"bespoke":{}}}
    return (True,"Account created.") if _save_db(db) else (False,"Could not write to storage.")

def verify_user(username,pw):
    username=(username or "").strip().lower()
    db=_load_db(); u=db.get(username)
    if not u: return False
    return u.get("pw")==_pw_hash(pw,u.get("salt",""))

def get_user_data(username):
    return _load_db().get((username or "").strip().lower(),{}).get("data",{"watchlist":[],"bespoke":{}})

def save_user_data(username,data):
    username=(username or "").strip().lower(); db=_load_db()
    if username not in db: return False
    db[username]["data"]=data; return _save_db(db)

with st.sidebar:
    st.header("⚙ Settings")

    # ---- account: log in to save your watchlist + bespoke combinations ----
    _user=st.session_state.get("user")
    if _user:
        st.success(f"👤 Signed in: **{_user}**")
        if st.button("💾 Save watchlist + Alphawire rules",use_container_width=True):
            wl=[st.session_state.get(f"sym{i}","").strip().upper() for i in range(5)]
            wl=[s for s in wl if s]
            bsp={k[len("bespoke_choice_"):]:v for k,v in st.session_state.items()
                 if k.startswith("bespoke_choice_")}
            ok=save_user_data(_user,{"watchlist":wl,"bespoke":bsp,
                                     "screens":st.session_state.get("screens_saved",{})})
            st.toast("Saved." if ok else "Save failed (storage).",icon="💾" if ok else "⚠️")
        if st.button("Log out",use_container_width=True):
            st.session_state.pop("user",None); st.rerun()
    else:
        with st.expander("👤 Log in / Register — save your setup"):
            _au=st.text_input("Username",key="auth_u")
            _ap=st.text_input("Password",type="password",key="auth_p")
            _ac=st.columns(2)
            if _ac[0].button("Log in",use_container_width=True,key="auth_login"):
                if verify_user(_au,_ap):
                    u=_au.strip().lower(); st.session_state["user"]=u
                    dat=get_user_data(u)
                    if dat.get("watchlist"): st.session_state["_load_syms"]=dat["watchlist"]
                    if dat.get("screens"): st.session_state["screens_saved"]=dat["screens"]
                    for tkr,choice in (dat.get("bespoke") or {}).items():
                        st.session_state[f"bespoke_choice_{tkr}"]=choice
                        st.session_state[f"besptog_{tkr}"]=True
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
            if _ac[1].button("Register",use_container_width=True,key="auth_reg"):
                ok,msg=register_user(_au,_ap)
                (st.success if ok else st.error)(msg)
                if ok: st.session_state["user"]=_au.strip().lower(); st.rerun()
        st.caption("Optional. Accounts persist while the app stays awake; on the free tier they reset when the "
                   "app reboots (no permanent database attached).")
    st.divider()
    period=st.selectbox("History window (data depth)",["1mo","3mo","6mo","1y","2y","5y","10y","max"],index=5,
        help="How far back to pull prices. 5y+ recommended — a few months can't reveal anything on a stock that only trends up.")

    st.markdown("**αAlphawire search**")
    split_default=st.slider("Train on first __% of history",50,85,70,5,
        help="Percent of history (NOT days) used to TUNE each combination. The remaining % is held out as the "
             "out-of-sample test. 70 = tune on the older 70%, test on the most recent 30%.")
    st.caption(f"Tuning on the oldest **{split_default}%**, testing on the newest **{100-split_default}%** (unseen).")
    awn_halflife=st.slider("AWN news half-life (days)",2,20,5,1,
        help="How fast the AlphaWire News signal decays after a catalyst. 5 = a catalyst's effect roughly halves "
             "every 5 trading days. Lower = news matters only briefly; higher = it lingers.")

    st.markdown("**Backtest strategy** (manual)")
    _keys=list(BT_STRATEGIES.keys())
    bt_strategy=st.selectbox("Rule to test",_keys,
        format_func=lambda k:BT_STRATEGIES[k][0],
        help="What decides when the backtest is long (in the stock) vs in cash.")
    _,need_th,db,ds,bcap,scap=BT_STRATEGIES[bt_strategy]
    if need_th:
        lo_b,hi_b=(50,95) if bt_strategy=="strength" else (5,60)
        lo_s,hi_s=(5,60) if bt_strategy=="strength" else (40,90)
        bt_buy=st.slider(bcap,lo_b,hi_b,db,1)
        bt_sell=st.slider(scap,lo_s,hi_s,ds,1)
        if bt_strategy=="strength" and bt_sell>=bt_buy:
            st.caption("⚠️ Sell level should be below buy level.")
    else:
        bt_buy,bt_sell=db,ds
    bt_cost=st.slider("Trade cost (% per switch)",0.0,0.5,0.10,0.05,
        help="Charged each time the strategy moves in or out of the stock.")/100

    st.markdown("**Charts**")
    chart_style="line" if st.radio("Price style",["Candlestick","Line"],horizontal=True)=="Line" else "candles"
    show_fib=st.toggle("Fibonacci levels on charts",value=False,
        help="Auto-draws retracement lines from the visible swing high/low. "
             "(Plotly has no freehand Fib tool; use the line tool for custom ones.)")

    use_ai=st.toggle("Claude news sentiment",value=bool(api_key),disabled=not api_key,
        help="Add ANTHROPIC_API_KEY in Secrets to enable AI-graded news.") if api_key else False
    if not api_key: st.caption("ℹ️ Using built-in finance sentiment analyzer.")

    with st.expander("🔌 News data status"):
        _fk=_finnhub_key()
        if _fk:
            st.success(f"Finnhub key detected ({_fk[:4]}…{_fk[-3:]}). Dated, company-specific history is enabled.")
        else:
            st.warning("No Finnhub key detected — news falls back to Yahoo's recent-only feed.")
            st.caption("Add it under **Manage app → Settings → Secrets** as a **top-level** line "
                       "(not inside a [section]), exactly:")
            st.code('FINNHUB_API_KEY = "your_key_here"', language="toml")
            st.caption("Save, then **Reboot** the app from the same menu.")
        if st.button("Test Finnhub now", use_container_width=True):
            if not _fk:
                st.error("No key found in Secrets or environment — nothing to test yet.")
            else:
                import requests as _rq, datetime as _d
                try:
                    _r=_rq.get("https://finnhub.io/api/v1/company-news",
                        params={"symbol":"AAPL","from":str(_d.date.today()-_d.timedelta(days=30)),
                                "to":str(_d.date.today()),"token":_fk},timeout=12)
                    if _r.status_code==200 and isinstance(_r.json(),list):
                        st.success(f"OK — Finnhub returned {len(_r.json())} AAPL headlines (HTTP 200). "
                                   "Reopen a stock and news will be Finnhub-sourced.")
                    elif _r.status_code in (401,403):
                        st.error(f"HTTP {_r.status_code} — key sent but rejected. Re-check the key value "
                                 "(no surrounding quotes or spaces *inside* the key itself).")
                    elif _r.status_code==429:
                        st.error("HTTP 429 — rate-limited. Wait ~1 min and retry (free tier = 60 calls/min).")
                    else:
                        st.error(f"HTTP {_r.status_code} — unexpected: {str(_r.text)[:140]}")
                except Exception as _e:
                    st.error(f"Request failed ({type(_e).__name__}) — the host may be blocking outbound calls.")

st.markdown("##### Enter up to 5 symbols")
# movers picker hands tickers off via _load_syms; apply BEFORE inputs are created
if "_load_syms" in st.session_state:
    _picks=st.session_state.pop("_load_syms")
    for i in range(5):
        st.session_state[f"sym{i}"]=_picks[i] if i<len(_picks) else ""
for i,dv in enumerate(["AAPL","MSFT","NVDA","",""]):
    st.session_state.setdefault(f"sym{i}",dv)
cols=st.columns(5)
syms=[cols[i].text_input(f"#{i+1}",key=f"sym{i}",label_visibility="collapsed",
      placeholder=f"#{i+1}") for i in range(5)]
b1,b2=st.columns([2,1])
run=b1.button("⚡ Run AlphaWire",type="primary",use_container_width=True)
if b2.button("🔎 Screener: find & add stocks",use_container_width=True):
    _open_movers()
st.caption("The screener scores a whole index with AlphaRank, then filters by RSI, trend, breakouts, volume and more "
           "— tap names there to fill the boxes above, then Run to score them in full.")

tickers=[]
for s in syms:
    s=s.strip().upper()
    if s and s not in tickers: tickers.append(s)
tickers=tickers[:5]

if tickers and (run or any(syms)):
    vix_series=get_vix_hist(period)
    spy_series=get_spy(period)
    vix_now=float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) else None
    data={}
    prog=st.progress(0.0,text="Loading market data…")
    for k,t in enumerate(tickers):
        hist=get_hist(t,period)
        if hist is None or len(hist)<60:
            st.error(f"⚠️ No usable data for **{t}** — check the symbol."); prog.progress((k+1)/len(tickers)); continue
        o=enrich(hist)
        composite,strength,state,votes,max_w=signal_frame(o,vix_series,spy_series)
        funda=dict(get_fundamentals(t))   # copy (cached dict) before adding price-derived stat
        # price-derived backfill — always reliable since we already hold OHLCV (yfinance .info is flaky)
        try:
            if funda.get("hi52") is None: funda["hi52"]=float(o.High.tail(252).max())
            if funda.get("lo52") is None: funda["lo52"]=float(o.Low.tail(252).min())
            if funda.get("volume") is None: funda["volume"]=float(o.Volume.iloc[-1])
            if funda.get("avg_vol") is None:
                va0=o.VolAvg20.iloc[-1]
                if va0==va0: funda["avg_vol"]=float(va0)
        except Exception: pass
        try:
            va=o.VolAvg20.iloc[-1]
            funda["vol_chg"]=(o.Volume.iloc[-1]/va-1)*100 if va and va==va else None
        except Exception:
            funda["vol_chg"]=None
        try:
            funda["price"]=float(o.Close.iloc[-1])
            funda["chg"]=(float(o.Close.iloc[-1])/float(o.Close.iloc[-2])-1)*100 if len(o)>1 else 0.0
        except Exception:
            funda["price"]=funda["chg"]=None
        yrs=PERIOD_YEARS.get(period,2)
        news_items,news_src=get_company_news(t, period_years=yrs)
        titles=[n["title"] for n in news_items if n["title"]]
        scores=[vader(x) for x in titles]                       # per-item (display + marks); cheap
        moves=[price_move_after(o,n["when"]) for n in news_items]
        # LIVE sentiment uses only RECENT headlines (not years of history) so the verdict isn't diluted
        _lastd=pd.Timestamp(o.index[-1]); _lastd=_lastd.tz_localize(None) if _lastd.tz is not None else _lastd
        def _recent(w):
            try: return (_lastd-pd.Timestamp(w)).days<=120
            except Exception: return False
        recent_titles=[n["title"] for n in news_items if n["title"] and _recent(n["when"])][:25] or titles[:20]
        rscores=(llm_sentiment(recent_titles,api_key) if (use_ai and api_key)
                 else [vader(x) for x in recent_titles]) if recent_titles else []
        news_avg=float(np.mean(rscores)) if rscores else 0.0
        news_vote=1 if news_avg>0.1 else -1 if news_avg<-0.1 else 0
        live_comp=float(composite.iloc[-1]+WEIGHTS["news"]*news_vote)
        live_max=max_w+WEIGHTS["news"]
        lr=live_comp/live_max
        live_state="BUY" if lr>=BUY_TH else "SELL" if lr<=-BUY_TH else "HOLD"
        live_strength=round((lr+1)/2*100)
        # ---- MULTI-YEAR MATERIAL NEWS from the SAME deep, staggered set ----
        mat=material_news(news_items)
        big_moves=find_big_moves(o,window=3,top_n=14,min_pct=6.0,sep=8)
        move_rows=correlate_moves_news(big_moves,mat,window=4)
        kw_stats=news_keyword_stats(o,mat,fwd=5)
        # AWN — AlphaWire News indicator (decaying, causal catalyst signal)
        awn_raw,awn_long,awn_score=awn_series(o,mat,half_life=awn_halflife,fwd=5)
        awn_now=float(awn_score.iloc[-1]); awn_last=awn_latest(mat)
        # chart markers = MATERIAL catalysts only (not the recent noise headlines)
        mat_marks=[(m["when"],vader(m["title"]),f"[{m['category']}] {m['title']}") for m in mat if m.get("when")]
        news_marks=mat_marks if mat_marks else [(n["when"],sc,n["title"]) for n,sc in zip(news_items,scores) if n["when"]]
        data[t]=dict(o=o,votes=votes,strength=live_strength,state=live_state,
            strength_series=strength, state_series=state, vix_now=vix_now,
            vix_series=vix_series, spy_series=spy_series, news=news_items, news_src=news_src,
            news_avg=news_avg, news_vote=news_vote, scores=scores, titles=titles,
            moves=moves, news_marks=news_marks, funda=funda,
            material=mat, move_rows=move_rows, kw_stats=kw_stats, news_years=yrs,
            awn=awn_long, awn_score=awn_now, awn_raw=awn_raw, awn_last=awn_last)
        prog.progress((k+1)/len(tickers))
    prog.empty()

    if data:
        if vix_now is not None:
            vc=GREEN if vix_now<14 else RED if vix_now>=28 else AMBER
            st.markdown(f"<div class='subnote'>Market fear · <b style='color:{vc}'>VIX {vix_now:.1f}</b> "
                        f"({'calm' if vix_now<17 else 'normal' if vix_now<22 else 'elevated' if vix_now<30 else 'high fear'}) "
                        f"— folded into every score.</div>",unsafe_allow_html=True)
        # SCORE CARDS — each card carries: backtest-data lifecycle, a bespoke toggle, breakdown button
        ccols=st.columns(len(data))
        for col,(t,d) in zip(ccols,data.items()):
            with col:
                ores=st.session_state.get(f"optres_{t}"); has=bool(ores and ores.get("rows"))
                asof=st.session_state.get(f"optres_asof_{t}")
                # toggle state is read BEFORE the card so the card reflects bespoke; the widget renders below
                besp_on=bool(has and st.session_state.get(f"besptog_{t}",False))
                bespoke_disp=None
                if besp_on:
                    choice=st.session_state.get(f"bespoke_choice_{t}")
                    if not choice:
                        b0=ores["best"]                                   # best HISTORICAL fit (no test peeking)
                        choice=dict(spec=b0["spec"],label=b0["label"])
                        st.session_state[f"bespoke_choice_{t}"]=choice
                    st.session_state[f"bespoke_{t}"]=choice               # drives chart + backtest downstream
                    rr=next((r for r in ores["rows"] if r["label"]==choice["label"]),ores["best"])
                    posnow=combo_series(d["o"],rr["spec"],awn=d.get("awn")).iloc[-1]
                    bespoke_disp=dict(long=float(posnow)>0.5,rule=rr["label"],
                                      in_ret=rr["train_ret"],in_bh=rr["bh_train"],in_beat=rr["train_beat"],
                                      oos=rr["test_ret"],bh=rr["bh_test"],beat=rr["test_beat"])
                else:
                    st.session_state.pop(f"bespoke_{t}",None)
                # --- card first ---
                st.markdown(score_card(t,d["strength"],d["state"],d["funda"],bespoke=bespoke_disp),
                            unsafe_allow_html=True)
                # --- AWN: AlphaWire News indicator chip ---
                awnv=d.get("awn_score",0.0); awc=GREEN if awnv>3 else RED if awnv<-3 else MUTE
                last=d.get("awn_last")
                if last and last.get("when"):
                    try: _days=(pd.Timestamp(d["o"].index[-1]).tz_localize(None)-pd.Timestamp(last["when"])).days
                    except Exception: _days=None
                    ago=f"{_days}d ago" if _days is not None and _days>=0 else "recent"
                    cat=last.get("category","news"); tail=f"{cat} · {ago}"
                else:
                    tail="no material catalysts in window"
                st.markdown(f"<div style='border:1px solid {awc}55;border-radius:8px;padding:5px 9px;margin:2px 0 6px;"
                            f"background:{awc}14;font-size:12px'><b style='color:{awc}'>AWN {awnv:+.0f}</b> "
                            f"<span style='color:{MUTE}'>· AlphaWire News · {tail}</span></div>",unsafe_allow_html=True)
                bdlbl=f"🔢 How {int(d['strength'])}/100 is built"
                if st.button(bdlbl,key=f"bd_{t}",use_container_width=True,help="Per-indicator composite score breakdown"):
                    show_breakdown(d,t)
                # --- α Alphawire signal control (prominent; needs backtest data) ---
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-family:Chakra Petch;font-weight:700;font-size:14px;color:{CYAN}'>"
                        f"αAlphawire signal</div>"
                        f"<div style='color:{MUTE};font-size:11px;margin:-2px 0 6px'>"
                        f"{t}'s own out-of-sample-tested combination — overrides the generic composite.</div>",
                        unsafe_allow_html=True)
                    st.toggle("Use αAlphawire signal",key=f"besptog_{t}",
                        help=("ON = drive this stock's signal, chart & backtest from its tuned combination."
                              if has else "Load backtest data first (button below) to enable this."))
                    if has:
                        st.caption(f"📈 Backtest as of **{asof}** · {ores.get('n','?')} bars · {ores.get('n_tested','?')} combos tested")
                        if st.button("↻ Update backtest data",key=f"upd_{t}",use_container_width=True,
                                     help="Re-fetch latest prices and re-run the combination search."):
                            try: get_hist.clear()
                            except Exception: pass
                            st.session_state[f"optres_{t}"]=optimize_combo_for_ticker(
                                d,split_frac=split_default/100,cost=bt_cost,awn=d.get("awn"))
                            st.session_state[f"optres_asof_{t}"]=str(d["o"].index[-1])[:10]
                            st.session_state.pop(f"bespoke_choice_{t}",None)
                            st.rerun()
                    else:
                        if st.session_state.get(f"besptog_{t}",False):
                            st.warning("⚠️ Load backtest data first to use the αAlphawire signal — tap the button below.")
                        if st.button("⚡ Load backtest data",key=f"load_{t}",type="primary",use_container_width=True,
                                     help="Search indicator combinations on this stock & out-of-sample-test them, unlocking the Alphawire signal."):
                            with st.spinner(f"Searching {t} indicator combinations…"):
                                st.session_state[f"optres_{t}"]=optimize_combo_for_ticker(
                                    d,split_frac=split_default/100,cost=bt_cost,awn=d.get("awn"))
                                st.session_state[f"optres_asof_{t}"]=str(d["o"].index[-1])[:10]
                            st.rerun()

        # MATRIX (dark HTML table — not st.dataframe, which renders white without a dark theme)
        st.markdown("#### Indicator matrix")
        dd,cc=matrix(data)
        st.markdown(matrix_html(dd,cc),unsafe_allow_html=True)
        choice=pick_indicator()
        st.markdown(f"##### 🔍 {choice}")
        st.plotly_chart(overlay_indicator(choice,data),use_container_width=True)

        # PER-STOCK PRICE + SIGNALS
        st.markdown("#### Price & historical signals")
        _tk_order=list(data.keys())
        tabs=st.tabs(_tk_order)
        _tabcols=[TAB_PALETTE[i%len(TAB_PALETTE)] for i in range(len(_tk_order))]
        components.html(f"""
        <script>
        const COLORS={_json.dumps(_tabcols)};
        const PALETTE=['rgb(22, 199, 132)','rgb(77, 139, 240)','rgb(245, 166, 35)','rgb(234, 57, 67)','rgb(167, 139, 250)'];
        function paint(){{
          try{{
            const doc=window.parent.document;
            const lists=doc.querySelectorAll('[data-baseweb="tab-list"]');
            if(!lists.length) return;
            const list=lists[lists.length-1];
            const tabs=list.querySelectorAll('button[data-baseweb="tab"]');
            if(!tabs.length) return;
            let active=0;
            tabs.forEach((b,i)=>{{ if(b.getAttribute('aria-selected')==='true') active=i; }});
            const c=COLORS[active]||'#16c784';
            // each tab label shows its OWN fixed color; active one bold + full opacity
            tabs.forEach((b,i)=>{{
              const p=b.querySelector('p')||b;
              p.style.setProperty('color', COLORS[i]||'#c7d0db','important');
              p.style.setProperty('opacity', i===active?'1':'0.62','important');
              p.style.setProperty('font-weight', i===active?'800':'600','important');
            }});
            // title follows the active tab's color
            doc.querySelectorAll('h4,h3,h2').forEach(h=>{{
              if((h.textContent||'').indexOf('Price & historical signals')>-1)
                h.style.setProperty('color',c,'important'); }});
            // underline bar = active tab color (structural + palette sweep, reversible across all colors)
            const wrap=list.parentElement;
            const hl=wrap.querySelector('[data-baseweb="tab-highlight"]'); if(hl) hl.style.setProperty('background-color',c,'important');
            const bd=wrap.querySelector('[data-baseweb="tab-border"]'); if(bd) bd.style.setProperty('background-color',c,'important');
            const root=list.closest('[data-testid="stTabs"]')||wrap;
            root.querySelectorAll('*').forEach(el=>{{
              const cs=getComputedStyle(el);
              if(PALETTE.includes(cs.backgroundColor)) el.style.setProperty('background-color',c,'important');
              if(PALETTE.includes(cs.borderBottomColor)) el.style.setProperty('border-bottom-color',c,'important');
              if(PALETTE.includes(cs.borderTopColor)) el.style.setProperty('border-top-color',c,'important');
            }});
          }}catch(e){{}}
        }}
        clearInterval(window.__phsPaint);
        window.__phsPaint=setInterval(paint,250);
        paint();
        </script>
        """, height=0)
        for _ti,(tab,(t,d)) in enumerate(zip(tabs,data.items())):
            with tab:
                f=d["funda"]
                acc=TAB_PALETTE[_ti%len(TAB_PALETTE)]
                sect=" · ".join(x for x in [f.get("sector"),f.get("industry")] if x)
                st.markdown(f"<div style='font-family:\"Chakra Petch\";font-size:15px;color:{acc}'>{f.get('name',t)}"
                            f"<span style='color:{MUTE};font-size:12px'>  {sect}</span></div>",unsafe_allow_html=True)
                st.markdown(fundamentals_grid(f,accent=acc),unsafe_allow_html=True)

                # ---- resolve the chosen backtest strategy ONCE (drives both chart + backtest) ----
                # A per-ticker BESPOKE combination (locked from the optimizer) overrides the sidebar choice.
                _bsp=st.session_state.get(f"bespoke_{t}")
                if _bsp:
                    strat_name="α"+_bsp["label"]
                    rule=(f"<span style='color:{AMBER}'>**αAlphawire locked for {t}**</span> "
                          f"(the indicator mix that best fit its history) — overriding the sidebar. "
                          f"**Long when:** {_bsp['label']}.")
                    pos=combo_series(d["o"],_bsp["spec"],awn=d.get("awn"))
                else:
                    _strat,_buy,_sell=bt_strategy,bt_buy,bt_sell
                    if _strat=="strength":
                        strat_name=f"AlphaRank ≥{_buy} / ≤{_sell}"
                        rule=(f"**Strategy:** hold the stock when **AlphaRank ≥ {_buy}**, "
                              f"go to **cash when AlphaRank ≤ {_sell}** (hold in between).")
                    elif _strat=="rsi":
                        strat_name=f"RSI ≤{_buy} / ≥{_sell}"
                        rule=(f"**Strategy:** buy when **RSI ≤ {_buy}** (oversold), sell to cash when "
                              f"**RSI ≥ {_sell}** (overbought) — a mean-reversion rule.")
                    elif _strat=="macd":
                        strat_name="MACD cross"
                        rule="**Strategy:** hold while the **MACD line is above its signal line**, cash when below."
                    elif _strat=="ema":
                        strat_name="EMA 50/200 cross"
                        rule="**Strategy:** hold while **EMA50 ≥ EMA200** (golden cross), cash on death cross."
                    else:
                        strat_name="AlphaWire signal"
                        rule=("**Strategy:** hold while the signal is **BUY**, cash on **SELL** "
                              "(**HOLD** keeps the prior position) — the composite shown above.")
                    pos=strategy_positions(d,_strat,_buy,_sell)

                # ---- price chart: ACTUAL trades of the chosen strategy + drawing tools ----
                st.markdown(f"##### 📈 Price &amp; trades — {strat_name}")
                st.caption(f"Full backtest window. ▲/▼ = where **{strat_name}** buys / sells, joined by a thin line "
                           "(**green = that trade won, red = it lost**). **Green shading = holding, dark = in cash** — "
                           "the dark gaps during rallies are the missed upside that makes a strategy trail buy & hold. "
                           "Diamonds are news; toolbar = drawing tools."
                           + (" Dotted amber = Fibonacci." if show_fib else ""))
                st.plotly_chart(price_signals(d["o"],d["state_series"],d["strength_series"],t,
                                    big=True,news_marks=d["news_marks"],fib=show_fib,style=chart_style,trade_pos=pos),
                                use_container_width=True,config=PLOTLY_DRAW,key=f"px_{t}")

                # ---- financial results (revenue & net income) ----
                fin=get_financials(t)
                if fin and (fin["annual"] is not None or fin["quarterly"] is not None):
                    with st.expander("📊 Financial results — revenue & net income"):
                        if fin["annual"] is not None:
                            st.plotly_chart(financials_chart(fin["annual"],t,f"{t} — annual"),
                                            use_container_width=True,key=f"finA_{t}")
                        if fin["quarterly"] is not None:
                            st.plotly_chart(financials_chart(fin["quarterly"],t,f"{t} — quarterly"),
                                            use_container_width=True,key=f"finQ_{t}")
                        st.caption("Source: Yahoo. Free depth ≈ last 4 fiscal years and ~4–5 recent quarters "
                                   "(full 5-year quarterly history requires a paid data feed).")

                # ---- backtest: chosen strategy vs buy & hold (strategy already resolved above) ----
                st.markdown(f"##### 📉 Backtest — {strat_name} vs buy &amp; hold")
                st.caption(rule, unsafe_allow_html=True)
                bt=backtest(d["o"],pos,cost=bt_cost)
                st.plotly_chart(equity_chart(bt,t,label=strat_name),use_container_width=True,key=f"eq_{t}")
                edge=bt["strat_ret"]-bt["bh_ret"]
                r1=st.columns(3)
                r1[0].metric("Strategy return",f"{bt['strat_ret']:+.1f}%",f"{edge:+.1f}% vs buy & hold")
                r1[1].metric("Buy & hold",f"{bt['bh_ret']:+.1f}%")
                r1[2].metric("Trades",f"{bt['trades']}")
                r2=st.columns(3)
                r2[0].metric("Max drawdown",f"{bt['strat_mdd']:.1f}%",
                             f"{bt['strat_mdd']-bt['bh_mdd']:+.1f}% vs B&H")
                r2[1].metric("Sharpe (strat)",f"{bt['strat_sharpe']:.2f}")
                r2[2].metric("Time in market",f"{bt['exposure']:.0f}%")
                verdict=("✅ Beat buy & hold over this window." if edge>0 else
                         "➖ Roughly matched buy & hold." if abs(edge)<2 else
                         "❌ Underperformed buy & hold here.")
                why=""
                if edge<-2:
                    bits=[]
                    if bt["exposure"]<70: bits.append(f"it sat in **cash {100-bt['exposure']:.0f}%** of the time, missing rallies")
                    if bt["trades"]>=20: bits.append(f"**{bt['trades']} trades** racked up cost/whipsaw")
                    why=" Here, "+" and ".join(bits)+"." if bits else ""
                st.caption(f"{verdict}{why}  **In-sample** test, {bt_cost*100:.2f}%/switch cost, "
                           "next-day execution (no look-ahead). Tune the rule in the sidebar; "
                           "**most timing rules underperform buy & hold on a stock that mostly rose.**")

                # ---- forward projections: 1M–10Y, generic signal vs bespoke vs buy & hold ----
                _gen_bt=backtest(d["o"],strategy_positions(d,"signal",72,42),cost=bt_cost)
                _has_bsp=bool(_bsp)
                st.markdown("##### 🔮 Projected potential — 1M to 10Y")
                st.markdown(projection_table_html(bt["bh_ret"],_gen_bt["strat_ret"],
                            (bt["strat_ret"] if _has_bsp else None),len(d["o"]),_has_bsp),
                            unsafe_allow_html=True)

                # ---- BESPOKE OPTIMIZER: search indicator COMBINATIONS, test them out-of-sample ----
                with st.expander(f"α Build {t}'s Alphawire — best indicator combination, tested on unseen data"):
                    st.caption("Searches ~25 indicator primitives singly and in **AND / OR / 3-way combinations**, "
                               "tunes them on the **older** part of this stock's history, then scores each on the "
                               "**recent** held-out part. In-sample is the fit; the **out-of-sample column is the only "
                               "one that says anything about the future.** A big gap = curve-fitting.")
                    split_pct=st.slider("Train on first __% of history",50,85,split_default,5,key=f"split_{t}",
                        help="Percent of history (not days) used to tune. The rest is the out-of-sample test.")
                    if st.button(f"⚡ Search {t} combinations",key=f"optrun_{t}",use_container_width=True):
                        with st.spinner(f"Searching {t} indicator combinations on the train split…"):
                            st.session_state[f"optres_{t}"]=optimize_combo_for_ticker(
                                d,split_frac=split_pct/100,cost=bt_cost,awn=d.get("awn"))
                            st.session_state[f"optres_asof_{t}"]=str(d["o"].index[-1])[:10]
                            st.session_state.pop(f"bespoke_choice_{t}",None)
                    ores=st.session_state.get(f"optres_{t}")
                    if ores and ores["rows"]:
                        st.markdown(combo_table_html(ores,limit=12),unsafe_allow_html=True)
                        surv,tot=ores["survivors"],ores["total"]
                        msg=("**None** beat buy &amp; hold out-of-sample — the glittering in-sample fits were curve-fit "
                             "to this stock's past and fell apart on unseen data. That is the trap, made visible."
                             if surv==0 else
                             "These held an edge on data they were never tuned on — promising, but re-check as fresh "
                             "data arrives; an edge can fade.")
                        st.caption(f"Tested **{ores['n_tested']}** combinations on **{ores['n']}** bars. "
                                   f"**{surv} of {tot}** beat buy &amp; hold **out-of-sample** "
                                   f"(trained {ores['train_start']} → {ores['split_date']}, tested → {ores['test_end']}). {msg}")
                        top=ores["rows"][:12]
                        labels=[r["label"] for r in top]
                        pick=st.selectbox(f"Preferred combination for {t}",["★ best historical fit"]+labels,
                            key=f"lockpick_{t}",
                            help="Which combination the αAlphawire toggle uses. Default = best fit on historical data.")
                        if st.button(f"Set as {t}'s Alphawire signal",key=f"lockbtn_{t}",use_container_width=True):
                            rr=ores["best"] if pick.startswith("★") else next(r for r in top if r["label"]==pick)
                            st.session_state[f"bespoke_choice_{t}"]=dict(spec=rr["spec"],label=rr["label"])
                            st.rerun()
                        cur=st.session_state.get(f"bespoke_choice_{t}")
                        if cur:
                            st.caption(f"Preferred: **{cur['label']}**. The **αAlphawire signal** toggle on {t}'s card uses it "
                                       "for the live signal, price chart and backtest.")
                        st.caption("⚠️ A combo that beats B&amp;H **in-sample** but not **out-of-sample** is curve-fit "
                                   "to the past and won't carry forward. Judge by the out-of-sample column.")

                # ---- the REAL per-trade ledger (replaces eyeballing the chart) ----
                tl=trade_log(d["o"],pos); ts=trade_log_stats(tl)
                if ts["n"]:
                    bh_factor=1+bt["bh_ret"]/100.0
                    in_factor=1+ts["gross"]
                    out_pct=(bh_factor/in_factor-1)*100 if in_factor>0 else 0.0
                    st.markdown(f"**Every actual trade:** {ts['n']} round-trips · "
                        f"<span style='color:{GREEN}'>{ts['wins']} winners</span> / "
                        f"<span style='color:{RED}'>{ts['losses']} losers</span> · "
                        f"avg {ts['avg']*100:+.1f}% · best {ts['best']*100:+.1f}% · worst {ts['worst']*100:+.1f}%",
                        unsafe_allow_html=True)
                    st.caption(f"Buying at each green's close and selling at each red's close, **reinvesting each "
                               f"result** (full compounding), turns **$100 into ${100*(1+ts['gross']):,.2f}** "
                               f"= **{ts['gross']*100:+.0f}% before costs** across {ts['n']} trades — that's the strategy "
                               "return above (costs trim it). The ledger's last 'Equity' row shows this build-up.")
                    st.info(f"**The identity that caps this:** a stock's total growth = (growth while you HOLD it) × "
                            f"(growth while you're in CASH). For {t}: buy & hold **{bt['bh_ret']:+.0f}%** = "
                            f"trades **{ts['gross']*100:+.0f}%** × in-cash **{out_pct:+.0f}%**. "
                            f"While this strategy sat in cash, {t} moved **{out_pct:+.0f}%** — the return it skipped. "
                            "To make +300% on a +83% stock, the in-cash periods would've had to *lose ~55%* — they didn't.")
                    with st.expander(f"📋 See all {ts['n']} trades (entry → exit → actual %)"):
                        st.markdown(trade_log_html(tl),unsafe_allow_html=True)

                    # spelled-out arithmetic for the FIRST trades (clean table — $ in markdown breaks as LaTeX)
                    first=tl[:8]; eqs=100.0; rws=[]
                    TDx=f"padding:5px 10px;border-bottom:1px solid {GRID};font-family:'IBM Plex Mono',monospace;font-size:12px"
                    for j,tr in enumerate(first,1):
                        eqs*=(1+tr["ret"]); rc=GREEN if tr["ret"]>=0 else RED
                        rws.append(f"<tr><td style='{TDx};color:{MUTE}'>{j}</td>"
                                   f"<td style='{TDx};text-align:right;color:{TXT}'>${tr['bp']:.2f}</td>"
                                   f"<td style='{TDx};text-align:right;color:{TXT}'>${tr['sp']:.2f}</td>"
                                   f"<td style='{TDx};text-align:right;color:{rc};font-weight:600'>{tr['ret']*100:+.1f}%</td>"
                                   f"<td style='{TDx};text-align:right;color:{TXT}'>${eqs:.2f}</td></tr>")
                    biggest=max((x['ret'] for x in tl),default=0)*100
                    THx=f"padding:6px 10px;color:{MUTE};font-size:10px;letter-spacing:.04em;text-align:right;border-bottom:1px solid {GRID}"
                    head=(f"<tr><th style='{THx};text-align:left'>#</th><th style='{THx}'>Bought</th>"
                          f"<th style='{THx}'>Sold</th><th style='{THx}'>Trade</th><th style='{THx}'>$100 →</th></tr>")
                    st.markdown(f"**First {len(first)} trades — real prices off the chart, reinvesting each time:**")
                    st.markdown(f"<div style='overflow:auto;border:1px solid {GRID};border-radius:10px'>"
                                f"<table style='width:100%;border-collapse:collapse;background:{BG};"
                                f"font-family:\"Chakra Petch\",sans-serif'>{head}{''.join(rws)}</table></div>",
                                unsafe_allow_html=True)
                    st.caption(f"Biggest single winner across all {ts['n']} trades: {biggest:+.1f}%. "
                               "If a price here doesn't match the chart, tell me which line.")

                # ---- MATERIAL catalysts vs the biggest moves (multi-year) ----
                mrows=d.get("move_rows") or []
                if mrows:
                    nmat=len(d.get("material") or []); hitn=sum(1 for r in mrows if r["had"])
                    with st.expander(f"📰 Material catalysts vs {t}'s biggest moves — last {d.get('news_years','?')}y "
                                     f"({nmat} catalysts, {hitn}/{len(mrows)} big moves had news)",expanded=False):
                        st.caption("The stock's largest ~3-day moves, each lined up against **material** news "
                                   "(earnings, guidance, M&A, analyst, regulatory, etc.) within ±4 days — the noise "
                                   "headlines are filtered out. Some moves had a clear catalyst; some didn't.")
                        st.markdown(big_moves_table_html(mrows),unsafe_allow_html=True)
                        if d.get("kw_stats"):
                            st.markdown("**Which catalysts moved the stock** — average 5-day move after each type fired:")
                            st.markdown(keyword_stats_html(d["kw_stats"],fwd=5),unsafe_allow_html=True)
                            st.caption("Correlation, not causation — a small sample of past events. A catalyst that "
                                       "averaged positive historically can still disappoint next time.")
                        if not _has_finnhub:
                            st.caption("⚠️ Using Yahoo's shallow recent feed. Add **FINNHUB_API_KEY** in Secrets for "
                                       "multi-year history — that's what makes this table reach back across the chart.")
                        elif d["news_src"]!="Finnhub":
                            st.caption("Finnhub returned no history for this symbol — showing Yahoo's recent feed instead.")

                # ---- news: two-pane table + reaction summary ----
                st.markdown(f"##### 📰 News & price reaction  <span style='color:{MUTE};font-size:12px'>via {d['news_src']} · {len(d['news'])} headlines</span>",unsafe_allow_html=True)
                if d["titles"]:
                    sgn=np.sign(d["news_avg"]) if abs(d["news_avg"])>0.05 else 0
                    mom=d["strength_series"].iloc[-1]-50
                    if sgn==0: msg="📰 Recent news is roughly neutral overall."
                    elif np.sign(mom)==sgn: msg=f"✅ Overall news {'positive' if sgn>0 else 'negative'} — **confirms** the current price trend."
                    else: msg=f"⚠️ Overall news {'positive' if sgn>0 else 'negative'} but the trend is {'up' if mom>0 else 'down'} — **divergence**."
                    st.info(msg)
                    summary=news_reaction_summary(d["scores"],d["moves"])
                    if summary: st.markdown(f"**What tends to move {t}:** {summary}")
                    st.markdown(news_table_html(d["news"][:30],d["scores"][:30],d["moves"][:30]),unsafe_allow_html=True)
                    if not _has_finnhub:
                        st.caption("Tip: add a FINNHUB_API_KEY in Secrets for dated, company-specific news "
                                   "(~1yr free) — that powers the chart markers and the reaction stats above.")
                else:
                    st.caption("No recent headlines for this ticker.")

        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown(f"""<div style="color:{MUTE};font-size:11px;line-height:1.6">
        Scores describe current &amp; past conditions — not predictions. Historical ▲/▼ use
        technical indicators + VIX (news has no daily history and only affects the live verdict).
        Not financial advice.
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);
        font-family:'Chakra Petch',sans-serif;letter-spacing:.04em;color:{MUTE}">
        <span style="color:{GREEN}">▲</span> ALPHAWIRE &nbsp;·&nbsp; © 2026 7562609 Manitoba Inc. &nbsp;·&nbsp; All rights reserved.
        </div></div>""",unsafe_allow_html=True)
elif not tickers:
    st.info("Enter at least one ticker above, then hit **Run AlphaWire**.")
