"""
ALPHAWIRE — interactive multi-stock screener (Streamlit)
========================================================
© 2026 7562609 Manitoba Inc. All rights reserved.
Up to 5 tickers -> one composite STRENGTH score each (0-100) that maps to a
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================================================
# THEME
# ==========================================================================
BG="#0a0e14"; PANEL="#121a26"; GRID="rgba(255,255,255,0.05)"
TXT="#e6edf3"; MUTE="#7d8694"
GREEN="#16c784"; RED="#ea3943"; AMBER="#f5a623"; CYAN="#3ec1d3"

st.set_page_config(page_title="AlphaWire", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp {{ background:
    radial-gradient(1200px 600px at 80% -10%, rgba(62,193,211,0.08), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(22,199,132,0.07), transparent 55%),
    {BG}; color:{TXT}; }}
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
[data-testid="stPills"] button, .stPills button, [data-baseweb="pill"]{
  background:#0d1622 !important; color:#cfd6df !important;
  border:1px solid rgba(255,255,255,0.18) !important;
}
[data-testid="stPills"] button[aria-selected="true"],
[data-testid="stPills"] button[kind="primary"],
[data-testid="stPills"] button[aria-pressed="true"]{
  background:rgba(22,199,132,0.18) !important; color:#16c784 !important;
  border-color:#16c784 !important;
}
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
    return o

# ---- composite weighting (tunable: how much each indicator moves the score) ----
WEIGHTS={"trend50":1.0,"trend200":1.0,"cross":1.0,"di":1.0,"rsi":1.0,"macd":1.5,
         "stoch":1.0,"boll":1.0,"obv":1.0,"mfi":1.0,"rs":1.5,"vix":1.0,"news":1.0}
BUY_TH=0.25   # net bullish fraction (of max weight) needed to flip BUY / SELL

def _vix_vote(level):   # calm = mild tailwind, fear = headwind
    return np.select([level<14, level>22],[1,-1],0)

def signal_frame(o, vix_aligned=None, spy_aligned=None):
    """Per-bar weighted votes -> composite, 0-100 strength, BUY/HOLD/SELL state.
    News is NOT included here (no daily history); it's added to the LIVE score
    in the main loop. Returns (composite, strength, state, votes_df, max_weight)."""
    c=o.Close
    V=pd.DataFrame(index=o.index)
    V["trend50"]=_sign(c-o.EMA50)
    V["trend200"]=_sign(c-o.EMA200)
    V["cross"]=_sign(o.EMA50-o.EMA200)
    V["di"]=_sign(o.plus_DI-o.minus_DI)
    V["rsi"]=np.select([o.RSI<30,o.RSI>70],[1,-1],_sign(o.RSI-50))
    V["macd"]=_sign(o.MACD_hist)
    V["stoch"]=np.select([o.Stoch_K<20,o.Stoch_K>80],[1,-1],_sign(o.Stoch_K-o.Stoch_D))
    V["boll"]=np.select([c<o.BB_low,c>o.BB_up],[1,-1],_sign(c-o.BB_mid))
    V["obv"]=_sign(o.OBV-o.OBV.shift(10))
    V["mfi"]=np.select([o.MFI<20,o.MFI>80],[1,-1],_sign(o.MFI-50))
    if spy_aligned is not None:                       # relative strength vs market
        ratio=c/spy_aligned.reindex(o.index).ffill()
        V["rs"]=_sign(ratio-ratio.shift(21))
    else: V["rs"]=0.0
    if vix_aligned is not None:
        lvl=vix_aligned.reindex(o.index).ffill()
        V["vix"]=pd.Series(_vix_vote(lvl.values),index=o.index)
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

@st.cache_data(ttl=900,show_spinner=False)
def get_finnhub_news(symbol, days=365):
    """Dated, company-specific news from Finnhub (needs FINNHUB_API_KEY in Secrets)."""
    key=st.secrets.get("FINNHUB_API_KEY")
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

def get_company_news(t, days=365):
    """Finnhub if a key is set (dated, company-specific); else fall back to Yahoo."""
    fh=get_finnhub_news(t,days)
    if fh: return fh,"Finnhub"
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
    try: info=yf.Ticker(t).info or {}
    except Exception: info={}
    def g(*keys):
        for k in keys:
            x=info.get(k)
            if x not in (None,"",0): return x
        return None
    return dict(name=g("shortName","longName") or t, sector=g("sector"),
        industry=g("industry"), market_cap=g("marketCap"), pe=g("trailingPE"),
        fpe=g("forwardPE"), eps=g("trailingEps"), div=g("dividendYield"),
        beta=g("beta"), pb=g("priceToBook"), margin=g("profitMargins"),
        hi52=g("fiftyTwoWeekHigh"), lo52=g("fiftyTwoWeekLow"),
        avg_vol=g("averageVolume","averageDailyVolume10Day"),
        volume=g("volume","regularMarketVolume"))

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

def fundamentals_grid(f):
    cells=[("Market Cap",money(f["market_cap"])),("P/E (TTM)",fnum(f["pe"])),
           ("Fwd P/E",fnum(f["fpe"])),("EPS",fnum(f["eps"])),
           ("Div Yield",fpct(f["div"])),("Beta",fnum(f["beta"])),
           ("P/B",fnum(f["pb"])),("Profit Margin",fpct(f["margin"])),
           ("52W High",fnum(f["hi52"])),("52W Low",fnum(f["lo52"])),
           ("Avg Vol",human(f["avg_vol"])),("Volume",human(f["volume"]))]
    inner="".join(f'<div class="fstat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k,v in cells)
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

# ==========================================================================
# COMPONENTS / VIEWS
# ==========================================================================
def verdict_color(state): return {"BUY":GREEN,"SELL":RED}.get(state,AMBER)

def score_card(tkr,strength,state,funda):
    col=verdict_color(state)
    vc=funda.get("vol_chg")
    voltxt="—" if vc is None else f"{'▲' if vc>=0 else '▼'} {abs(vc):.0f}%"
    volcol=GREEN if (vc is not None and vc>=0) else (RED if vc is not None else MUTE)
    sub=(f"MCAP {money(funda.get('market_cap'))} · P/E {fnum(funda.get('pe'))} · "
         f"DIV {fpct(funda.get('div'))} · VOL <span style='color:{volcol}'>{voltxt}</span>")
    return f"""<div class="card">
      <span class="deck-tkr">{tkr}</span>
      <span class="verdict" style="background:{col}22;color:{col};border:1px solid {col}66">{state}</span>
      <div class="lab">strength</div>
      <div class="num" style="color:{col}">{int(strength)}<span style="font-size:16px;color:{MUTE}">/100</span></div>
      <div class="gauge"><span class="tick" style="left:calc({strength}% - 1.5px)"></span></div>
      <div class="cardsub">{sub}</div>
    </div>"""

INDICATORS=["VERDICT","STRENGTH","TREND","EMA 50/200","RSI","MACD","STOCH %K",
            "BOLLINGER","MFI","REL STR","OBV","ADX","ATR %","VIX","NEWS"]

def matrix(data):
    """data: {tkr: dict(...)} -> (disp_df, color_df) using the latest votes."""
    disp={}; color={}
    for t,d in data.items():
        o=d["o"].iloc[-1]; v=d["votes"].iloc[-1]; stg=d["strength"]; stt=d["state"]
        news=d["news_avg"]; nv=d["news_vote"]; vv=int(v.get("vix",0))
        def cell(val,s): return val,(GREEN if s>0 else RED if s<0 else MUTE)
        rows={}
        rows["VERDICT"]=(stt,verdict_color(stt))
        rows["STRENGTH"]=(f"{int(stg)}", GREEN if stg>=60 else RED if stg<=40 else AMBER)
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
            strong = r in ("VERDICT","STRENGTH")
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
            strong = r in ("VERDICT","STRENGTH")
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
    if name in ("VERDICT","STRENGTH"):
        for t,d in data.items():
            s=d["strength_series"].tail(180)
            fig.add_trace(go.Scatter(x=s.index,y=s,name=t,mode="lines",line=dict(width=1.8)))
        fig.add_hrect(y0=60,y1=100,fillcolor=GREEN,opacity=0.06,line_width=0)
        fig.add_hrect(y0=0,y1=40,fillcolor=RED,opacity=0.06,line_width=0)
        title="STRENGTH over time (0-100)"
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
                      subplot_titles=("","strength (0-100)"))
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
    fig.add_trace(go.Scatter(x=s4.index,y=s4,name="strength",
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
 "strength": ("Strength threshold",            True, 72,42,"Go long when strength ≥","Go to cash when strength ≤"),
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

SCORE_LABELS={"trend50":"Price vs EMA50","trend200":"Price vs EMA200","cross":"EMA 50/200 cross",
 "di":"DI direction","rsi":"RSI","macd":"MACD","stoch":"Stochastic","boll":"Bollinger",
 "obv":"OBV","mfi":"MFI","rs":"Rel strength","vix":"VIX","news":"News"}

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
    st.caption("Biggest 1-day movers in the chosen index. **Tap rows in the table** to select (up to 5), then load them.")
    src=st.selectbox("Universe",list(UNIVERSES.keys()),key="mv_src")
    with st.spinner(f"Scanning {src}…"):
        mv=fetch_movers(src)
    if mv is None or mv.empty:
        st.error("Couldn't load movers right now (Yahoo may be rate-limiting). "
                 "Try again shortly, or just type tickers manually.")
        return
    top=mv.head(40).reset_index(drop=True)
    show=pd.DataFrame({"Ticker":top["ticker"],
        "% move":top["chg"].map(lambda x:f"{x:+.2f}%"),
        "Price":top["price"].map(lambda x:f"{x:,.2f}")})
    if "volume" in top.columns:
        show["Volume"]=top["volume"].map(lambda x:human(x) if pd.notna(x) else "—")
    css=pd.DataFrame("",index=show.index,columns=show.columns)
    for i in show.index:
        css.loc[i,"% move"]=f"color:{GREEN if top.loc[i,'chg']>=0 else RED};font-weight:600"
    styled=show.style.apply(lambda _:css,axis=None)

    picks=[]
    try:    # native: tap rows to select (Streamlit ≥1.35)
        ev=st.dataframe(styled,use_container_width=True,height=420,hide_index=True,
                        on_select="rerun",selection_mode="multi-row",key="mv_table")
        rows=(ev.selection.rows if hasattr(ev,"selection") else ev["selection"]["rows"])
        picks=[top.loc[i,"ticker"] for i in rows]
    except TypeError:   # older Streamlit: fall back to tappable chips (kept in table order)
        st.dataframe(styled,use_container_width=True,height=300,hide_index=True)
        chg=dict(zip(top["ticker"],top["chg"])); opts=top["ticker"].tolist()
        if hasattr(st,"pills"):
            picks=st.pills("Tap to select",opts,selection_mode="multi",key="mv_pills",
                           format_func=lambda t:f"{t} {chg[t]:+.0f}%") or []
        else:
            picks=st.multiselect("Pick up to 5",opts,max_selections=5,key="mv_pills")

    if len(picks)>5:
        st.warning("Max 5 — using your first five."); picks=picks[:5]
    st.caption("A live AlphaWire **score** needs each name's full price history, so it's computed when you "
               "load picks (it then appears in the comparison matrix).")
    st.caption(f"**Selected:** {', '.join(picks) if picks else 'tap rows above'}")
    if st.button("⚡ Load into AlphaWire",type="primary",disabled=not picks,use_container_width=True):
        st.session_state["_load_syms"]=list(picks)[:5]
        st.session_state.pop("mv_pills",None); st.session_state.pop("mv_table",None)
        st.rerun()

# real modal if available (Streamlit >=1.37), else inline fallback
_open_movers = st.dialog("📈 Market Movers")(render_movers) if hasattr(st,"dialog") else render_movers

def _breakdown_body(d, t):
    bfig,comp,maxw,bstr=score_breakdown(d)
    st.markdown(f"**{t}** · strength **{int(d['strength'])}/100** · **{d['state']}**")
    st.plotly_chart(bfig,use_container_width=True,config={"displayModeBar":False})
    st.markdown(f"Sum of contributions = **{comp:+.1f}** of ±{maxw:.0f} possible "
                f"→ ( {comp:+.1f}/{maxw:.0f} + 1 ) ÷ 2 = **{bstr}/100**. "
                "Each bar is one indicator's vote (−1/0/+1) × its weight. "
                "VIX & News only shift the live score; the historical chart uses the technical part.")
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
                    default="STRENGTH",selection_mode="single",key="indpick")
        return ch or "STRENGTH"
    return st.selectbox("Expand indicator",INDICATORS,index=INDICATORS.index("STRENGTH"),key="indpick")

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
with st.sidebar:
    st.header("⚙ Settings")
    period=st.selectbox("History window",["1y","2y","5y"],index=1)

    st.markdown("**Backtest strategy**")
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
st.caption("The screener ranks index movers (price, % move, volume) — tap names there to fill the boxes above, "
           "then Run to score them.")

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
        try:
            va=o.VolAvg20.iloc[-1]
            funda["vol_chg"]=(o.Volume.iloc[-1]/va-1)*100 if va and va==va else None
        except Exception:
            funda["vol_chg"]=None
        news_items,news_src=get_company_news(t)
        titles=[n["title"] for n in news_items if n["title"]]
        scores=(llm_sentiment(titles,api_key) if (use_ai and api_key) else [vader(x) for x in titles]) if titles else []
        moves=[price_move_after(o,n["when"]) for n in news_items]
        news_marks=[(n["when"],sc,n["title"]) for n,sc in zip(news_items,scores) if n["when"]]
        news_avg=float(np.mean(scores)) if scores else 0.0
        # news is a weighted vote folded into the LIVE score only (no daily history)
        news_vote=1 if news_avg>0.1 else -1 if news_avg<-0.1 else 0
        live_comp=float(composite.iloc[-1]+WEIGHTS["news"]*news_vote)
        live_max=max_w+WEIGHTS["news"]
        lr=live_comp/live_max
        live_state="BUY" if lr>=BUY_TH else "SELL" if lr<=-BUY_TH else "HOLD"
        live_strength=round((lr+1)/2*100)
        data[t]=dict(o=o,votes=votes,strength=live_strength,state=live_state,
            strength_series=strength, state_series=state, vix_now=vix_now,
            vix_series=vix_series, spy_series=spy_series, news=news_items, news_src=news_src,
            news_avg=news_avg, news_vote=news_vote, scores=scores, titles=titles,
            moves=moves, news_marks=news_marks, funda=funda)
        prog.progress((k+1)/len(tickers))
    prog.empty()

    if data:
        if vix_now is not None:
            vc=GREEN if vix_now<14 else RED if vix_now>=28 else AMBER
            st.markdown(f"<div class='subnote'>Market fear · <b style='color:{vc}'>VIX {vix_now:.1f}</b> "
                        f"({'calm' if vix_now<17 else 'normal' if vix_now<22 else 'elevated' if vix_now<30 else 'high fear'}) "
                        f"— folded into every score.</div>",unsafe_allow_html=True)
        # SCORE CARDS — each card gets its own breakdown button attached directly beneath it
        ccols=st.columns(len(data))
        for col,(t,d) in zip(ccols,data.items()):
            with col:
                st.markdown(score_card(t,d["strength"],d["state"],d["funda"]),unsafe_allow_html=True)
                if st.button(f"🔢 How {int(d['strength'])}/100 is built",key=f"bd_{t}",
                             use_container_width=True,help="Per-indicator score breakdown"):
                    show_breakdown(d,t)

        # MATRIX (dark HTML table — not st.dataframe, which renders white without a dark theme)
        st.markdown("#### Indicator matrix")
        dd,cc=matrix(data)
        st.markdown(matrix_html(dd,cc),unsafe_allow_html=True)
        choice=pick_indicator()
        st.markdown(f"##### 🔍 {choice}")
        st.plotly_chart(overlay_indicator(choice,data),use_container_width=True)

        # PER-STOCK PRICE + SIGNALS
        st.markdown("#### Price & historical signals")
        tabs=st.tabs(list(data.keys()))
        for tab,(t,d) in zip(tabs,data.items()):
            with tab:
                f=d["funda"]
                sect=" · ".join(x for x in [f.get("sector"),f.get("industry")] if x)
                st.markdown(f"<div style='font-family:\"Chakra Petch\";font-size:15px;color:{TXT}'>{f.get('name',t)}"
                            f"<span style='color:{MUTE};font-size:12px'>  {sect}</span></div>",unsafe_allow_html=True)
                st.markdown(fundamentals_grid(f),unsafe_allow_html=True)

                # ---- resolve the chosen backtest strategy ONCE (drives both chart + backtest) ----
                # A per-ticker BESPOKE rule (locked from the optimizer) overrides the global sidebar choice.
                _bsp=st.session_state.get(f"bespoke_{t}")
                _strat=_bsp["strat"] if _bsp else bt_strategy
                _buy  =_bsp["buy"]   if _bsp else bt_buy
                _sell =_bsp["sell"]  if _bsp else bt_sell
                if _strat=="strength":
                    strat_name=f"Strength ≥{_buy} / ≤{_sell}"
                    rule=(f"**Strategy:** hold the stock when AlphaWire **strength ≥ {_buy}**, "
                          f"go to **cash when strength ≤ {_sell}** (hold in between).")
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
                if _bsp:
                    strat_name="🔬 "+strat_name
                    rule=(f"<span style='color:{AMBER}'>**Bespoke rule locked for {t}**</span> "
                          f"(tuned to its own history) — overriding the sidebar. "+rule)
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
                st.caption(rule)
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

                # ---- BESPOKE OPTIMIZER: fit each rule to this stock's past, then test it out-of-sample ----
                with st.expander(f"🔬 Find {t}'s best-fit rule — and test it on data it never saw"):
                    st.caption("Tunes each rule to the **early** part of this stock's history, then scores that "
                               "exact rule on the **recent** part held out as unseen. The in-sample column is the "
                               "fit; the **out-of-sample column is the only one that says anything about the future.** "
                               "A big gap between them = curve-fitting.")
                    oc=st.columns(2)
                    split_pct=oc[0].slider("Train on first … of history",50,85,70,5,key=f"split_{t}",
                        help="The remainder is held out as the out-of-sample test — a stand-in for the future.")
                    opt_set=oc[1].multiselect("Rules to search",list(BT_STRATEGIES.keys()),
                        default=list(BT_STRATEGIES.keys()),format_func=lambda k:BT_STRATEGIES[k][0],key=f"optset_{t}")
                    if st.button(f"⚡ Optimise {t}",key=f"optrun_{t}",use_container_width=True) and opt_set:
                        with st.spinner(f"Searching {t}'s parameter space on the train half…"):
                            st.session_state[f"optres_{t}"]=optimize_for_ticker(
                                d,opt_set,split_frac=split_pct/100,cost=bt_cost)
                    ores=st.session_state.get(f"optres_{t}")
                    if ores and ores["rows"]:
                        st.markdown(optimize_table_html(ores),unsafe_allow_html=True)
                        surv,tot=ores["survivors"],ores["total"]
                        msg=("**None** survived — the glittering in-sample fits were curve-fit to this stock's "
                             "past and fell apart on unseen data. That is the trap, made visible."
                             if surv==0 else
                             f"These held an edge on data they were never tuned on — promising, but re-check as "
                             "fresh data arrives; an edge can fade.")
                        st.caption(f"**{surv} of {tot}** rules beat buy &amp; hold **out-of-sample** "
                                   f"(trained {ores['train_start']} → {ores['split_date']}, tested → "
                                   f"{ores['test_end']}). {msg}")
                        best_oos=max(ores["rows"],key=lambda r:r["test_ret"])
                        opts=["—"]+[r["strat"] for r in ores["rows"]]
                        def _lbl(k):
                            if k=="—": return "Keep global sidebar rule"
                            nm=BT_STRATEGIES[k][0]
                            return nm+("  ·  ★ best out-of-sample" if k==best_oos["strat"] else "")
                        pick=st.selectbox(f"Lock a rule onto {t} (drives its signal, chart & backtest going forward)",
                            opts,format_func=_lbl,key=f"lockpick_{t}")
                        lc=st.columns(2)
                        if lc[0].button(f"🔒 Apply to {t}",key=f"lockbtn_{t}",use_container_width=True):
                            if pick=="—": st.session_state.pop(f"bespoke_{t}",None)
                            else:
                                rr=next(r for r in ores["rows"] if r["strat"]==pick)
                                st.session_state[f"bespoke_{t}"]=dict(strat=rr["strat"],buy=rr["buy"],sell=rr["sell"])
                            st.rerun()
                        if lc[1].button("↺ Clear bespoke",key=f"clr_{t}",use_container_width=True):
                            st.session_state.pop(f"bespoke_{t}",None); st.rerun()
                        st.caption("⚠️ Don't pick by the in-sample column — that's choosing the best curve-fit. "
                                   "If you lock a rule, judge it by **out-of-sample** and keep re-testing.")

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

                    # spelled-out arithmetic for the FIRST trades (the ones that "look hugely profitable")
                    first=tl[:8]; eqs=100.0; lines=[]
                    for j,tr in enumerate(first,1):
                        eqs*=(1+tr["ret"])
                        lines.append(f"{j}. bought **${tr['bp']:.2f}** → sold **${tr['sp']:.2f}** = "
                                     f"**{tr['ret']*100:+.1f}%**  →  $100 is now **${eqs:.2f}**")
                    biggest=max((t['ret'] for t in tl),default=0)*100
                    st.markdown("**The first trades, spelled out with the real prices off your chart:**\n\n"
                                + "\n\n".join(lines)
                                + f"\n\n*(Reinvesting every time. The single biggest winner in all {ts['n']} trades "
                                  f"was **{biggest:+.1f}%** — there is no 20%+ trade.)* "
                                  "If any price here doesn't match the chart, that's the bug — tell me which line.")

                # ---- news: two-pane table + reaction summary ----
                st.markdown(f"##### 📰 News & price reaction  <span style='color:{MUTE};font-size:12px'>via {d['news_src']}</span>",unsafe_allow_html=True)
                if d["titles"]:
                    sgn=np.sign(d["news_avg"]) if abs(d["news_avg"])>0.05 else 0
                    mom=d["strength_series"].iloc[-1]-50
                    if sgn==0: msg="📰 Recent news is roughly neutral overall."
                    elif np.sign(mom)==sgn: msg=f"✅ Overall news {'positive' if sgn>0 else 'negative'} — **confirms** the current price trend."
                    else: msg=f"⚠️ Overall news {'positive' if sgn>0 else 'negative'} but the trend is {'up' if mom>0 else 'down'} — **divergence**."
                    st.info(msg)
                    summary=news_reaction_summary(d["scores"],d["moves"])
                    if summary: st.markdown(f"**What tends to move {t}:** {summary}")
                    st.markdown(news_table_html(d["news"][:14],d["scores"][:14],d["moves"][:14]),unsafe_allow_html=True)
                    if d["news_src"]=="Yahoo":
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
