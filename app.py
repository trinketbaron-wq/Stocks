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

PLOTLY_FONT = dict(family="IBM Plex Mono, monospace", color=TXT, size=12)
def dark(fig, h=420):
    fig.update_layout(height=h, template="plotly_dark", font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8,r=8,t=34,b=8), legend=dict(orientation="h",y=1.08,font=dict(size=11)),
        hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

# ==========================================================================
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
# COMPONENTS / VIEWS
# ==========================================================================
def verdict_color(state): return {"BUY":GREEN,"SELL":RED}.get(state,AMBER)

def score_card(tkr,strength,state,funda):
    col=verdict_color(state)
    sub=f"MCAP {money(funda.get('market_cap'))} · P/E {fnum(funda.get('pe'))} · VOL {human(funda.get('volume') or funda.get('avg_vol'))}"
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

def price_signals(o, state_series, strength_series, tkr):
    d=o.tail(180)
    buys,sells=alternating_signals(state_series)
    idx=state_series.index
    bi=[i for i in buys if idx[i] in d.index]
    si=[i for i in sells if idx[i] in d.index]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05,row_heights=[0.7,0.3],
                      subplot_titles=(f"{tkr} — price + signals","strength (0-100)"))
    fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,
        name="px",increasing_line_color=GREEN,decreasing_line_color=RED),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA50,name="EMA50",line=dict(width=1,color=CYAN)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.index,y=d.EMA200,name="EMA200",line=dict(width=1,color=AMBER)),row=1,col=1)
    if bi:
        fig.add_trace(go.Scatter(x=[idx[i] for i in bi],
            y=[o.Low.loc[idx[i]]*0.985 for i in bi],mode="markers",name="BUY",
            marker=dict(symbol="triangle-up",size=14,color=GREEN,line=dict(width=1,color="#063"))),row=1,col=1)
    if si:
        fig.add_trace(go.Scatter(x=[idx[i] for i in si],
            y=[o.High.loc[idx[i]]*1.015 for i in si],mode="markers",name="SELL",
            marker=dict(symbol="triangle-down",size=14,color=RED,line=dict(width=1,color="#600"))),row=1,col=1)
    s4=strength_series.tail(180)
    fig.add_trace(go.Scatter(x=s4.index,y=s4,name="strength",
        line=dict(width=1.4,color=CYAN),fill="tozeroy",fillcolor="rgba(62,193,211,0.08)"),row=2,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return dark(fig,560)

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
    use_ai=st.toggle("Claude news sentiment",value=bool(api_key),disabled=not api_key,
        help="Add ANTHROPIC_API_KEY in Secrets to enable AI-graded news.") if api_key else False
    if not api_key: st.caption("ℹ️ Using built-in finance sentiment analyzer.")

st.markdown("##### Enter up to 5 symbols")
cols=st.columns(5)
defaults=["AAPL","MSFT","NVDA","",""]
syms=[cols[i].text_input(f"#{i+1}",value=defaults[i],key=f"sym{i}",label_visibility="collapsed",
      placeholder=f"#{i+1}") for i in range(5)]
run=st.button("⚡ Run AlphaWire",type="primary",use_container_width=True)

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
        funda=get_fundamentals(t)
        news=get_news(t)
        titles=[n["title"] for n in news if n["title"]]
        scores=(llm_sentiment(titles,api_key) if (use_ai and api_key) else [vader(x) for x in titles]) if titles else []
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
            vix_series=vix_series, spy_series=spy_series, news=news, news_avg=news_avg,
            news_vote=news_vote, scores=scores, titles=titles, funda=funda)
        prog.progress((k+1)/len(tickers))
    prog.empty()

    if data:
        if vix_now is not None:
            vc=GREEN if vix_now<14 else RED if vix_now>=28 else AMBER
            st.markdown(f"<div class='subnote'>Market fear · <b style='color:{vc}'>VIX {vix_now:.1f}</b> "
                        f"({'calm' if vix_now<17 else 'normal' if vix_now<22 else 'elevated' if vix_now<30 else 'high fear'}) "
                        f"— folded into every score.</div>",unsafe_allow_html=True)
        # SCORE CARDS
        cards="".join(score_card(t,d["strength"],d["state"],d["funda"]) for t,d in data.items())
        st.markdown(f"<div class='cardwrap'>{cards}</div>",unsafe_allow_html=True)

        # MATRIX
        st.markdown("#### Indicator matrix  ·  tap a row to expand →")
        dd,cc=matrix(data)
        ev=st.dataframe(style_matrix(dd,cc),use_container_width=True,
                        on_select="rerun",selection_mode="single-row",key="matrix")
        sel_rows=[]
        try: sel_rows=ev.selection.rows
        except Exception: pass
        from_table=INDICATORS[sel_rows[0]] if sel_rows else None
        if from_table: st.session_state.indchoice=from_table
        choice=st.session_state.get("indchoice","STRENGTH")

        c1,c2=st.columns([3,1])
        with c2:
            choice=st.selectbox("Expand indicator",INDICATORS,
                index=INDICATORS.index(choice),key="indselect")
        st.session_state.indchoice=choice
        with c1:
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
                st.plotly_chart(price_signals(d["o"],d["state_series"],d["strength_series"],t),use_container_width=True)
                if d["titles"]:
                    sgn=np.sign(d["news_avg"]) if abs(d["news_avg"])>0.05 else 0
                    mom=d["strength_series"].iloc[-1]-50   # >0 bullish, <0 bearish
                    if sgn==0: msg="📰 News roughly neutral."
                    elif np.sign(mom)==sgn: msg=f"✅ News {'positive' if sgn>0 else 'negative'} — **confirms** the current price inertia."
                    else: msg=f"⚠️ News {'positive' if sgn>0 else 'negative'} but momentum is {'up' if mom>0 else 'down'} — **divergence**, watch closely."
                    st.info(msg)
                    with st.expander(f"{len(d['titles'])} headlines"):
                        for n,sc in zip(d["news"],d["scores"]):
                            tag="🟢" if sc>0.05 else "🔴" if sc<-0.05 else "⚪"
                            line=f"{tag} **{sc:+.2f}** — {n['title']}"
                            if n["url"]: line+=f"  \n<small>[{n['publisher'] or 'src'} · {n['when']}]({n['url']})</small>"
                            st.markdown(line,unsafe_allow_html=True)
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
