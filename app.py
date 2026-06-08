"""
Interactive Stock Screener  (Streamlit web app)
================================================
Type tickers in the box -> get an interactive chart, a multi-timeframe
technical outlook, the VIX "fear" reading, and a news-sentiment panel that
compares how positive/negative the headlines are vs the stock's current
price momentum ("market inertia").

Run locally:   pip install -r requirements.txt   then   streamlit run app.py
Deploy free:   push app.py + requirements.txt to GitHub -> share.streamlit.io

NEWS SENTIMENT has two modes:
  * Built-in  : finance-tuned VADER lexicon (free, no key, decent baseline)
  * AI mode   : Claude via API key (optional, paid) -> better nuance.
                Add ANTHROPIC_API_KEY in the app's Secrets to enable.

NOTE: outlooks/sentiment are DESCRIPTIVE of current conditions, not a
price forecast. Analysis tool, not investment advice.
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
# Indicator math  (same engine validated in the CLI version)
# ==========================================================================

def ema(s, span):  return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100/(1+rs)

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

def stochastic(h, l, c, k=14, d=3):
    lo, hi = l.rolling(k).min(), h.rolling(k).max()
    pk = 100*(c-lo)/(hi-lo).replace(0, np.nan)
    return pk, pk.rolling(d).mean()

def bollinger(c, period=20, n=2.0):
    m = c.rolling(period).mean(); sd = c.rolling(period).std()
    return m+n*sd, m, m-n*sd

def atr(h, l, c, period=14):
    pc = c.shift()
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(h, l, c, period=14):
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    pc = c.shift()
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1/period, adjust=False).mean()/a
    mdi = 100*minus.ewm(alpha=1/period, adjust=False).mean()/a
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean(), pdi, mdi

def obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def add_indicators(df):
    o = df.copy(); c, h, l, v = o.Close, o.High, o.Low, o.Volume
    o["EMA50"], o["EMA200"] = ema(c, 50), ema(c, 200)
    o["RSI"] = rsi(c)
    o["MACD"], o["MACD_signal"], o["MACD_hist"] = macd(c)
    o["Stoch_K"], o["Stoch_D"] = stochastic(h, l, c)
    o["BB_up"], o["BB_mid"], o["BB_low"] = bollinger(c)
    o["ATR"] = atr(h, l, c)
    o["ADX"], o["plus_DI"], o["minus_DI"] = adx(h, l, c)
    o["OBV"] = obv(c, v)
    return o


# ==========================================================================
# Outlook scoring  (window-relative -> timeframes genuinely differ)
# ==========================================================================

TIMEFRAMES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}

def score_window(df):
    last, first = df.iloc[-1], df.iloc[0]
    votes = {}
    hz = (last.Close / first.Close) - 1
    votes["horizon trend"] = 1 if hz > 0 else -1
    votes["EMA50 slope"] = 1 if last.EMA50 > df.EMA50.iloc[0] else -1
    hi, lo = df.High.max(), df.Low.min()
    pos = (last.Close-lo)/(hi-lo) if hi > lo else 0.5
    votes["position in range"] = 1 if pos > 0.66 else (-1 if pos < 0.33 else 0)
    votes["OBV trend"] = 1 if df.OBV.iloc[-1] > df.OBV.iloc[0] else -1
    votes["price vs EMA200"] = 1 if last.Close > last.EMA200 else -1
    votes["DI direction"] = 1 if last.plus_DI > last.minus_DI else -1
    votes["RSI"] = 1 if last.RSI < 30 else (-1 if last.RSI > 70 else (1 if last.RSI > 50 else -1))
    votes["MACD"] = 1 if last.MACD_hist > 0 else -1
    votes["Stochastic"] = 1 if last.Stoch_K < 20 else (-1 if last.Stoch_K > 80 else (1 if last.Stoch_K > last.Stoch_D else -1))
    if last.Close < last.BB_low:   votes["Bollinger"] = 1
    elif last.Close > last.BB_up:  votes["Bollinger"] = -1
    else:                          votes["Bollinger"] = 1 if last.Close > last.BB_mid else -1

    raw = sum(votes.values())
    adx_avg = float(df.ADX.mean())
    if adx_avg < 20:   adj, regime = raw*0.5, "choppy / range-bound"
    elif adx_avg > 25: adj, regime = raw,     "trending"
    else:              adj, regime = raw*0.75, "developing trend"
    lean = "BULLISH" if adj >= 3 else ("BEARISH" if adj <= -3 else "NEUTRAL")
    return {"lean": lean, "score": round(adj, 1), "adx": round(adx_avg, 1),
            "regime": regime, "return_pct": round(hz*100, 1)}

def multi_timeframe(enr):
    return {lbl: score_window(enr.tail(d)) for lbl, d in TIMEFRAMES.items()}


def interpret_vix(v):
    if v < 13: return "very low - complacency"
    if v < 17: return "low - calm"
    if v < 22: return "normal"
    if v < 30: return "elevated - rising fear"
    return "high - fear / stress"


# ==========================================================================
# News + sentiment
# ==========================================================================

FINANCE_LEXICON = {
    "beats": 2.5, "beat": 2.0, "crushes": 3.0, "crushed": 2.5, "tops": 2.0,
    "topped": 2.0, "surge": 2.5, "surges": 2.5, "soars": 3.0, "soar": 2.5,
    "soared": 2.5, "rally": 2.0, "rallies": 2.0, "upgrade": 2.5, "upgraded": 2.5,
    "outperform": 2.5, "bullish": 3.0, "record": 1.5, "jumps": 2.0, "jump": 1.5,
    "gains": 1.5, "rebound": 2.0, "raises": 1.5, "raised": 1.5, "approval": 2.0,
    "misses": -2.5, "miss": -2.0, "missed": -2.0, "plunge": -3.0, "plunges": -3.0,
    "slumps": -2.5, "tumbles": -2.5, "downgrade": -2.5, "downgraded": -2.5,
    "lawsuit": -2.0, "probe": -1.5, "recall": -2.0, "bankruptcy": -3.5,
    "default": -2.5, "bearish": -3.0, "warns": -2.0, "warning": -1.5,
    "sinks": -2.5, "slides": -1.5, "drops": -1.5, "cuts": -1.5, "halts": -2.0,
}

@st.cache_resource
def get_analyzer():
    a = SentimentIntensityAnalyzer()
    a.lexicon.update(FINANCE_LEXICON)
    return a

def vader_sentiment(text):
    return get_analyzer().polarity_scores(text)["compound"]

def llm_sentiment(headlines, api_key):
    """Optional Claude-powered scoring. Returns list of floats in [-1,1].
    Falls back to VADER on any error so the app never breaks."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content":
                "Score each financial news headline for sentiment toward the "
                "company's stock, from -1 (very negative) to 1 (very positive). "
                "Reply ONLY with a JSON array of numbers, in order.\n\n" + numbered}],
        )
        import json, re
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        arr = json.loads(re.search(r"\[.*\]", txt, re.S).group())
        return [float(x) for x in arr][:len(headlines)]
    except Exception:
        return [vader_sentiment(h) for h in headlines]


def parse_news_item(item):
    """yfinance news schema varies by version -> extract defensively."""
    c = item.get("content", item)
    title = c.get("title") or item.get("title") or ""
    pub = (c.get("provider", {}) or {}).get("displayName") \
        or item.get("publisher") or ""
    url = (c.get("canonicalUrl", {}) or {}).get("url") or item.get("link") or ""
    when = c.get("pubDate") or item.get("providerPublishTime") or ""
    if isinstance(when, (int, float)):
        when = dt.datetime.fromtimestamp(when).strftime("%Y-%m-%d")
    elif isinstance(when, str) and "T" in when:
        when = when.split("T")[0]
    return {"title": title, "publisher": pub, "url": url, "when": when}


# ==========================================================================
# Data fetch  (cached to be gentle on Yahoo + fast on rerun)
# ==========================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker, period="2y"):
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    return df[["Open", "High", "Low", "Close", "Volume"]] if not df.empty else None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker, n=8):
    try:
        raw = yf.Ticker(ticker).news or []
        return [parse_news_item(x) for x in raw[:n]]
    except Exception:
        return []

@st.cache_data(ttl=900, show_spinner=False)
def fetch_vix():
    try:
        v = yf.Ticker("^VIX").history(period="5d", auto_adjust=True)
        return float(v.Close.iloc[-1]) if not v.empty else None
    except Exception:
        return None


# ==========================================================================
# Chart
# ==========================================================================

def build_chart(df, ticker):
    d = df.tail(180)  # ~9 months for readability
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.22, 0.23],
                        subplot_titles=(f"{ticker} price + EMAs + Bollinger", "RSI", "MACD"))
    # Price
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                  close=d.Close, name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.EMA50, name="EMA50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.EMA200, name="EMA200", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.BB_up, name="BB up", line=dict(width=0.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.BB_low, name="BB low", line=dict(width=0.5, dash="dot"),
                  fill="tonexty", fillcolor="rgba(120,120,200,0.07)"), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI", line=dict(width=1)), row=2, col=1)
    fig.add_hline(y=70, line=dict(width=0.5, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(width=0.5, dash="dash"), row=2, col=1)
    # MACD
    fig.add_trace(go.Bar(x=d.index, y=d.MACD_hist, name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD_signal, name="signal", line=dict(width=1)), row=3, col=1)

    fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06),
                      hovermode="x unified")
    return fig


# ==========================================================================
# UI
# ==========================================================================

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")
st.title("📈 Stock Screener + News Sentiment")
st.caption("Technical outlook across 1M/3M/6M/1Y, VIX context, and news "
           "sentiment vs price momentum. Descriptive analysis, not advice.")

api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)
    use_ai = False
    if api_key:
        use_ai = st.toggle("Use Claude for news sentiment", value=True,
                           help="More accurate than the built-in analyzer.")
        st.caption("✅ AI mode available")
    else:
        st.caption("ℹ️ Built-in finance analyzer (add ANTHROPIC_API_KEY in "
                   "Secrets to unlock Claude-powered AI mode).")

tickers_raw = st.text_input("Enter ticker(s)", value="AAPL",
                            placeholder="e.g. AAPL  or  AAPL, MSFT, NVDA")
go_btn = st.button("Analyze", type="primary")

if go_btn or tickers_raw:
    tickers = [t.strip().upper() for t in tickers_raw.replace(",", " ").split() if t.strip()]
    vix = fetch_vix()
    if vix is not None:
        st.info(f"**Market fear (VIX):** {vix:.1f} — {interpret_vix(vix)}")

    for tkr in tickers:
        st.divider()
        with st.spinner(f"Loading {tkr}…"):
            hist = fetch_history(tkr, period)
        if hist is None or len(hist) < 30:
            st.error(f"No usable data for **{tkr}** (check the symbol).")
            continue

        enr = add_indicators(hist)
        last = enr.iloc[-1]
        outlooks = multi_timeframe(enr)

        st.subheader(tkr)
        m = st.columns(4)
        m[0].metric("Price", f"${last.Close:,.2f}")
        m[1].metric("RSI", f"{last.RSI:.0f}")
        m[2].metric("ADX (trend str.)", f"{last.ADX:.0f}")
        m[3].metric("MACD hist", f"{last.MACD_hist:+.2f}")

        # Chart
        st.plotly_chart(build_chart(enr, tkr), use_container_width=True)

        # Outlook table
        st.markdown("**Multi-timeframe outlook**")
        tbl = pd.DataFrame(outlooks).T[["lean", "score", "return_pct", "adx", "regime"]]
        tbl.columns = ["Lean", "Score", "Return %", "ADX", "Regime"]
        st.dataframe(tbl, use_container_width=True)

        # ---- News + sentiment + inertia ----
        st.markdown("**News & sentiment vs momentum (inertia)**")
        news = fetch_news(tkr)
        if not news:
            st.caption("No recent headlines returned for this ticker.")
        else:
            titles = [n["title"] for n in news if n["title"]]
            scores = llm_sentiment(titles, api_key) if (use_ai and api_key) \
                else [vader_sentiment(t) for t in titles]
            avg = float(np.mean(scores)) if scores else 0.0

            # "Market inertia" = direction + persistence of recent price move
            mom_1m = outlooks["1M"]["return_pct"]
            adx_now = last.ADX
            mom_sign = np.sign(mom_1m)
            news_sign = np.sign(avg) if abs(avg) > 0.05 else 0

            cols = st.columns(3)
            cols[0].metric("News sentiment", f"{avg:+.2f}",
                           "positive" if news_sign > 0 else ("negative" if news_sign < 0 else "neutral"))
            cols[1].metric("1M momentum", f"{mom_1m:+.1f}%")
            cols[2].metric("Trend strength (ADX)", f"{adx_now:.0f}")

            if news_sign == 0:
                verdict = "📰 News is roughly **neutral** — little sentiment signal either way."
            elif news_sign == mom_sign:
                verdict = (f"✅ **Confirms inertia:** {'positive' if news_sign>0 else 'negative'} "
                           f"news lines up with the stock's {'up' if mom_sign>0 else 'down'} momentum"
                           + (" (and the trend is strong)." if adx_now > 25 else "."))
            else:
                verdict = (f"⚠️ **Divergence:** news is {'positive' if news_sign>0 else 'negative'} "
                           f"but price momentum is {'up' if mom_sign>0 else 'down'} — the news may be "
                           f"unpriced, fading, or hinting at a turn. Worth a closer look.")
            st.info(verdict)

            with st.expander(f"Show {len(titles)} headlines"):
                for n, s in zip(news, scores):
                    tag = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
                    line = f"{tag} **{s:+.2f}** — {n['title']}"
                    if n["url"]:
                        line += f"  \n[{n['publisher'] or 'source'} · {n['when']}]({n['url']})"
                    st.markdown(line)

    st.divider()
    st.caption("Indicators and sentiment describe current conditions — they are "
               "not predictions. Do your own research; this isn't financial advice.")
