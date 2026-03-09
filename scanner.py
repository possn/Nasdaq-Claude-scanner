#!/usr/bin/env python3

# -*- coding: utf-8 -*-

“””
Nasdaq 100 - Daily Trading Signal Scanner
Gera sugestoes de day trading e swing trading (ate 1 semana).
Dados: Yahoo Finance (gratuito, sem API key necessaria)

Uso local:
pip install yfinance pandas numpy colorama tabulate requests
python scanner.py
python scanner.py –mode swing
python scanner.py –top 10
“””

import os
import argparse
import warnings
warnings.filterwarnings(“ignore”)

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

# –––––––––––––––––––––––––

# NASDAQ 100 tickers

# –––––––––––––––––––––––––

NASDAQ100 = [
“AAPL”,“MSFT”,“NVDA”,“AMZN”,“META”,“GOOGL”,“GOOG”,“TSLA”,“AVGO”,“COST”,
“ASML”,“NFLX”,“AMD”,“AZN”,“QCOM”,“CSCO”,“TMUS”,“INTU”,“LIN”,“AMGN”,
“ISRG”,“AMAT”,“TXN”,“BKNG”,“VRTX”,“MU”,“REGN”,“ADI”,“PANW”,“LRCX”,
“SBUX”,“MDLZ”,“SNPS”,“KLAC”,“CDNS”,“MELI”,“GILD”,“CRWD”,“CTAS”,“PDD”,
“CEG”,“INTC”,“ABNB”,“ORLY”,“MAR”,“ADSK”,“FTNT”,“DXCM”,“ROP”,“MNST”,
“WDAY”,“ROST”,“ADP”,“PCAR”,“MRVL”,“CHTR”,“ODFL”,“CPRT”,“MCHP”,“KDP”,
“PAYX”,“FANG”,“EXC”,“CTSH”,“NXPI”,“EA”,“FAST”,“IDXX”,“VRSK”,“GEHC”,
“CSGP”,“XEL”,“DDOG”,“WBD”,“ZS”,“SGEN”,“TEAM”,“ON”,“GFS”,“ANSS”,
“LULU”,“DLTR”,“BIIB”,“TTWO”,“ILMN”,“WBA”,“LCID”,“RIVN”,“ENPH”,“ALGN”,
“SIRI”,“MTCH”,“SWKS”,“OKTA”,“ZM”,“DOCU”,“PTON”,“COIN”,“HOOD”,“RBLX”
]

# –––––––––––––––––––––––––

# PARAMETROS DE RISCO

# –––––––––––––––––––––––––

STOP_ATR_MULT = 1.5
TARGET_RR     = 2.0
MIN_SCORE     = 3
MIN_AVG_VOL   = 1_000_000

# –––––––––––––––––––––––––

# INDICADORES TECNICOS

# –––––––––––––––––––––––––

def compute_rsi(series, period=14):
delta = series.diff()
gain  = delta.clip(lower=0).rolling(period).mean()
loss  = (-delta.clip(upper=0)).rolling(period).mean()
rs    = gain / loss.replace(0, np.nan)
return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
tr = pd.concat([
high - low,
(high - close.shift()).abs(),
(low  - close.shift()).abs()
], axis=1).max(axis=1)
return tr.rolling(period).mean()

def compute_macd(series):
ema12  = series.ewm(span=12, adjust=False).mean()
ema26  = series.ewm(span=26, adjust=False).mean()
macd   = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
return macd, signal

def compute_bb(series, period=20, std=2.0):
mid   = series.rolling(period).mean()
sigma = series.rolling(period).std()
return mid + std * sigma, mid, mid - std * sigma

def compute_adx(high, low, close, period=14):
up   = high.diff()
down = -low.diff()
plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
minus_dm = np.where((down > up) & (down > 0), down, 0.0)
atr      = compute_atr(high, low, close, period)
plus_di  = 100 * pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr
minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr
dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
return dx.rolling(period).mean(), plus_di, minus_di

# –––––––––––––––––––––––––

# ANALISE POR TICKER

# –––––––––––––––––––––––––

def analyse_ticker(ticker, mode=“all”):
try:
df = yf.download(ticker, period=“6mo”, interval=“1d”,
auto_adjust=True, progress=False)
if df is None or len(df) < 50:
return None

```
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    avg_vol = volume.iloc[-20:].mean()
    if avg_vol < MIN_AVG_VOL:
        return None

    rsi                    = compute_rsi(close)
    macd, macd_sig         = compute_macd(close)
    atr                    = compute_atr(high, low, close)
    bb_up, bb_mid, bb_low  = compute_bb(close)
    adx, plus_di, minus_di = compute_adx(high, low, close)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    c    = float(close.iloc[-1])
    r    = float(rsi.iloc[-1])
    m    = float(macd.iloc[-1])
    ms   = float(macd_sig.iloc[-1])
    at   = float(atr.iloc[-1])
    bbu  = float(bb_up.iloc[-1])
    bbl  = float(bb_low.iloc[-1])
    adxv = float(adx.iloc[-1])
    pdi  = float(plus_di.iloc[-1])
    mdi  = float(minus_di.iloc[-1])
    e20  = float(ema20.iloc[-1])
    e50  = float(ema50.iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / float(avg_vol)

    # Scoring LONG
    score_long, reasons_long = 0, []
    if 40 < r < 65:
        score_long += 1
        reasons_long.append("RSI %.0f (zona bullish)" % r)
    if m > ms:
        score_long += 1
        reasons_long.append("MACD acima do sinal")
    if c > e20 > e50:
        score_long += 1
        reasons_long.append("Preco > EMA20 > EMA50")
    if adxv > 20 and pdi > mdi:
        score_long += 1
        reasons_long.append("ADX %.0f tendencia bullish" % adxv)
    if vol_ratio > 1.3:
        score_long += 1
        reasons_long.append("Volume %.1fx acima da media" % vol_ratio)

    # Scoring SHORT
    score_short, reasons_short = 0, []
    if 35 < r < 60:
        score_short += 1
        reasons_short.append("RSI %.0f (zona bearish)" % r)
    if m < ms:
        score_short += 1
        reasons_short.append("MACD abaixo do sinal")
    if c < e20 < e50:
        score_short += 1
        reasons_short.append("Preco < EMA20 < EMA50")
    if adxv > 20 and mdi > pdi:
        score_short += 1
        reasons_short.append("ADX %.0f tendencia bearish" % adxv)
    if vol_ratio > 1.3:
        score_short += 1
        reasons_short.append("Volume %.1fx acima da media" % vol_ratio)

    is_day_long  = (r < 38) and (vol_ratio > 1.5) and (c < bbl * 1.01)
    is_day_short = (r > 62) and (vol_ratio > 1.5) and (c > bbu * 0.99)

    if score_long >= score_short and score_long >= MIN_SCORE:
        direction  = "LONG"
        score      = score_long
        reasons    = reasons_long
        trade_type = "DAY" if is_day_long else "SWING"
        entry      = c
        stop       = round(entry - at * STOP_ATR_MULT, 2)
        risk       = entry - stop
        target     = round(entry + risk * TARGET_RR, 2)
    elif score_short > score_long and score_short >= MIN_SCORE:
        direction  = "SHORT"
        score      = score_short
        reasons    = reasons_short
        trade_type = "DAY" if is_day_short else "SWING"
        entry      = c
        stop       = round(entry + at * STOP_ATR_MULT, 2)
        risk       = stop - entry
        target     = round(entry - risk * TARGET_RR, 2)
    else:
        return None

    if mode == "day"   and trade_type != "DAY":   return None
    if mode == "swing" and trade_type != "SWING": return None

    rr = round(abs(target - entry) / abs(stop - entry), 2)

    return {
        "ticker":    ticker,
        "tipo":      trade_type,
        "dir":       direction,
        "preco":     round(c, 2),
        "entrada":   round(entry, 2),
        "stop":      stop,
        "target":    round(target, 2),
        "rr":        rr,
        "score":     "%d/5" % score,
        "vol_rel":   round(vol_ratio, 2),
        "rsi":       round(r, 1),
        "reasons":   reasons,
        "atr":       round(at, 2),
    }

except Exception:
    return None
```

# –––––––––––––––––––––––––

# TELEGRAM

# –––––––––––––––––––––––––

def send_telegram(message):
token   = os.environ.get(“TELEGRAM_TOKEN”)
chat_id = os.environ.get(“TELEGRAM_CHAT_ID”)
if not token or not chat_id:
return
url    = “https://api.telegram.org/bot%s/sendMessage” % token
chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
for chunk in chunks:
try:
requests.post(url, data={“chat_id”: chat_id, “text”: chunk,
“parse_mode”: “Markdown”}, timeout=10)
except Exception as e:
print(“Erro Telegram: %s” % e)

def format_for_telegram(signals):
date_str = datetime.now().strftime(”%d/%m/%Y”)
lines = [
“*NASDAQ 100 - SINAIS DO DIA*”,
“*%s*” % date_str,
“”
]
for s in signals:
emoji = “BUY” if s[“dir”] == “LONG” else “SELL”
tipo  = “DAY” if s[“tipo”] == “DAY” else “SWING”
lines.append(”*%s* | %s | %s | Score: %s” % (s[“ticker”], emoji, tipo, s[“score”]))
lines.append(”  Entrada: `$%s` | Stop: `$%s` | Target: `$%s` | RR: `%sx`” % (
s[“entrada”], s[“stop”], s[“target”], s[“rr”]))
lines.append(””)
lines.append(”*Apenas referencia. Nao e aconselhamento financeiro.*”)
return “\n”.join(lines)

# –––––––––––––––––––––––––

# OUTPUT TERMINAL

# –––––––––––––––––––––––––

def print_header():
print(”\n” + “=”*70)
print(”  NASDAQ 100 - SCANNER DE SINAIS”)
print(”  %s” % datetime.now().strftime(”%d/%m/%Y %H:%M”))
print(”  APENAS PARA REFERENCIA - Nao e aconselhamento financeiro”)
print(”=”*70 + “\n”)

def print_signal(s, idx):
d_color = Fore.GREEN if s[“dir”] == “LONG” else Fore.RED
t_color = Fore.YELLOW if s[“tipo”] == “DAY” else Fore.CYAN
print(”-”*60)
print(”  #%-3d %-8s | %s | %s | Score: %s” % (
idx, s[“ticker”],
t_color + s[“tipo”] + Style.RESET_ALL,
d_color + s[“dir”]  + Style.RESET_ALL,
s[“score”]))
print(”       Preco actual : $%s” % s[“preco”])
print(”       Entrada      : $%s” % s[“entrada”])
print(”  >>   Target       : $%s” % s[“target”])
print(”  XX   Stop Loss    : $%s” % s[“stop”])
print(”       R/R: %sx  RSI: %s  Vol: %sx” % (s[“rr”], s[“rsi”], s[“vol_rel”]))
print(”       “ + “ | “.join(s[“reasons”]))

def print_table(signals):
rows = [[
s[“ticker”], s[“tipo”], s[“dir”],
“$%s” % s[“preco”], “$%s” % s[“entrada”],
“$%s” % s[“stop”],  “$%s” % s[“target”],
“%sx” % s[“rr”],    s[“score”]
] for s in signals]
headers = [“Ticker”,“Tipo”,“Dir”,“Preco”,“Entrada”,“Stop”,“Target”,“R/R”,“Score”]
print(tabulate(rows, headers=headers, tablefmt=“rounded_outline”))

# –––––––––––––––––––––––––

# MAIN

# –––––––––––––––––––––––––

def main():
parser = argparse.ArgumentParser(description=“Nasdaq 100 Trading Scanner”)
parser.add_argument(”–mode”,  choices=[“all”,“day”,“swing”], default=“all”)
parser.add_argument(”–top”,   type=int, default=15)
parser.add_argument(”–table”, action=“store_true”)
args = parser.parse_args()

```
telegram_mode = bool(os.environ.get("TELEGRAM_TOKEN"))

print_header()
print("  A analisar %d tickers... (pode demorar ~60s)\n" % len(NASDAQ100))

signals = []
for i, ticker in enumerate(NASDAQ100, 1):
    print("\r  Progresso: %d/%d - %s" % (i, len(NASDAQ100), ticker), end="", flush=True)
    result = analyse_ticker(ticker, mode=args.mode)
    if result:
        signals.append(result)

print("\r" + " "*50 + "\r", end="")

if not signals:
    print("  Nenhum sinal encontrado com os criterios actuais.")
    if telegram_mode:
        send_telegram("*NASDAQ 100 - %s*\n\nNenhum sinal encontrado hoje." %
                      datetime.now().strftime("%d/%m/%Y"))
    return

signals.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
signals = signals[:args.top]

if args.table:
    print_table(signals)
else:
    for idx, s in enumerate(signals, 1):
        print_signal(s, idx)

n_long  = sum(1 for s in signals if s["dir"] == "LONG")
n_short = sum(1 for s in signals if s["dir"] == "SHORT")
n_day   = sum(1 for s in signals if s["tipo"] == "DAY")
n_swing = sum(1 for s in signals if s["tipo"] == "SWING")

print("\n" + "="*60)
print("  SUMARIO: %d sinais | LONG: %d | SHORT: %d | DAY: %d | SWING: %d" % (
    len(signals), n_long, n_short, n_day, n_swing))
print("="*60)
print("\n  AVISO: Sinais baseados em analise tecnica.")
print("  Nao constituem aconselhamento financeiro.\n")

if telegram_mode:
    msg = format_for_telegram(signals)
    send_telegram(msg)
    print("  Mensagem enviada ao Telegram!\n")
```

if **name** == “**main**”:
main()
