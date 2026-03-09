#!/usr/bin/env python3
“””
Nasdaq 100 — Daily Trading Signal Scanner
Gera sugestões de day trading e swing trading (até 1 semana).
Dados: Yahoo Finance (gratuito, sem API key necessária)

Uso local:
pip install yfinance pandas numpy colorama tabulate requests
python scanner.py
python scanner.py –mode swing
python scanner.py –top 10

Telegram:
Definir variáveis de ambiente TELEGRAM_TOKEN e TELEGRAM_CHAT_ID
(feito automaticamente pelo GitHub Actions)
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

# ─────────────────────────────────────────────

# NASDAQ 100 — tickers

# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────

# PARÂMETROS DE RISCO

# ─────────────────────────────────────────────

STOP_ATR_MULT = 1.5    # Stop loss = entrada ± (ATR × mult)
TARGET_RR     = 2.0    # Risk/Reward mínimo para mostrar sinal
MIN_SCORE     = 3      # Score mínimo (em 5) para incluir sinal
MIN_AVG_VOL   = 1_000_000  # Volume médio diário mínimo (liquidez)

# ─────────────────────────────────────────────

# INDICADORES TÉCNICOS

# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
delta = series.diff()
gain  = delta.clip(lower=0).rolling(period).mean()
loss  = (-delta.clip(upper=0)).rolling(period).mean()
rs    = gain / loss.replace(0, np.nan)
return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period: int = 14) -> pd.Series:
tr = pd.concat([
high - low,
(high - close.shift()).abs(),
(low  - close.shift()).abs()
], axis=1).max(axis=1)
return tr.rolling(period).mean()

def compute_macd(series: pd.Series):
ema12  = series.ewm(span=12, adjust=False).mean()
ema26  = series.ewm(span=26, adjust=False).mean()
macd   = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
return macd, signal

def compute_bb(series: pd.Series, period: int = 20, std: float = 2.0):
mid   = series.rolling(period).mean()
sigma = series.rolling(period).std()
return mid + std * sigma, mid, mid - std * sigma

def compute_adx(high, low, close, period: int = 14):
up   = high.diff()
down = -low.diff()
plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
minus_dm = np.where((down > up) & (down > 0), down, 0.0)
atr      = compute_atr(high, low, close, period)
plus_di  = 100 * pd.Series(plus_dm,  index=close.index).rolling(period).mean() / atr
minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).mean() / atr
dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
return dx.rolling(period).mean(), plus_di, minus_di

# ─────────────────────────────────────────────

# ANÁLISE POR TICKER

# ─────────────────────────────────────────────

def analyse_ticker(ticker: str, mode: str = “all”) -> dict | None:
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
        score_long += 1; reasons_long.append(f"RSI {r:.0f} (zona bullish)")
    if m > ms:
        score_long += 1; reasons_long.append("MACD acima do sinal")
    if c > e20 > e50:
        score_long += 1; reasons_long.append("Preço > EMA20 > EMA50")
    if adxv > 20 and pdi > mdi:
        score_long += 1; reasons_long.append(f"ADX {adxv:.0f} — tendência bullish")
    if vol_ratio > 1.3:
        score_long += 1; reasons_long.append(f"Volume {vol_ratio:.1f}x acima da média")

    # Scoring SHORT
    score_short, reasons_short = 0, []
    if 35 < r < 60:
        score_short += 1; reasons_short.append(f"RSI {r:.0f} (zona bearish)")
    if m < ms:
        score_short += 1; reasons_short.append("MACD abaixo do sinal")
    if c < e20 < e50:
        score_short += 1; reasons_short.append("Preço < EMA20 < EMA50")
    if adxv > 20 and mdi > pdi:
        score_short += 1; reasons_short.append(f"ADX {adxv:.0f} — tendência bearish")
    if vol_ratio > 1.3:
        score_short += 1; reasons_short.append(f"Volume {vol_ratio:.1f}x acima da média")

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
        "direcção":  direction,
        "preço":     round(c, 2),
        "entrada":   round(entry, 2),
        "stop":      stop,
        "target":    round(target, 2),
        "R/R":       rr,
        "score":     f"{score}/5",
        "vol_rel":   round(vol_ratio, 2),
        "rsi":       round(r, 1),
        "reasons":   reasons,
        "atr":       round(at, 2),
    }

except Exception:
    return None
```

# ─────────────────────────────────────────────

# TELEGRAM

# ─────────────────────────────────────────────

def send_telegram(message: str) -> None:
“”“Envia mensagem para o Telegram via bot.”””
token   = os.environ.get(“TELEGRAM_TOKEN”)
chat_id = os.environ.get(“TELEGRAM_CHAT_ID”)
if not token or not chat_id:
return
url = f”https://api.telegram.org/bot{token}/sendMessage”
# Telegram tem limite de 4096 chars por mensagem — dividir se necessário
chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
for chunk in chunks:
try:
requests.post(url, data={“chat_id”: chat_id, “text”: chunk, “parse_mode”: “Markdown”}, timeout=10)
except Exception as e:
print(f”  ⚠️  Erro ao enviar Telegram: {e}”)

def format_for_telegram(signals: list) -> str:
“”“Formata os sinais para mensagem Markdown no Telegram.”””
date_str = datetime.now().strftime(”%d/%m/%Y”)
lines = [
f”*📊 NASDAQ 100 — SINAIS DO DIA*”,
f”*{date_str}*”,
“”
]
for s in signals:
emoji_dir  = “🟢” if s[“direcção”] == “LONG” else “🔴”
emoji_type = “⚡” if s[“tipo”] == “DAY” else “📅”
lines.append(f”{emoji_dir} *{s[‘ticker’]}*  {emoji_type} {s[‘tipo’]}  |  Score: {s[‘score’]}”)
lines.append(f”  Entrada: `${s['entrada']}`  |  Stop: `${s['stop']}`  |  Target: `${s['target']}`”)
lines.append(f”  R/R: `{s['R/R']}x`  |  RSI: `{s['rsi']}`  |  Vol: `{s['vol_rel']}x`”)
lines.append(””)
lines.append(”*⚠️ Apenas referência. Não é aconselhamento financeiro.*”)
return “\n”.join(lines)

# ─────────────────────────────────────────────

# OUTPUT TERMINAL

# ─────────────────────────────────────────────

def print_header():
print(”\n” + “═”*70)
print(f”  📊  NASDAQ 100 — SCANNER DE SINAIS”)
print(f”  📅  {datetime.now().strftime(’%d/%m/%Y %H:%M’)}”)
print(f”  ⚠️   APENAS PARA REFERÊNCIA — Não é aconselhamento financeiro”)
print(“═”*70 + “\n”)

def print_signal(s, idx):
d_color = Fore.GREEN if s[“direcção”] == “LONG” else Fore.RED
t_color = Fore.YELLOW if s[“tipo”] == “DAY” else Fore.CYAN
print(f”{‘─’*60}”)
print(f”  #{idx:<3} {Style.BRIGHT}{s[‘ticker’]:<8}{Style.RESET_ALL}”
f” | {t_color}{s[‘tipo’]}{Style.RESET_ALL}”
f” | {d_color}{s[‘direcção’]}{Style.RESET_ALL}”
f” | Score: {s[‘score’]}”)
print(f”       Preço actual : ${s[‘preço’]}”)
print(f”       Entrada      : ${s[‘entrada’]}”)
print(f”  🟢   Target       : ${s[‘target’]}”)
print(f”  🔴   Stop Loss    : ${s[‘stop’]}”)
print(f”       R/R          : {s[‘R/R’]}x  |  RSI: {s[‘rsi’]}  |  Vol: {s[‘vol_rel’]}x”)
print(f”  📌  {’ | ’.join(s[‘reasons’])}”)

def print_table(signals):
rows = [[
s[“ticker”], s[“tipo”], s[“direcção”],
f”${s[‘preço’]}”, f”${s[‘entrada’]}”, f”${s[‘stop’]}”,
f”${s[‘target’]}”, f”{s[‘R/R’]}x”, s[“score”]
] for s in signals]
headers = [“Ticker”,“Tipo”,“Dir.”,“Preço”,“Entrada”,“Stop”,“Target”,“R/R”,“Score”]
print(tabulate(rows, headers=headers, tablefmt=“rounded_outline”))

# ─────────────────────────────────────────────

# MAIN

# ─────────────────────────────────────────────

def main():
parser = argparse.ArgumentParser(description=“Nasdaq 100 Trading Scanner”)
parser.add_argument(”–mode”,  choices=[“all”,“day”,“swing”], default=“all”)
parser.add_argument(”–top”,   type=int, default=15)
parser.add_argument(”–table”, action=“store_true”)
args = parser.parse_args()

```
telegram_mode = bool(os.environ.get("TELEGRAM_TOKEN"))

print_header()
print(f"  🔍  A analisar {len(NASDAQ100)} tickers... (pode demorar ~60s)\n")

signals = []
for i, ticker in enumerate(NASDAQ100, 1):
    print(f"\r  Progresso: {i}/{len(NASDAQ100)} — {ticker:<8}", end="", flush=True)
    result = analyse_ticker(ticker, mode=args.mode)
    if result:
        signals.append(result)

print("\r" + " "*50 + "\r", end="")

if not signals:
    print(Fore.YELLOW + "  Nenhum sinal encontrado com os critérios actuais.")
    if telegram_mode:
        send_telegram(f"📊 *NASDAQ 100 — {datetime.now().strftime('%d/%m/%Y')}*\n\n_Nenhum sinal encontrado hoje._")
    return

signals.sort(key=lambda x: (x["score"], x["R/R"]), reverse=True)
signals = signals[:args.top]

if args.table:
    print_table(signals)
else:
    for idx, s in enumerate(signals, 1):
        print_signal(s, idx)

n_long  = sum(1 for s in signals if s["direcção"] == "LONG")
n_short = sum(1 for s in signals if s["direcção"] == "SHORT")
n_day   = sum(1 for s in signals if s["tipo"] == "DAY")
n_swing = sum(1 for s in signals if s["tipo"] == "SWING")

print(f"\n{'═'*60}")
print(f"  SUMÁRIO: {len(signals)} sinais  |  "
      f"{Fore.GREEN}LONG: {n_long}{Style.RESET_ALL}  |  "
      f"{Fore.RED}SHORT: {n_short}{Style.RESET_ALL}  |  "
      f"{Fore.YELLOW}DAY: {n_day}{Style.RESET_ALL}  |  "
      f"{Fore.CYAN}SWING: {n_swing}{Style.RESET_ALL}")
print(f"{'═'*60}\n")
print("  ⚠️  Estes sinais são baseados em análise técnica e não")
print("      constituem aconselhamento financeiro.\n")

# Enviar para Telegram se variáveis configuradas
if telegram_mode:
    msg = format_for_telegram(signals)
    send_telegram(msg)
    print("  ✅  Mensagem enviada ao Telegram!\n")
```

if **name** == “**main**”:
main()
