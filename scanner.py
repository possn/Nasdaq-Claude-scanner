#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Trading Scanner v6 - Small/Micro Cap dinamico via ETF holdings
# Universo: Russell 2000 (IWM) + Micro Cap (IWC) - sempre actualizado
# Preco: $2-$50 | Volume min: 75k | Score min: 5.0 | Apenas BUY

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

# --------------------------------------------------
# PARAMETROS
# --------------------------------------------------
STOP_ATR_MULT = 1.5
TARGET_RR     = 2.0
MIN_SCORE     = 5.0
MIN_AVG_VOL   = 75_000
MIN_ATR_PCT   = 0.5
MIN_PRICE     = 2.0
MAX_PRICE     = 50.0
MAX_TICKERS   = 400   # limite para nao demorar mais de 5min no Actions

# ETFs fonte dos universos
ETF_SOURCES = {
    "russell2000": "IWM",    # iShares Russell 2000 - ~2000 small caps
    "microcap":    "IWC",    # iShares Micro Cap    - ~1400 micro caps
    "smallcap":    "SCHA",   # Schwab US Small Cap  - ~1700 small caps
}

# Fallback: lista estatica caso o download dos ETFs falhe
FALLBACK_TICKERS = [
    "RIVN","SIRI","LCID","HOOD","GRAB","HIMS","IONQ","JOBY","RKLB","BLNK",
    "CHPT","EVGO","MVIS","BYND","NKLA","CLOV","SPCE","STEM","CLNE","GEVO",
    "LAZR","OUST","OPEN","SKLZ","WKHS","XPEV","ZETA","MGNI","FSLY","LPSN",
    "DOMO","BIGC","AEHR","HEAR","GILT","MNKD","NVTS","MTTR","MVST","MARK",
    "AXSM","BEAM","BHVN","BNGO","ARCT","AGEN","AVIR","AMRX","ATRS","AVDL",
    "AMPY","CRGY","CDEV","ARCH","BATL","PRPL","UWMC","ORGN","RIDE","ZVIA",
    "COIN","RBLX","PTON","DKNG","MAPS","ACCD","EVBG","EXPI","CPSI","PESI",
    "ATIP","ATLC","MFIN","ACNB","IONQ","RKLB","AEHR","HEAR","GILT","BLNK",
]

# --------------------------------------------------
# OBTER TICKERS DOS ETFs DINAMICAMENTE
# --------------------------------------------------

def get_etf_tickers(etf_symbols, max_tickers=MAX_TICKERS):
    all_tickers = set()
    for name, symbol in etf_symbols.items():
        try:
            etf = yf.Ticker(symbol)
            holdings = etf.funds_data.top_holdings if hasattr(etf, 'funds_data') else None
            if holdings is not None and not holdings.empty:
                tickers = list(holdings.index)
                all_tickers.update(tickers)
                print("  ETF %s (%s): %d holdings obtidos" % (symbol, name, len(tickers)))
                continue
        except Exception:
            pass

        # Metodo alternativo: usar yf.download para obter constituintes
        try:
            etf_data = yf.Ticker(symbol)
            info = etf_data.info
            # Tentar via fast_info
            if hasattr(etf_data, 'funds_data'):
                fd = etf_data.funds_data
                if hasattr(fd, 'top_holdings') and fd.top_holdings is not None:
                    tickers = list(fd.top_holdings.index)
                    all_tickers.update(tickers)
                    print("  ETF %s (%s): %d holdings obtidos" % (symbol, name, len(tickers)))
        except Exception as e:
            print("  ETF %s: falhou (%s)" % (symbol, str(e)[:50]))

    # Se nao conseguiu tickers suficientes, usar fallback
    if len(all_tickers) < 50:
        print("  Usando lista fallback (%d tickers)" % len(FALLBACK_TICKERS))
        return FALLBACK_TICKERS

    # Filtrar tickers validos (sem caracteres especiais, 1-5 letras)
    clean = [t for t in all_tickers
             if isinstance(t, str) and t.isalpha() and 1 <= len(t) <= 5]

    # Limitar ao maximo definido (aleatoriamente para variar dia a dia)
    if len(clean) > max_tickers:
        import random
        random.seed(datetime.now().toordinal())  # seed pelo dia - mesmo resultado no mesmo dia
        clean = random.sample(clean, max_tickers)

    print("  Total universo: %d tickers unicos" % len(clean))
    return sorted(clean)

# --------------------------------------------------
# INDICADORES
# --------------------------------------------------

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
    hist   = macd - signal
    return macd, signal, hist

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

# --------------------------------------------------
# ANALISE POR TICKER
# --------------------------------------------------

def analyse_ticker(ticker, mode="all"):
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 40:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        avg_vol = volume.iloc[-50:].mean()
        if avg_vol < MIN_AVG_VOL:
            return None

        c = float(close.iloc[-1])
        if c < MIN_PRICE or c > MAX_PRICE:
            return None

        rsi                    = compute_rsi(close)
        macd, macd_sig, mhist  = compute_macd(close)
        atr                    = compute_atr(high, low, close)
        bb_up, bb_mid, bb_low  = compute_bb(close)
        adx, plus_di, minus_di = compute_adx(high, low, close)
        ema20  = close.ewm(span=20, adjust=False).mean()
        ema50  = close.ewm(span=50, adjust=False).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        r     = float(rsi.iloc[-1])
        r_1   = float(rsi.iloc[-2])
        m     = float(macd.iloc[-1])
        ms    = float(macd_sig.iloc[-1])
        mh    = float(mhist.iloc[-1])
        mh_1  = float(mhist.iloc[-2])
        at    = float(atr.iloc[-1])
        bbu   = float(bb_up.iloc[-1])
        bbl   = float(bb_low.iloc[-1])
        adxv  = float(adx.iloc[-1])
        pdi   = float(plus_di.iloc[-1])
        mdi   = float(minus_di.iloc[-1])
        e20   = float(ema20.iloc[-1])
        e50   = float(ema50.iloc[-1])
        o     = float(df["Open"].iloc[-1])
        vol_ratio = float(volume.iloc[-1]) / float(avg_vol)

        atr_pct = (at / c) * 100
        if atr_pct < MIN_ATR_PCT:
            return None

        s50       = float(sma50.iloc[-1])
        s200_vals = sma200.dropna()
        if len(s200_vals) >= 20:
            golden_cross = s50 > float(s200_vals.iloc[-1])
        else:
            golden_cross = c > e50

        above_ema20    = (close.iloc[-10:] > ema20.iloc[-10:]).sum() / 10.0
        bullish_candle = c > o

        # Score ponderado LONG
        score   = 0.0
        reasons = []

        if 35 < r < 55:
            score += 1.5
            reasons.append("RSI %.0f zona bullish" % r)
        elif 55 <= r < 65:
            score += 0.5
            reasons.append("RSI %.0f aceitavel" % r)

        if r > r_1:
            score += 0.5
            reasons.append("RSI subindo")

        if m > ms and mh > mh_1:
            score += 2.0
            reasons.append("MACD bullish + histograma crescente")
        elif m > ms:
            score += 1.0
            reasons.append("MACD acima do sinal")

        if c > e20 > e50:
            score += 2.0
            reasons.append("Preco > EMA20 > EMA50")
        elif c > e20:
            score += 1.0
            reasons.append("Preco > EMA20")

        if golden_cross:
            score += 1.5
            reasons.append("Tendencia macro bullish")

        if adxv > 25 and pdi > mdi:
            score += 1.5
            reasons.append("ADX %.0f forte bullish" % adxv)
        elif adxv > 20 and pdi > mdi:
            score += 0.75
            reasons.append("ADX %.0f moderado bullish" % adxv)

        if vol_ratio > 2.0:
            score += 1.0
            reasons.append("Volume %.1fx confirmacao forte" % vol_ratio)
        elif vol_ratio > 1.3:
            score += 0.5
            reasons.append("Volume %.1fx acima media" % vol_ratio)

        if bullish_candle:
            score += 0.5
            reasons.append("Candle bullish")

        if above_ema20 >= 0.7:
            score += 0.5
            reasons.append("%.0f%% dias acima EMA20" % (above_ema20 * 100))

        warnings = []
        if c > bbu * 0.97:
            score -= 1.5
            warnings.append("Perto resistencia BB")

        if score < MIN_SCORE:
            return None

        is_day     = (r < 35) and (vol_ratio > 1.5) and (c < bbl * 1.01)
        trade_type = "DAY" if is_day else "SWING"

        if mode == "day"   and trade_type != "DAY":   return None
        if mode == "swing" and trade_type != "SWING": return None

        entry  = c
        stop   = round(entry - at * STOP_ATR_MULT, 2)
        risk   = entry - stop
        target = round(entry + risk * TARGET_RR, 2)
        rr     = round(abs(target - entry) / abs(stop - entry), 2)

        return {
            "ticker":   ticker,
            "tipo":     trade_type,
            "preco":    round(c, 2),
            "entrada":  round(entry, 2),
            "stop":     stop,
            "target":   round(target, 2),
            "rr":       rr,
            "score":    round(score, 1),
            "vol_rel":  round(vol_ratio, 2),
            "rsi":      round(r, 1),
            "atr_pct":  round(atr_pct, 2),
            "reasons":  reasons,
            "warnings": warnings,
        }

    except Exception:
        return None

# --------------------------------------------------
# TELEGRAM
# --------------------------------------------------

def send_telegram(message):
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url    = "https://api.telegram.org/bot%s/sendMessage" % token
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, data={"chat_id": chat_id, "text": chunk,
                                     "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print("Erro Telegram: %s" % e)

def format_for_telegram(signals, n_scanned):
    date_str = datetime.now().strftime("%d/%m/%Y")
    lines = [
        "*SMALL/MICRO CAP - SINAIS DO DIA*",
        "_%s | %d acoes analisadas_" % (date_str, n_scanned),
        ""
    ]
    for s in signals:
        tipo = "DAY" if s["tipo"] == "DAY" else "SWING"
        warn = " [!]" if s["warnings"] else ""
        lines.append("*%s* | BUY | %s | Score: %.1f%s" % (
            s["ticker"], tipo, s["score"], warn))
        lines.append("  Entrada: `$%s` | Stop: `$%s` | Target: `$%s` | RR: `%sx`" % (
            s["entrada"], s["stop"], s["target"], s["rr"]))
        lines.append("  RSI: `%s` | Vol: `%sx` | ATR: `%s%%`" % (
            s["rsi"], s["vol_rel"], s["atr_pct"]))
        lines.append("")
    lines.append("_Apenas referencia. Nao e aconselhamento financeiro._")
    return "\n".join(lines)

# --------------------------------------------------
# OUTPUT TERMINAL
# --------------------------------------------------

def print_header(n_tickers):
    print("\n" + "="*70)
    print("  SCANNER v6 - Small/Micro Cap | Preco: $%.0f-$%.0f | Vol min: %dk" % (
        MIN_PRICE, MAX_PRICE, MIN_AVG_VOL // 1000))
    print("  %s | Fonte: Russell 2000 (IWM) + Micro Cap (IWC)" % (
        datetime.now().strftime("%d/%m/%Y %H:%M")))
    print("="*70 + "\n")

def print_signal(s, idx):
    t_color = Fore.YELLOW if s["tipo"] == "DAY" else Fore.CYAN
    w_str   = (" | AVISO: " + " | ".join(s["warnings"])) if s["warnings"] else ""
    print("-"*65)
    print("  #%-3d %-8s | BUY | %s | Score: %.1f/10%s" % (
        idx, s["ticker"],
        t_color + s["tipo"] + Style.RESET_ALL,
        s["score"],
        Fore.YELLOW + w_str + Style.RESET_ALL))
    print("       Preco  : $%s  (ATR: %s%%)" % (s["preco"], s["atr_pct"]))
    print("  >>   Target : $%s" % s["target"])
    print("  XX   Stop   : $%s" % s["stop"])
    print("       R/R: %sx  RSI: %s  Vol: %sx" % (s["rr"], s["rsi"], s["vol_rel"]))
    print("       " + " | ".join(s["reasons"]))

def print_table(signals):
    rows = [[
        s["ticker"], s["tipo"],
        "$%s" % s["preco"], "$%s" % s["stop"],
        "$%s" % s["target"], "%sx" % s["rr"],
        "%.1f" % s["score"], "%s%%" % s["atr_pct"]
    ] for s in signals]
    headers = ["Ticker","Tipo","Preco","Stop","Target","R/R","Score","ATR%"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trading Scanner v6")
    parser.add_argument("--mode",  choices=["all","day","swing"], default="all")
    parser.add_argument("--top",   type=int, default=10)
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    telegram_mode = bool(os.environ.get("TELEGRAM_TOKEN"))

    print_header(0)
    print("  A obter universo de tickers via ETFs...\n")

    tickers = get_etf_tickers(ETF_SOURCES, MAX_TICKERS)

    print("\n  A analisar %d tickers...\n" % len(tickers))

    signals   = []
    n_scanned = 0
    for i, ticker in enumerate(tickers, 1):
        print("\r  Progresso: %d/%d - %-8s | Sinais: %d" % (
            i, len(tickers), ticker, len(signals)),
            end="", flush=True)
        result = analyse_ticker(ticker, mode=args.mode)
        n_scanned += 1
        if result:
            signals.append(result)

    print("\r" + " "*65 + "\r", end="")

    if not signals:
        print("  Nenhum sinal encontrado hoje.")
        if telegram_mode:
            send_telegram("*SMALL CAP SCANNER - %s*\n\n_%d acoes analisadas._\nNenhum sinal encontrado hoje." % (
                datetime.now().strftime("%d/%m/%Y"), n_scanned))
        return

    signals.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    top_signals = signals[:args.top]

    if args.table:
        print_table(top_signals)
    else:
        for idx, s in enumerate(top_signals, 1):
            print_signal(s, idx)

    n_day  = sum(1 for s in top_signals if s["tipo"] == "DAY")
    n_sw   = sum(1 for s in top_signals if s["tipo"] == "SWING")
    n_warn = sum(1 for s in top_signals if s["warnings"])

    print("\n" + "="*65)
    print("  SUMARIO: %d sinais BUY (de %d analisadas) | DAY: %d | SWING: %d | Avisos: %d" % (
        len(signals), n_scanned, n_day, n_sw, n_warn))
    print("="*65)
    print("  Nao constitui aconselhamento financeiro.\n")

    if telegram_mode:
        msg = format_for_telegram(top_signals, n_scanned)
        send_telegram(msg)
        print("  Mensagem enviada ao Telegram!\n")

if __name__ == "__main__":
    main()
