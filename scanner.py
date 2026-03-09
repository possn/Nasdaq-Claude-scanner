#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Trading Scanner v3 - Nasdaq100 / SmallCap - Apenas sinais BUY

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
# UNIVERSOS
# --------------------------------------------------

NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "ASML","NFLX","AMD","AZN","QCOM","CSCO","TMUS","INTU","LIN","AMGN",
    "ISRG","AMAT","TXN","BKNG","VRTX","MU","REGN","ADI","PANW","LRCX",
    "SBUX","MDLZ","SNPS","KLAC","CDNS","MELI","GILD","CRWD","CTAS","PDD",
    "CEG","INTC","ABNB","ORLY","MAR","ADSK","FTNT","DXCM","ROP","MNST",
    "WDAY","ROST","ADP","PCAR","MRVL","CHTR","ODFL","CPRT","MCHP","KDP",
    "PAYX","FANG","EXC","CTSH","NXPI","EA","FAST","IDXX","VRSK","GEHC",
    "CSGP","XEL","DDOG","WBD","ZS","SGEN","TEAM","ON","GFS","ANSS",
    "LULU","DLTR","BIIB","TTWO","ILMN","WBA","LCID","RIVN","ENPH","ALGN",
    "SIRI","MTCH","SWKS","OKTA","ZM","DOCU","PTON","COIN","HOOD","RBLX"
]

# Lista curada ~150 small/micro caps liquidaveis (volume medio > 100k/dia)
# Criterios de seleccao: listadas NYSE/Nasdaq, market cap $50M-$2B,
# volume medio diario > 100k accoes, sem OTC/pink sheets
SMALLCAP = [
    # Tecnologia / Software
    "ARQT","BIGC","BLNK","BYRN","CABA","CARE","CLFD","CNXC","CODA","COMS",
    "CPSI","CSSE","DOMO","DTRT","EVBG","EXPI","FSLY","GFAI","GILT","HEAR",
    "HIMS","IMVT","INPX","IONQ","JOBY","KALA","LIDR","LPSN","MAPS","MARK",
    "MFIN","MGNI","MIMO","MKTW","MNKD","MTTR","MVST","NKLA","NRDS","NTST",
    "NVTS","OPEN","OPAD","OPTX","ORMP","OTRK","OWLT","PAYO","PESI","PHVS",
    # Saude / Biotech small cap liquidaveis
    "ACCD","AGEN","ALDX","ALEC","ALLO","ALNY","ALTO","ALVR","AMRX","APLS",
    "ARCT","ARDX","ARHS","ARIX","ARQT","ATEX","ATRC","ATRS","AVAH","AVDL",
    "AVIR","AXSM","BCAB","BCYC","BDSX","BEAM","BHVN","BNGO","BOLD","BPMC",
    # Energia / Materiais
    "AMPY","ARCH","BATL","BCEI","CDEV","CLNE","CLOV","CNSL","CODI","COHN",
    "COOP","CORT","CPNG","CRGY","CRIS","CRSR","CSTE","CTRN","CUTR","CYBN",
    # Financas / REIT small cap
    "AAME","ACNB","AGIO","AGMH","AGNCN","AGRX","AGYS","AHCO","AHPI","AINC",
    "AIXI","AKBA","AKRO","AKTS","ALAB","ALBO","ALCO","ALEC","ALEX","ALGT",
    # Consumo / Retalho
    "ACMR","ACRX","ACST","ACTG","ADAP","ADBE","ADCT","ADIL","ADMA","ADMP",
    "ADNT","ADOC","ADPT","ADSE","ADTX","ADUS","ADVM","ADXN","AEAC","AEHR",
    # Industriais
    "AEIS","AELX","AENZ","AEYE","AEZS","AFAR","AFBI","AFCG","AFIB","AFMD",
    "AFRI","AFYA","AGAE","AGBA","AGFS","AGFY","AGHI","AGIL","AGIO","AGIIL",
    # Lista adicional de small caps conhecidas e liquidaveis
    "ARCO","ATER","ATIF","ATIP","ATLC","ATLX","ATMC","ATMS","ATMU","ATNF",
    "ATOM","ATPC","ATRA","ATRC","ATRI","ATRM","ATRS","ATSG","ATTO","ATVI",
    "AUID","AULT","AUMN","AUNA","AUPH","AURX","AUTL","AUVI","AUXO","AVAH",
    "AVAV","AVDL","AVEO","AVER","AVGO","AVHI","AVID","AVIR","AVNS","AVPT",
    # Mais small caps liquidas conhecidas
    "BYND","CHPT","CLOV","DFLI","DKNG","EVGO","GEVO","GOEV","GRAB","HIMS",
    "IONQ","JOBY","LAZR","LIDR","LUCY","MAPS","MVIS","NKLA","OPEN","ORGN",
    "OUST","PAYA","PNTM","PRPL","RIDE","RKLB","SKLZ","SPCE","STEM","UWMC",
    "VIEW","WKHS","XPEV","ZETA","ZVIA"
]

UNIVERSES = {
    "nasdaq100": NASDAQ100,
    "smallcap":  SMALLCAP,
    "all":       list(set(NASDAQ100 + SMALLCAP))
}

# --------------------------------------------------
# PARAMETROS
# --------------------------------------------------
STOP_ATR_MULT    = 1.5
TARGET_RR        = 2.0
MIN_SCORE        = 5.0    # score ponderado minimo (escala 0-10)
MIN_AVG_VOL      = 100_000
MIN_ATR_PCT      = 0.5    # volatilidade minima: ATR deve ser >= 0.5% do preco
MIN_PRICE        = 2.0    # preco minimo por accao (elimina penny stocks)
MAX_PRICE        = 30.0   # preco maximo por accao

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
# ANALISE
# --------------------------------------------------

def analyse_ticker(ticker, mode="all"):
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # Volume medio sobre 50 dias (mais estavel que 20)
        avg_vol = volume.iloc[-50:].mean()
        if avg_vol < MIN_AVG_VOL:
            return None

        # Indicadores
        rsi                    = compute_rsi(close)
        macd, macd_sig, mhist  = compute_macd(close)
        atr                    = compute_atr(high, low, close)
        bb_up, bb_mid, bb_low  = compute_bb(close)
        adx, plus_di, minus_di = compute_adx(high, low, close)

        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        # Valores actuais
        c     = float(close.iloc[-1])
        c_1   = float(close.iloc[-2])   # fecho anterior
        o     = float(df["Open"].iloc[-1])
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
        s50   = float(sma50.iloc[-1])
        s200  = float(sma200.iloc[-1]) if len(close) >= 200 else None
        vol_ratio = float(volume.iloc[-1]) / float(avg_vol)

        # Filtro: preco (eliminar penny stocks e accoes caras)
        if c < MIN_PRICE or c > MAX_PRICE:
            return None

        # Filtro: volatilidade minima
        atr_pct = (at / c) * 100
        if atr_pct < MIN_ATR_PCT:
            return None

        # Consistencia de tendencia: % de dias acima de EMA20 nos ultimos 10 dias
        above_ema20 = (close.iloc[-10:] > ema20.iloc[-10:]).sum() / 10.0
        below_ema20 = 1.0 - above_ema20

        # Confirmacao de candle:
        # LONG: fecho > abertura (candle verde)
        # SHORT: fecho < abertura (candle vermelho)
        bullish_candle = c > o
        bearish_candle = c < o

        # Tendencia macro
        golden_cross = (s200 is not None) and (s50 > s200)
        death_cross  = (s200 is not None) and (s50 < s200)
        # Se sem sma200 (menos de 200 dias de dados), usar ema50 vs preco
        if s200 is None:
            golden_cross = c > e50
            death_cross  = c < e50

        # ── SCORE PONDERADO LONG (escala 0-10) ───────────────
        # Cada criterio tem peso diferente baseado em evidencia empirica
        score_long = 0.0
        reasons_long = []

        # RSI: zona 35-55 ideal para entrada LONG (nao sobrecomprado)
        # Peso 1.5 — RSI e o indicador mais discriminante em swing
        if 35 < r < 55:
            score_long += 1.5
            reasons_long.append("RSI %.0f (zona entrada long)" % r)
        elif 55 <= r < 65:
            score_long += 0.5
            reasons_long.append("RSI %.0f (aceitavel)" % r)

        # RSI em ascensao (momentum crescente)
        if r > r_1:
            score_long += 0.5
            reasons_long.append("RSI subindo")

        # MACD: cruzamento recente ou histograma a crescer — peso 2.0
        if m > ms and mh > mh_1:
            score_long += 2.0
            reasons_long.append("MACD bullish + histograma crescente")
        elif m > ms:
            score_long += 1.0
            reasons_long.append("MACD acima do sinal")

        # Estrutura de tendencia EMA — peso 2.0
        if c > e20 > e50:
            score_long += 2.0
            reasons_long.append("Preco > EMA20 > EMA50")
        elif c > e20:
            score_long += 1.0
            reasons_long.append("Preco > EMA20")

        # Tendencia macro (SMA50 vs SMA200) — peso 1.5
        if golden_cross:
            score_long += 1.5
            reasons_long.append("SMA50 > SMA200 (uptrend macro)")

        # ADX forte + DI+ dominante — peso 1.5
        if adxv > 25 and pdi > mdi:
            score_long += 1.5
            reasons_long.append("ADX %.0f forte bullish" % adxv)
        elif adxv > 20 and pdi > mdi:
            score_long += 0.75
            reasons_long.append("ADX %.0f moderado bullish" % adxv)

        # Volume relativo — peso 1.0
        if vol_ratio > 2.0:
            score_long += 1.0
            reasons_long.append("Volume %.1fx (forte confirmacao)" % vol_ratio)
        elif vol_ratio > 1.3:
            score_long += 0.5
            reasons_long.append("Volume %.1fx acima media" % vol_ratio)

        # Confirmacao de candle — peso 0.5
        if bullish_candle:
            score_long += 0.5
            reasons_long.append("Candle de confirmacao bullish")

        # Consistencia: maioria dos dias acima de EMA20 — peso 0.5
        if above_ema20 >= 0.7:
            score_long += 0.5
            reasons_long.append("%.0f%% dias acima EMA20" % (above_ema20 * 100))

        # Penalizacao: preco perto de resistencia (BB superior)
        if c > bbu * 0.97:
            score_long -= 1.5
            reasons_long.append("[!] Perto de resistencia BB")

        # ── SCORE PONDERADO SHORT (escala 0-10) ──────────────
        score_short = 0.0
        reasons_short = []

        if 45 < r < 65:
            score_short += 1.5
            reasons_short.append("RSI %.0f (zona entrada short)" % r)
        elif 35 <= r <= 45:
            score_short += 0.5
            reasons_short.append("RSI %.0f (aceitavel)" % r)

        if r < r_1:
            score_short += 0.5
            reasons_short.append("RSI descendo")

        if m < ms and mh < mh_1:
            score_short += 2.0
            reasons_short.append("MACD bearish + histograma decrescente")
        elif m < ms:
            score_short += 1.0
            reasons_short.append("MACD abaixo do sinal")

        if c < e20 < e50:
            score_short += 2.0
            reasons_short.append("Preco < EMA20 < EMA50")
        elif c < e20:
            score_short += 1.0
            reasons_short.append("Preco < EMA20")

        if death_cross:
            score_short += 1.5
            reasons_short.append("SMA50 < SMA200 (downtrend macro)")

        if adxv > 25 and mdi > pdi:
            score_short += 1.5
            reasons_short.append("ADX %.0f forte bearish" % adxv)
        elif adxv > 20 and mdi > pdi:
            score_short += 0.75
            reasons_short.append("ADX %.0f moderado bearish" % adxv)

        if vol_ratio > 2.0:
            score_short += 1.0
            reasons_short.append("Volume %.1fx (forte confirmacao)" % vol_ratio)
        elif vol_ratio > 1.3:
            score_short += 0.5
            reasons_short.append("Volume %.1fx acima media" % vol_ratio)

        if bearish_candle:
            score_short += 0.5
            reasons_short.append("Candle de confirmacao bearish")

        if below_ema20 >= 0.7:
            score_short += 0.5
            reasons_short.append("%.0f%% dias abaixo EMA20" % (below_ema20 * 100))

        if c < bbl * 1.03:
            score_short -= 1.5
            reasons_short.append("[!] Perto de suporte BB")

        # ── SELECCAO ─────────────────────────────────────────
        is_day_long = (r < 35) and (vol_ratio > 1.5) and (c < bbl * 1.01)

        # Apenas sinais de compra (LONG)
        if score_long >= MIN_SCORE:
            direction  = "LONG"
            score      = score_long
            reasons    = reasons_long
            trade_type = "DAY" if is_day_long else "SWING"
            entry      = c
            stop       = round(entry - at * STOP_ATR_MULT, 2)
            risk       = entry - stop
            target     = round(entry + risk * TARGET_RR, 2)
        else:
            return None

        if mode == "day"   and trade_type != "DAY":   return None
        if mode == "swing" and trade_type != "SWING": return None

        rr      = round(abs(target - entry) / abs(stop - entry), 2)
        atr_pct = round((at / c) * 100, 2)

        return {
            "ticker":    ticker,
            "tipo":      trade_type,
            "dir":       direction,
            "preco":     round(c, 2),
            "entrada":   round(entry, 2),
            "stop":      stop,
            "target":    round(target, 2),
            "rr":        rr,
            "score":     round(score, 1),
            "vol_rel":   round(vol_ratio, 2),
            "rsi":       round(r, 1),
            "atr_pct":   atr_pct,
            "reasons":   [r for r in reasons if not r.startswith("[!")],
            "warnings":  [r for r in reasons if r.startswith("[!")],
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

def format_for_telegram(signals):
    date_str = datetime.now().strftime("%d/%m/%Y")
    lines = [
        "*NASDAQ / SMALL CAP - SINAIS DO DIA*",
        "_%s_" % date_str,
        ""
    ]
    for s in signals:
        emoji = "BUY" if s["dir"] == "LONG" else "SELL"
        tipo  = "DAY" if s["tipo"] == "DAY" else "SWING"
        warn  = " [!]" if s["warnings"] else ""
        lines.append("*%s* | %s | %s | Score: %.1f%s" % (
            s["ticker"], emoji, tipo, s["score"], warn))
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

def print_header(universe_name, n_tickers):
    print("\n" + "="*70)
    print("  SCANNER DE SINAIS v2 - %s (%d tickers)" % (universe_name.upper(), n_tickers))
    print("  %s" % datetime.now().strftime("%d/%m/%Y %H:%M"))
    print("  Score minimo: %.1f/10 | Vol minimo: %dk/dia | ATR minimo: %.1f%%" % (
        MIN_SCORE, MIN_AVG_VOL // 1000, MIN_ATR_PCT))
    print("="*70 + "\n")

def print_signal(s, idx):
    d_color = Fore.GREEN if s["dir"] == "LONG" else Fore.RED
    t_color = Fore.YELLOW if s["tipo"] == "DAY" else Fore.CYAN
    w_str   = (" | AVISO: " + " | ".join(s["warnings"])) if s["warnings"] else ""
    print("-"*65)
    print("  #%-3d %-8s | %s | %s | Score: %.1f/10%s" % (
        idx, s["ticker"],
        t_color + s["tipo"] + Style.RESET_ALL,
        d_color + s["dir"]  + Style.RESET_ALL,
        s["score"],
        Fore.YELLOW + w_str + Style.RESET_ALL))
    print("       Preco actual : $%s  (ATR: %s%%)" % (s["preco"], s["atr_pct"]))
    print("       Entrada      : $%s" % s["entrada"])
    print("  >>   Target       : $%s" % s["target"])
    print("  XX   Stop Loss    : $%s" % s["stop"])
    print("       R/R: %sx  RSI: %s  Vol: %sx" % (s["rr"], s["rsi"], s["vol_rel"]))
    print("       " + " | ".join(s["reasons"]))

def print_table(signals):
    rows = [[
        s["ticker"], s["tipo"], s["dir"],
        "$%s" % s["preco"], "$%s" % s["entrada"],
        "$%s" % s["stop"],  "$%s" % s["target"],
        "%sx" % s["rr"],    "%.1f" % s["score"],
        "%s%%" % s["atr_pct"]
    ] for s in signals]
    headers = ["Ticker","Tipo","Dir","Preco","Entrada","Stop","Target","R/R","Score","ATR%"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trading Scanner v2")
    parser.add_argument("--mode",      choices=["all","day","swing"], default="all")
    parser.add_argument("--top",       type=int, default=15)
    parser.add_argument("--table",     action="store_true")
    parser.add_argument("--universe",  choices=["nasdaq100","smallcap","all"],
                        default="smallcap")
    args = parser.parse_args()

    tickers        = UNIVERSES[args.universe]
    telegram_mode  = bool(os.environ.get("TELEGRAM_TOKEN"))

    print_header(args.universe, len(tickers))
    print("  A analisar %d tickers...\n" % len(tickers))

    signals = []
    for i, ticker in enumerate(tickers, 1):
        print("\r  Progresso: %d/%d - %-8s" % (i, len(tickers), ticker),
              end="", flush=True)
        result = analyse_ticker(ticker, mode=args.mode)
        if result:
            signals.append(result)

    print("\r" + " "*55 + "\r", end="")

    if not signals:
        print("  Nenhum sinal encontrado com os criterios actuais.")
        if telegram_mode:
            send_telegram("*SCANNER - %s*\n\nNenhum sinal encontrado hoje." %
                          datetime.now().strftime("%d/%m/%Y"))
        return

    # Ordenar por score desc, depois R/R desc
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
    n_warn  = sum(1 for s in signals if s["warnings"])

    print("\n" + "="*65)
    print("  SUMARIO: %d sinais | LONG:%d SHORT:%d | DAY:%d SWING:%d | Avisos:%d" % (
        len(signals), n_long, n_short, n_day, n_swing, n_warn))
    print("="*65)
    print("  AVISO: Nao constitui aconselhamento financeiro.\n")

    if telegram_mode:
        msg = format_for_telegram(signals)
        send_telegram(msg)
        print("  Mensagem enviada ao Telegram!\n")

if __name__ == "__main__":
    main()
