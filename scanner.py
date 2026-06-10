#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Trading Scanner v10 - Small/Micro Cap
#
# ALTERACOES v10 (apos primeiro run em producao da v9):
#  A. Universo: fonte primaria passa a ser o screener nativo do Yahoo
#     (yf.screen + EquityQuery) - mesmo backend dos downloads de precos,
#     pelo que funciona no GitHub Actions onde o CSV da iShares foi
#     bloqueado (v9 caiu no fallback de 50 tickers: "50 analisadas").
#     Query: US, mcap $50M-$2B, preco $2-$50, vol3m > 200k, ordenado por
#     liquidez. iShares CSV passa a fonte secundaria.
#  B. Filtros duros novos (na v9 eram apenas penalizacoes e nao chegaram):
#     RSI > 70 rejeitado (CLOV passou com 77.4);
#     extensao > 3 ATR acima da EMA20 rejeitada (CLOV: 4.0);
#     volume do dia < 0.6x media rejeitado (SKLZ passou com 0.31x).
#  C. MAX_ATR_PCT 12 -> 8: na v9 o SKLZ passou com ATR 11.9%, implicando
#     stop de ~18% - risco por trade intransaccionavel em swing.
#
# ALTERACOES v9 (correccao de precisao):
#  1. Universo real: holdings completos via CSV oficial iShares (IWM/IWC),
#     em vez de funds_data.top_holdings (que so devolve ~10 tickers e
#     forcava sempre a lista fallback estatica com empresas deslistadas).
#  2. Verificacao de frescura dos dados: sinais so com barra da ultima
#     sessao (<=4 dias). Elimina sinais sobre tickers mortos/deslistados.
#  3. Barra incompleta: se o script correr antes do fecho (21:05 UTC),
#     a barra do proprio dia e descartada (evita indicadores parciais).
#  4. RSI / ATR / ADX com suavizacao de Wilder (convencao standard -
#     valores agora coincidem com os do broker/TradingView).
#  5. Scoring redesenhado: indicadores colineares deixaram de somar em
#     duplicado; penalizacoes por sobre-extensao, gap-up, preco abaixo
#     da SMA200 e volatilidade extrema. Antes, o score saturava
#     exactamente em acoes esticadas no topo local.
#  6. Liquidez em dolares (mediana 20d de close*volume >= $2M) em vez de
#     75k accoes/dia, que em micro caps e intransaccionavel (spread
#     destruia o R/R teorico).
#  7. Download em lote (chunks de 100) - menos rate-limiting do Yahoo,
#     menos falhas silenciosas; falhas agora sao contadas e reportadas.
#  8. Removida a classificacao "DAY": era inalcancavel (exigia RSI<35
#     mas o score so premiava momentum bullish) e conceptualmente
#     contraditoria. O scanner e de swing/momentum.
#  9. Aviso de earnings proximos (<=7 dias) nos finalistas.
# 10. Bollinger com ddof=0 (convencao TA).

import os
import io
import csv
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

# --------------------------------------------------
# PARAMETROS
# --------------------------------------------------
STOP_ATR_MULT   = 1.5
TARGET_RR       = 2.0
MIN_SCORE       = 5.0
MIN_DOLLAR_VOL  = 2_000_000   # mediana 20d de close*volume (USD)
MIN_ATR_PCT     = 2.0   # v10.1: 0.5 deixava passar stops de centimos (CCO:
                        # stop $0.02 em accao de $2.41, abaixo do spread).
                        # ATR>=2% garante stop 1.5xATR >= 3% - transaccionavel.
MAX_ATR_PCT     = 8.0   # stop 1.5xATR fica <=12% (v9 permitia 18%)
MIN_PRICE       = 2.0
MAX_PRICE       = 50.0
MAX_TICKERS     = 400
MAX_STALE_DAYS  = 4           # barra mais recente nao pode ter mais que isto
CHUNK_SIZE      = 100         # tickers por chamada yf.download

# CSV oficiais de holdings da iShares (lista COMPLETA de constituintes)
ISHARES_CSV = {
    "IWM": ("https://www.ishares.com/us/products/239710/"
            "ishares-russell-2000-etf/1467271812596.ajax"
            "?fileType=csv&fileName=IWM_holdings&dataType=fund"),
    "IWC": ("https://www.ishares.com/us/products/239716/"
            "ishares-microcap-etf/1467271812596.ajax"
            "?fileType=csv&fileName=IWC_holdings&dataType=fund"),
}

# Fallback de ultimo recurso (limpo de tickers deslistados conhecidos).
# Nota: a verificacao de frescura (MAX_STALE_DAYS) protege mesmo que
# algum destes venha a ser deslistado no futuro.
FALLBACK_TICKERS = [
    "HIMS","IONQ","JOBY","RKLB","BLNK","CHPT","EVGO","BYND","CLOV","STEM",
    "GEVO","LAZR","OUST","OPEN","WKHS","ZETA","MGNI","FSLY","LPSN","BIGC",
    "AEHR","GILT","MNKD","NVTS","MVST","AXSM","BEAM","ARCT","AGEN","AVDL",
    "CRGY","ARCH","PRPL","UWMC","ORGN","ZVIA","DKNG","EXPI","ATLC","MFIN",
    "ACNB","SPCE","SKLZ","PESI","AMPY","GRAB","HOOD","RIVN","LCID","SIRI",
]

# --------------------------------------------------
# UNIVERSO DE TICKERS
# --------------------------------------------------

def _parse_ishares_csv(text):
    """Extrai tickers de equity do CSV de holdings da iShares.
    O ficheiro tem ~9 linhas de metadados antes do cabecalho."""
    tickers = []
    reader = csv.reader(io.StringIO(text))
    header_idx = None
    rows = list(reader)
    for i, row in enumerate(rows):
        if row and row[0].strip().lower() == "ticker":
            header_idx = i
            break
    if header_idx is None:
        return []
    header = [h.strip().lower() for h in rows[header_idx]]
    try:
        t_col = header.index("ticker")
    except ValueError:
        return []
    a_col = header.index("asset class") if "asset class" in header else None
    for row in rows[header_idx + 1:]:
        if len(row) <= t_col:
            continue
        t = row[t_col].strip()
        if a_col is not None and len(row) > a_col:
            if "equity" not in row[a_col].strip().lower():
                continue
        if t and t.isalpha() and 1 <= len(t) <= 5:
            tickers.append(t.upper())
    return tickers


def get_universe_yahoo_screener(max_tickers):
    """Fonte primaria: screener nativo do Yahoo via yfinance (mesmo backend
    dos downloads de precos - se um funciona no runner, o outro tambem).
    Small/micro caps US, ordenadas por volume medio 3m (mais liquidas 1o)."""
    try:
        from yfinance import EquityQuery
    except ImportError:
        return []
    if not hasattr(yf, "screen"):
        return []
    q = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NGM", "NCM", "NYQ", "ASE"]),
        EquityQuery("btwn", ["intradaymarketcap", 50_000_000, 2_000_000_000]),
        EquityQuery("btwn", ["intradayprice", MIN_PRICE, MAX_PRICE]),
        EquityQuery("gt", ["avgdailyvol3m", 200_000]),
    ])
    tickers = []
    for offset in range(0, 1000, 250):   # Yahoo: max 250 por chamada
        try:
            resp = yf.screen(q, offset=offset, size=250,
                             sortField="avgdailyvol3m", sortAsc=False)
            quotes = (resp or {}).get("quotes", [])
            if not quotes:
                break
            tickers.extend(qt.get("symbol", "") for qt in quotes)
            if len(tickers) >= max_tickers:
                break
        except Exception as e:
            print("  Screener Yahoo (offset %d): falhou (%s)" % (offset, str(e)[:60]))
            break
    clean = []
    seen = set()
    for t in tickers:
        if t and t.isalpha() and 1 <= len(t) <= 5 and t not in seen:
            seen.add(t)
            clean.append(t)
    return clean[:max_tickers]


def get_universe(max_tickers=MAX_TICKERS):
    # 1) Fonte primaria: screener Yahoo (small/micro cap, ja filtrado
    #    por preco e mcap, ordenado por liquidez)
    ts = get_universe_yahoo_screener(max_tickers)
    if len(ts) >= 100:
        print("  Screener Yahoo: %d tickers small/micro cap" % len(ts))
        return sorted(ts)
    print("  Screener Yahoo devolveu %d tickers - a tentar CSV iShares..." % len(ts))

    all_tickers = set(ts)

    # 2) Fonte secundaria: CSV completo da iShares (pode ser bloqueado
    #    por anti-bot em runners de CI)
    for symbol, url in ISHARES_CSV.items():
        try:
            r = requests.get(url, timeout=30, headers={
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0 Safari/537.36"),
                "Accept": "text/csv,*/*",
            })
            r.raise_for_status()
            ts2 = _parse_ishares_csv(r.text)
            if len(ts2) > 100:
                all_tickers.update(ts2)
                print("  ETF %s: %d constituintes (CSV iShares)" % (symbol, len(ts2)))
            else:
                print("  ETF %s: CSV devolveu apenas %d tickers - ignorado" % (symbol, len(ts2)))
        except Exception as e:
            print("  ETF %s: CSV falhou (%s)" % (symbol, str(e)[:60]))

    # 3) Fonte terciaria: top holdings via yfinance (limitado, mas algo)
    if len(all_tickers) < 100:
        for symbol in ISHARES_CSV:
            try:
                fd = yf.Ticker(symbol).funds_data
                th = getattr(fd, "top_holdings", None)
                if th is not None and not th.empty:
                    all_tickers.update(list(th.index))
                    print("  ETF %s: %d top holdings (yfinance)" % (symbol, len(th)))
            except Exception:
                pass

    # 4) Ultimo recurso
    if len(all_tickers) < 50:
        print("  AVISO: a usar lista fallback estatica (%d tickers). "
              "Universo NAO esta actualizado." % len(FALLBACK_TICKERS))
        return list(dict.fromkeys(FALLBACK_TICKERS))

    clean = sorted(t for t in all_tickers
                   if isinstance(t, str) and t.isalpha() and 1 <= len(t) <= 5)

    if len(clean) > max_tickers:
        import random
        random.seed(datetime.now().toordinal())  # estavel dentro do mesmo dia
        clean = sorted(random.sample(clean, max_tickers))

    print("  Total universo: %d tickers" % len(clean))
    return clean

# --------------------------------------------------
# INDICADORES (suavizacao de Wilder = convencao standard)
# --------------------------------------------------

def wilder_smooth(series, period):
    """Suavizacao de Wilder exacta: seed = SMA dos primeiros `period`
    valores, depois recursao prev*(n-1)/n + v/n. Coincide com os valores
    de broker/TradingView (o ewm do pandas usa seed diferente)."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid_idx = np.where(~np.isnan(vals))[0]
    if len(valid_idx) < period:
        return pd.Series(out, index=series.index)
    seed_pos = valid_idx[period - 1]
    prev = float(np.mean(vals[valid_idx[:period]]))
    out[seed_pos] = prev
    for i in range(seed_pos + 1, len(vals)):
        v = vals[i]
        if np.isnan(v):
            continue
        prev = (prev * (period - 1) + v) / period
        out[i] = prev
    return pd.Series(out, index=series.index)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = wilder_smooth(delta.clip(lower=0), period)
    loss = wilder_smooth(-delta.clip(upper=0), period)
    rs = gain / loss          # loss=0 -> rs=inf -> RSI=100 (correcto)
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    return pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

def compute_atr(high, low, close, period=14):
    return wilder_smooth(true_range(high, low, close), period)

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def compute_bb(series, period=20, std=2.0):
    mid = series.rolling(period).mean()
    sigma = series.rolling(period).std(ddof=0)   # convencao TA
    return mid + std * sigma, mid, mid - std * sigma

def compute_adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                        index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                         index=close.index)
    atr = wilder_smooth(true_range(high, low, close), period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_smooth(dx, period)
    return adx, plus_di, minus_di

# --------------------------------------------------
# DOWNLOAD EM LOTE
# --------------------------------------------------

def download_batches(tickers, chunk_size=CHUNK_SIZE):
    """Devolve dict ticker -> DataFrame OHLCV. Download em chunks para
    reduzir rate-limiting e falhas silenciosas."""
    data = {}
    n_chunks = (len(tickers) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = tickers[ci * chunk_size:(ci + 1) * chunk_size]
        print("  Download lote %d/%d (%d tickers)..." % (ci + 1, n_chunks, len(chunk)))
        try:
            raw = yf.download(chunk, period="1y", interval="1d",
                              auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as e:
            print("  Lote %d falhou: %s" % (ci + 1, str(e)[:60]))
            continue
        if raw is None or raw.empty:
            continue
        if len(chunk) == 1:
            data[chunk[0]] = raw
            continue
        for t in chunk:
            try:
                sub = raw[t].dropna(how="all")
                if not sub.empty:
                    data[t] = sub
            except KeyError:
                pass
    return data

# --------------------------------------------------
# ANALISE POR TICKER
# --------------------------------------------------

def market_closed_utc(now=None):
    """True se a sessao regular dos EUA ja fechou (conservador: 21:05 UTC
    cobre EST; em EDT o fecho e 20:00 UTC)."""
    now = now or datetime.now(timezone.utc)
    return now.hour > 21 or (now.hour == 21 and now.minute >= 5)

def analyse_ticker(ticker, df, today):
    try:
        if df is None or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(df) < 60:
            return None

        # --- Frescura: descartar barra do dia se a sessao ainda decorre
        last_date = df.index[-1].date()
        if last_date == today and not market_closed_utc():
            df = df.iloc[:-1]
            last_date = df.index[-1].date()

        # --- Frescura: rejeitar dados velhos (ticker deslistado/suspenso)
        if (today - last_date).days > MAX_STALE_DAYS:
            return None

        close, high, low = df["Close"], df["High"], df["Low"]
        volume, opn = df["Volume"], df["Open"]

        c = float(close.iloc[-1])
        if c < MIN_PRICE or c > MAX_PRICE:
            return None

        # --- Liquidez em dolares (mediana 20d), nao em accoes
        dollar_vol = float((close * volume).iloc[-20:].median())
        if dollar_vol < MIN_DOLLAR_VOL:
            return None

        rsi = compute_rsi(close)
        macd, macd_sig, mhist = compute_macd(close)
        atr = compute_atr(high, low, close)
        bb_up, bb_mid, bb_low = compute_bb(close)
        adx, plus_di, minus_di = compute_adx(high, low, close)
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        r = float(rsi.iloc[-1])
        m, ms = float(macd.iloc[-1]), float(macd_sig.iloc[-1])
        mh, mh_1 = float(mhist.iloc[-1]), float(mhist.iloc[-2])
        at = float(atr.iloc[-1])
        bbu = float(bb_up.iloc[-1])
        adxv = float(adx.iloc[-1])
        pdi, mdi = float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
        e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
        o = float(opn.iloc[-1])
        h_, l_ = float(high.iloc[-1]), float(low.iloc[-1])
        prev_c = float(close.iloc[-2])

        if not np.isfinite(at) or at <= 0 or not np.isfinite(r):
            return None

        atr_pct = (at / c) * 100
        if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
            return None

        avg_vol20 = float(volume.iloc[-20:].mean())
        vol_ratio = float(volume.iloc[-1]) / avg_vol20 if avg_vol20 > 0 else 0.0

        s200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
        s50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
        golden_cross = (s50 is not None and s200 is not None and s50 > s200)

        e50_slope_up = float(ema50.iloc[-1]) > float(ema50.iloc[-6])
        hi20 = float(high.iloc[-21:-1].max())  # maximo 20d ANTERIOR a hoje
        breakout20 = c > hi20
        gap_pct = (o / prev_c - 1.0) * 100 if prev_c > 0 else 0.0
        candle_range = max(h_ - l_, 1e-9)
        close_pos = (c - l_) / candle_range   # posicao do fecho no range do dia
        extension_atr = (c - e20) / at        # distancia a EMA20 em ATRs

        # ---------------- FILTROS DUROS (rejeicao, nao penalizacao) ------
        # Licao da v9 em producao: CLOV passou com RSI 77 + 4 ATR acima da
        # EMA20 (penalizacoes nao chegaram); SKLZ passou com volume 0.31x.
        if r > 70:
            return None          # perseguir sobrecompra nao e setup de swing
        if extension_atr > 3.0:
            return None          # demasiado esticado; reversao a media provavel
        if vol_ratio < 0.6:
            return None          # dia sem participacao = sem confirmacao

        # ---------------- SCORE (max teorico 10.0) ----------------
        score, reasons, warns = 0.0, [], []

        # Estrutura de tendencia (max 3.0)
        if c > e20 > e50:
            score += 1.5; reasons.append("Preco > EMA20 > EMA50")
        elif c > e20:
            score += 0.75; reasons.append("Preco > EMA20")
        if e50_slope_up:
            score += 0.5; reasons.append("EMA50 ascendente")
        if golden_cross:
            score += 1.0; reasons.append("SMA50 > SMA200")

        # Momentum (max 2.5)
        if m > ms and mh > mh_1:
            score += 1.5; reasons.append("MACD bullish, histograma crescente")
        elif m > ms:
            score += 0.75; reasons.append("MACD acima do sinal")
        if 45 <= r <= 65:
            score += 1.0; reasons.append("RSI %.0f saudavel" % r)
        elif 35 <= r < 45:
            score += 0.5; reasons.append("RSI %.0f recuperacao" % r)

        # Forca da tendencia (max 1.5)
        if adxv > 25 and pdi > mdi:
            score += 1.5; reasons.append("ADX %.0f forte" % adxv)
        elif adxv > 20 and pdi > mdi:
            score += 0.75; reasons.append("ADX %.0f moderado" % adxv)

        # Confirmacao de volume (max 1.5)
        if vol_ratio >= 2.0:
            score += 1.5; reasons.append("Volume %.1fx" % vol_ratio)
        elif vol_ratio >= 1.3:
            score += 0.75; reasons.append("Volume %.1fx" % vol_ratio)

        # Price action (max 1.5)
        if breakout20 and vol_ratio >= 1.3:
            score += 1.0; reasons.append("Breakout max 20d c/ volume")
        if c > o and close_pos >= 0.66:
            score += 0.5; reasons.append("Fecho no terco superior")

        # ---------------- PENALIZACOES ----------------
        if extension_atr > 2.0:
            score -= 1.5; warns.append("Sobre-extendido (%.1f ATR acima EMA20)" % extension_atr)
        if c > bbu:
            score -= 1.0; warns.append("Acima da banda Bollinger superior")
        if gap_pct > 4.0:
            score -= 1.0; warns.append("Gap-up %.1f%% (risco de perseguir)" % gap_pct)
        if s200 is not None and c < s200:
            score -= 1.0; warns.append("Abaixo da SMA200")
        if atr_pct > 8.0:
            score -= 0.5; warns.append("ATR %.1f%% elevado" % atr_pct)

        if score < MIN_SCORE:
            return None

        entry = c
        stop = round(entry - at * STOP_ATR_MULT, 2)
        if stop <= 0:
            return None
        target = round(entry + (entry - stop) * TARGET_RR, 2)
        rr = round((target - entry) / (entry - stop), 2)

        return {
            "ticker": ticker, "preco": round(c, 2), "entrada": round(entry, 2),
            "stop": stop, "target": target, "rr": rr,
            "score": round(score, 1), "vol_rel": round(vol_ratio, 2),
            "rsi": round(r, 1), "atr_pct": round(atr_pct, 2),
            "dollar_vol_m": round(dollar_vol / 1e6, 1),
            "reasons": reasons, "warnings": warns,
        }
    except Exception:
        return None

# --------------------------------------------------
# EARNINGS PROXIMOS (so para finalistas - chamadas lentas)
# --------------------------------------------------

def flag_upcoming_earnings(signals, days=7):
    today = datetime.now(timezone.utc).date()
    for s in signals:
        try:
            cal = yf.Ticker(s["ticker"]).calendar
            dates = []
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    dates = ed if isinstance(ed, (list, tuple)) else [ed]
            for d in dates:
                d = d.date() if hasattr(d, "date") else d
                delta = (d - today).days
                if 0 <= delta <= days:
                    s["warnings"].append("EARNINGS em %d dia(s)" % delta)
                    break
        except Exception:
            pass
    return signals

# --------------------------------------------------
# TELEGRAM
# --------------------------------------------------

def send_telegram(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = "https://api.telegram.org/bot%s/sendMessage" % token
    for i in range(0, len(message), 4000):
        try:
            requests.post(url, data={"chat_id": chat_id,
                                     "text": message[i:i + 4000],
                                     "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print("Erro Telegram: %s" % e)

def format_for_telegram(signals, n_scanned, n_data):
    date_str = datetime.now().strftime("%d/%m/%Y")
    lines = ["*SMALL/MICRO CAP - SINAIS DO DIA (v10.1)*",
             "_%s | %d analisadas (%d com dados validos)_" % (date_str, n_scanned, n_data),
             ""]
    for s in signals:
        warn = " [!]" if s["warnings"] else ""
        lines.append("*%s* | BUY | Score: %.1f%s" % (s["ticker"], s["score"], warn))
        lines.append("  Entrada: `$%s` | Stop: `$%s` | Target: `$%s` | RR: `%sx`" % (
            s["entrada"], s["stop"], s["target"], s["rr"]))
        lines.append("  RSI: `%s` | Vol: `%sx` | ATR: `%s%%` | Liq: `$%sM/dia`" % (
            s["rsi"], s["vol_rel"], s["atr_pct"], s["dollar_vol_m"]))
        if s["warnings"]:
            lines.append("  Avisos: %s" % "; ".join(s["warnings"]))
        lines.append("")
    lines.append("_Apenas referencia. Nao e aconselhamento financeiro._")
    return "\n".join(lines)

# --------------------------------------------------
# OUTPUT TERMINAL
# --------------------------------------------------

def print_header():
    print("\n" + "=" * 70)
    print("  SCANNER v10.1 - Small/Micro Cap | $%.0f-$%.0f | Liq min: $%.0fM/dia" % (
        MIN_PRICE, MAX_PRICE, MIN_DOLLAR_VOL / 1e6))
    print("  %s | Universo: screener Yahoo (mcap $50M-$2B)" % datetime.now().strftime("%d/%m/%Y %H:%M"))
    print("=" * 70 + "\n")

def print_signal(s, idx):
    w = (" | AVISO: " + " | ".join(s["warnings"])) if s["warnings"] else ""
    print("-" * 65)
    print("  #%-3d %-8s | BUY | SWING | Score: %.1f/10%s" % (
        idx, s["ticker"], s["score"], Fore.YELLOW + w + Style.RESET_ALL))
    print("       Preco  : $%s  (ATR: %s%% | Liq: $%sM/dia)" % (
        s["preco"], s["atr_pct"], s["dollar_vol_m"]))
    print("  >>   Target : $%s" % s["target"])
    print("  XX   Stop   : $%s" % s["stop"])
    print("       R/R: %sx  RSI: %s  Vol: %sx" % (s["rr"], s["rsi"], s["vol_rel"]))
    print("       " + " | ".join(s["reasons"]))

def print_table(signals):
    rows = [[s["ticker"], "$%s" % s["preco"], "$%s" % s["stop"],
             "$%s" % s["target"], "%sx" % s["rr"], "%.1f" % s["score"],
             "%s%%" % s["atr_pct"], "$%sM" % s["dollar_vol_m"]] for s in signals]
    print(tabulate(rows, headers=["Ticker", "Preco", "Stop", "Target",
                                  "R/R", "Score", "ATR%", "Liq/dia"],
                   tablefmt="rounded_outline"))

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trading Scanner v10")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--max-tickers", type=int, default=MAX_TICKERS)
    args = parser.parse_args()

    telegram_mode = bool(os.environ.get("TELEGRAM_TOKEN"))
    today = datetime.now(timezone.utc).date()

    print_header()
    print("  A obter universo de tickers...\n")
    tickers = get_universe(args.max_tickers)

    print("\n  A descarregar dados de %d tickers...\n" % len(tickers))
    data = download_batches(tickers)
    n_data = len(data)
    print("  Dados validos: %d/%d tickers\n" % (n_data, len(tickers)))
    if n_data < len(tickers) * 0.5:
        print("  AVISO: >50%% dos downloads falharam - possivel rate-limit. "
              "Resultados de hoje podem estar incompletos.\n")

    signals = []
    for i, (ticker, df) in enumerate(sorted(data.items()), 1):
        print("\r  Analise: %d/%d - %-8s | Sinais: %d" % (
            i, n_data, ticker, len(signals)), end="", flush=True)
        result = analyse_ticker(ticker, df, today)
        if result:
            signals.append(result)
    print("\r" + " " * 65 + "\r", end="")

    if not signals:
        print("  Nenhum sinal encontrado hoje.")
        if telegram_mode:
            send_telegram("*SMALL CAP SCANNER v10.1 - %s*\n\n_%d analisadas (%d com dados)._\n"
                          "Nenhum sinal encontrado hoje." % (
                              datetime.now().strftime("%d/%m/%Y"), len(tickers), n_data))
        return

    signals.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    top_signals = signals[:args.top]
    top_signals = flag_upcoming_earnings(top_signals)

    if args.table:
        print_table(top_signals)
    else:
        for idx, s in enumerate(top_signals, 1):
            print_signal(s, idx)

    n_warn = sum(1 for s in top_signals if s["warnings"])
    print("\n" + "=" * 65)
    print("  SUMARIO: %d sinais BUY (de %d com dados validos) | Avisos: %d" % (
        len(signals), n_data, n_warn))
    print("=" * 65)
    print("  Nao constitui aconselhamento financeiro.\n")

    if telegram_mode:
        send_telegram(format_for_telegram(top_signals, len(tickers), n_data))
        print("  Mensagem enviada ao Telegram!\n")

if __name__ == "__main__":
    main()
