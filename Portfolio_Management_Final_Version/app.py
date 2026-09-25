"""
app.py  —  Portfolio Manager · Application GUI unifiée
Lance ce fichier : python app.py

Contient :
  • Wizard de configuration (3 étapes)
  • Dashboard avec onglets (VaR, SMA, EMA, RSI, Distribution, Frontière efficiente)
  • Tout embarqué dans une seule fenêtre Tkinter — aucune fenêtre matplotlib externe
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import minimize
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE & FONTS
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":      "#0A0E17", "panel":   "#111827", "card":    "#1A2332",
    "border":  "#1E3A5F", "accent":  "#0EA5E9", "accent2": "#10B981",
    "accent3": "#F59E0B", "danger":  "#EF4444", "text":    "#F1F5F9",
    "sub":     "#64748B", "muted":   "#1E293B", "pos":     "#34D399",
    "grid":    "#1C2535",
}
MP = {  # matplotlib palette
    "bg": "#0A0E17", "panel": "#111827", "border": "#1E3A5F",
    "text": "#F1F5F9", "sub": "#64748B", "accent": "#0EA5E9",
    "accent2": "#10B981", "accent3": "#F59E0B", "danger": "#EF4444",
    "pos": "#34D399", "grid": "#1C2535", "muted": "#1E293B",
    "sma1": "#38BDF8", "sma2": "#FFA657", "ema1": "#C084FC", "ema2": "#FB7185",
}
FL = ("Courier New", 9,  "bold")   # font label
FB = ("Courier New", 9)            # font body
FM = ("Courier New", 10)           # font mono
FT = ("Courier New", 18, "bold")   # font title
FS = ("Courier New", 7)            # font small


def mpl_style():
    plt.rcParams.update({
        "figure.facecolor": MP["bg"], "axes.facecolor": MP["panel"],
        "axes.edgecolor": MP["border"], "axes.labelcolor": MP["sub"],
        "axes.titlecolor": MP["text"], "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": MP["grid"], "grid.linewidth": 0.5,
        "xtick.color": MP["sub"], "ytick.color": MP["sub"],
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.facecolor": MP["panel"], "legend.edgecolor": MP["border"],
        "legend.labelcolor": MP["text"], "legend.fontsize": 7,
        "lines.linewidth": 1.4, "text.color": MP["text"],
        "font.family": "monospace",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS (standalone — pas besoin des fichiers externes)
# ══════════════════════════════════════════════════════════════════════════════
def calc_var(series, confidence):
    returns = series.pct_change().dropna()
    return float(np.percentile(returns, (1 - confidence) * 100))

def calc_sma(series, period):
    return series.rolling(window=int(period), min_periods=1).mean()

def calc_ema(series, period):
    return series.ewm(span=int(period), adjust=False).mean()

def calc_rsi(series, period=14):
    values = list(np.zeros(period))
    for i in range(period, len(series)):
        gain = loss = 0
        for d in range(i - period, i + 1):
            diff = series.iloc[d] - series.iloc[d - 1]
            if diff > 0: gain += diff
            else:        loss -= diff
        mg, ml = gain / period, loss / period
        rs = mg / ml if ml != 0 else 0
        values.append(100 - 100 / (1 + rs))
    return values

# ── CAC 40 data (Yahoo Finance via yfinance, no API key) ─────────────────────
# Index composition changes over time: tickers without data are skipped.
CAC40_TICKERS = [
    "AC.PA", "AI.PA", "AIR.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA", "BVI.PA",
    "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EDEN.PA", "ENGI.PA", "EL.PA",
    "ERF.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA", "ORA.PA",
    "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA",
    "STLAP.PA", "STMPA.PA", "TEP.PA", "HO.PA", "TTE.PA", "URW.PA", "VIE.PA", "DG.PA",
]

def download_cac40(path, period="1y"):
    """
    Download daily OHLCV of the CAC 40 constituents and save it as a CSV in the
    format the app reads (Date, <TICKER>_Open/High/Low/Close/Volume).
    Returns (tickers kept, tickers skipped).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("CAC 40 download needs yfinance:  pip install yfinance")

    raw = yf.download(CAC40_TICKERS, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (check your connection).")

    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols, kept, skipped = {}, [], []
    for t in CAC40_TICKERS:
        if t not in raw.columns.get_level_values(0) or raw[t]["Close"].notna().sum() < 30:
            skipped.append(t)
            continue
        kept.append(t)
        for f in fields:
            cols[f"{t}_{f}"] = raw[t][f]
    if len(kept) < 2:
        raise RuntimeError("Not enough CAC 40 tickers returned data.")

    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.ffill().dropna()     # align trading days, drop the leading gaps
    df.to_csv(path, date_format="%Y-%m-%d")
    return kept, skipped

def portfolio_perf(w, mu, cov, td=252):
    r = np.dot(w, mu) * td
    v = np.sqrt(w @ cov @ w) * np.sqrt(td)
    s = r / v if v != 0 else 0.0
    return r, v, s

def min_var_target(target, mu, cov, n):
    def vol(w): return np.sqrt(w @ cov @ w)
    res = minimize(vol, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1},
                                 {"type":"eq","fun":lambda w: w@mu - target}],
                   options={"maxiter":500,"ftol":1e-9})
    return res if res.success else None

def max_sharpe(mu, cov, n, rf=0):
    def neg(w):
        r,v,_ = portfolio_perf(w,mu,cov)
        return -(r-rf)/v if v!=0 else 0
    res = minimize(neg, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"maxiter":500,"ftol":1e-9})
    return res.x if res.success else np.ones(n)/n

def min_variance(mu, cov, n):
    def vol(w): return np.sqrt(w @ cov @ w)
    res = minimize(vol, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                   options={"maxiter":500,"ftol":1e-9})
    return res.x if res.success else np.ones(n)/n

# ── UCITS 5/10/40 rule ────────────────────────────────────────────────────────
#   • no single asset may exceed 10% of the fund
#   • assets weighing more than 5% may not exceed 40% of the fund in aggregate
UCITS_MAX_SINGLE = 0.10
UCITS_THRESHOLD  = 0.05
UCITS_MAX_SUM    = 0.40
UCITS_MIN_ASSETS = 16   # 4 × 10% + 12 × 5% — smallest portfolio that can comply

def ucits_check(weights, tol=1e-6):
    """Return (compliant, max_weight, sum_of_weights_above_5%)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    max_w   = float(w.max()) if len(w) else 0.0
    big_sum = float(w[w > UCITS_THRESHOLD + tol].sum())
    ok = max_w <= UCITS_MAX_SINGLE + tol and big_sum <= UCITS_MAX_SUM + tol
    return ok, max_w, big_sum

def ucits_whole_shares(prices, amounts, budget, tol=1e-6, rounds=200):
    """
    Turn UCITS target amounts into whole-share quantities that still comply.

    Plain rounding down leaves cash aside, so the invested total is below the
    budget and every weight grows: assets optimised at exactly 5% / 10% end up
    just above the limit. Here the caps are measured against an estimate V of
    the invested total: allocate shares under those caps, then set V to what
    was actually invested and repeat until the two match.
    """
    p = np.asarray(prices, dtype=float)
    target = np.asarray(amounts, dtype=float)
    safe_p = np.where(p > 0, p, np.inf)
    big = target / budget > UCITS_THRESHOLD + tol     # assets allowed up to 10%
    cap = np.where(big, UCITS_MAX_SINGLE, UCITS_THRESHOLD)

    V = budget
    for _ in range(rounds):
        # Bulk allocation: each asset up to min(target, cap·V), big group ≤ 40%·V
        want = np.minimum(target, cap * V)
        if want[big].sum() > UCITS_MAX_SUM * V:
            want[big] *= UCITS_MAX_SUM * V / want[big].sum()
        q = np.floor(want / safe_p).astype(int)

        # Top-up: spend leftover cash on the most under-filled assets
        while True:
            v = q * p
            new_v = v + p
            ok = ((p > 0) & (p <= budget - v.sum())
                  & (new_v <= cap * V + tol) & (new_v <= target + p))
            if big.any():
                ok &= v[big].sum() + p * big <= UCITS_MAX_SUM * V + tol
            if not ok.any():
                break
            q[int(np.argmax(np.where(ok, target - v, -np.inf)))] += 1

        invested = float((q * p).sum())
        if invested <= 0 or ucits_check(q * p, tol)[0]:
            break
        V = invested    # caps were too loose for what we actually invested
    return q.tolist()

def _max_sharpe_bounded(mu, cov, bounds, extra_cons=(), restarts=10):
    """Max-Sharpe SLSQP with custom bounds/constraints. Returns None on failure."""
    n = len(mu)
    def neg_sharpe(w):
        r, v, _ = portfolio_perf(w, mu, cov)
        return -(r / v) if v != 0 else 0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_cons]
    ub   = np.array([b[1] for b in bounds], dtype=float)
    best_x, best_s = None, -np.inf
    for i in range(restarts):
        w0 = ub / ub.sum() if i == 0 else np.minimum(np.random.dirichlet(np.ones(n)), ub)
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
        if res.success and (-res.fun) > best_s:
            best_x, best_s = res.x.copy(), -res.fun
    return best_x

def ucits_optimize(mu, cov):
    """
    Max-Sharpe portfolio respecting the UCITS 5/10/40 rule.
    The 40% part is not smooth, so we pick a set of k 'large' assets allowed up
    to 10% (their sum capped at 40%) while every other asset is capped at 5%,
    and keep the best Sharpe over k.
    """
    n = len(mu)
    if n < UCITS_MIN_ASSETS:
        raise ValueError(
            f"UCITS 5/10/40 rule needs at least {UCITS_MIN_ASSETS} assets "
            f"(4 × 10% + 12 × 5%), but the CSV only has {n}.")

    # Rank assets using a plain 10% cap, to choose which ones may exceed 5%
    w_cap = _max_sharpe_bounded(mu, cov, [(0, UCITS_MAX_SINGLE)] * n, restarts=5)
    if w_cap is None:
        w_cap = mu / np.sqrt(np.diag(cov))
    order = np.argsort(w_cap)[::-1]

    best_w, best_s = None, -np.inf
    for k in range(0, min(8, n) + 1):
        cap_big = min(UCITS_MAX_SUM, UCITS_MAX_SINGLE * k)
        if cap_big + UCITS_THRESHOLD * (n - k) < 1 - 1e-9:
            continue   # not enough room to reach 100%
        big = np.zeros(n, dtype=bool); big[order[:k]] = True
        bounds = [(0, UCITS_MAX_SINGLE if b else UCITS_THRESHOLD) for b in big]
        cons = [{"type": "ineq", "fun": lambda w, m=big: UCITS_MAX_SUM - w[m].sum()}]
        w = _max_sharpe_bounded(mu, cov, bounds, cons, restarts=5)
        if w is None or not ucits_check(w)[0]:
            continue
        s = portfolio_perf(w, mu, cov)[2]
        if s > best_s:
            best_w, best_s = w, s

    if best_w is None:
        raise ValueError("UCITS optimisation could not find a compliant portfolio.")
    return best_w

def markowitz_optimize(df_close, total, min_assets=1, ucits=False):
    n   = df_close.shape[1]
    ret = df_close.pct_change().dropna()
    mu  = ret.mean().values
    cov = ret.cov().values
    min_assets = max(1, min(int(min_assets), n))

    def neg_sharpe(w):
        r, v, _ = portfolio_perf(w, mu, cov)
        return -(r / v) if v != 0 else 0

    best_x, best_s = None, -np.inf
    if ucits:
        best_x = ucits_optimize(mu, cov)
    else:
        # Multiple restarts to escape local minima
        for _ in range(30):
            w0 = np.random.dirichlet(np.ones(n))
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=[(0, 1)] * n,
                           constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                           options={"maxiter": 1000, "ftol": 1e-10})
            if res.success and (-res.fun) > best_s:
                best_x, best_s = res.x.copy(), -res.fun

    ws = best_x if best_x is not None else np.ones(n) / n

    # Enforce minimum number of assets: if fewer than min_assets have weight>1%,
    # redistribute weight from the top asset to raise the smallest until threshold met
    MIN_W = 0.01
    active = np.sum(ws >= MIN_W)
    if active < min_assets:
        idx_sorted = np.argsort(ws)  # ascending
        to_promote = min_assets - active
        for i in range(to_promote):
            idx = idx_sorted[i]
            deficit = MIN_W - ws[idx]
            ws[idx] = MIN_W
            # Take from the largest asset
            donor = idx_sorted[-(i+1)]
            ws[donor] = max(0, ws[donor] - deficit)
        ws = ws / ws.sum()  # renormalise

    return (ws * total).tolist()

def dashboard_metrics(mu, cov, weights, n_days, conf, port_rets, td=252):
    """Dashboard KPIs — shared by the Dashboard panel and the Excel export."""
    port_mu_d  = float(np.dot(weights, mu))               # daily drift
    port_sig_d = float(np.sqrt(weights @ cov @ weights))  # daily vol

    recent    = port_rets.iloc[-21:] if len(port_rets) >= 21 else port_rets
    cur_vol   = float(recent.std() * np.sqrt(td))         # realised, last 21 days
    est_vol   = port_sig_d * np.sqrt(td)
    est_perf  = float(np.exp(port_mu_d * n_days) - 1)     # log-normal mean
    var_daily = float(np.percentile(port_rets.dropna(), (1 - conf) * 100))
    var_horiz = var_daily * np.sqrt(n_days)               # square-root scaling
    sharpe    = (port_mu_d * td) / (port_sig_d * np.sqrt(td)) if port_sig_d > 0 else 0.0

    # Risk indicator (0-10 composite): vol, VaR severity, inverse Sharpe
    vol_score  = min(10, (est_vol / 0.40) * 10)           # 40% vol → score 10
    var_score  = min(10, (abs(var_horiz) / 0.30) * 10)    # 30% horizon VaR → 10
    shrp_score = max(0, 10 - abs(sharpe) * 2)
    risk = round(vol_score * 0.45 + var_score * 0.35 + shrp_score * 0.20, 1)
    risk = min(10, max(0, risk))
    label = "LOW" if risk <= 3 else "MEDIUM" if risk <= 6 else "HIGH"

    return {"est_perf": est_perf, "est_vol": est_vol, "cur_vol": cur_vol,
            "var_horizon": var_horiz, "sharpe": sharpe,
            "risk_score": risk, "risk_label": label}


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════
XL_PCT, XL_EUR, XL_NUM = "0.00%", '#,##0.00 "€"', "#,##0.00"
XL_INT, XL_DATE, XL_RATIO = "#,##0", "yyyy-mm-dd", "0.00"

def _xl_clean(v):
    """Convert numpy / pandas values into something openpyxl can write."""
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v

def _xl_block(ws, row, headers, rows, fmts=None, title=None, bold_last=False):
    """Write a styled table at `row`; returns the next free row (with a gap)."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1E3A5F")
    band_fill = PatternFill("solid", fgColor="F1F5F9")
    thin      = Side(style="thin", color="CBD5E1")

    if title:
        ws.cell(row, 1, title).font = Font(bold=True, size=12, color="0EA5E9")
        row += 1
    for j, h in enumerate(headers, 1):
        c = ws.cell(row, j, h)
        c.font, c.fill = head_font, head_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, r in enumerate(rows):
        row += 1
        for j, v in enumerate(r, 1):
            c = ws.cell(row, j, _xl_clean(v))
            if fmts and fmts[j - 1]:
                c.number_format = fmts[j - 1]
            if i % 2:
                c.fill = band_fill
            if bold_last and i == len(rows) - 1:
                c.font = Font(bold=True)
                c.border = Border(top=thin)
    return row + 2

def _xl_kv(ws, row, title, items):
    """Write a 2-column 'Metric | Value' section with a per-row number format."""
    from openpyxl.styles import Font
    ws.cell(row, 1, title).font = Font(bold=True, size=12, color="0EA5E9")
    row += 1
    for label, value, fmt in items:
        ws.cell(row, 1, label).font = Font(color="475569")
        c = ws.cell(row, 2, _xl_clean(value))
        c.font = Font(bold=True)
        if fmt:
            c.number_format = fmt
        row += 1
    return row + 1

def _xl_autofit(ws, min_w=10, max_w=48):
    for col in ws.columns:
        width = min_w
        for c in list(col)[:300]:
            if c.value is None:
                continue
            n = 12 if isinstance(c.value, (int, float)) else len(str(c.value))
            width = max(width, min(max_w, n + 2))
        ws.column_dimensions[col[0].column_letter].width = width

def export_to_excel(path, r, td=252):
    """Write the portfolio weights and every indicator to an .xlsx, one tab per topic."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.formatting.rule import ColorScaleRule
    except ImportError:
        raise RuntimeError("Excel export needs openpyxl:  pip install openpyxl")

    names, df_close, port = r["names"], r["df_close"], r["portfolio"]
    conf = r["conf"]
    conf_lbl = f"{conf:.0%}"

    rets = df_close.pct_change().dropna()
    mu, cov = rets.mean().values, rets.cov().values
    qty = np.asarray(r["quantities"], dtype=float)
    p0, p1 = df_close.iloc[0].values, df_close.iloc[-1].values
    invested, current = qty * p0, qty * p1
    w_target = np.asarray(r["amounts"], dtype=float)
    w_target = w_target / w_target.sum()
    w_actual = invested / invested.sum() if invested.sum() > 0 else w_target

    port_rets = port.pct_change().dropna()
    drawdown  = port / port.cummax() - 1
    var_d     = float(np.percentile(port_rets, (1 - conf) * 100))
    cvar_d    = float(port_rets[port_rets <= var_d].mean())
    ann_ret   = float(port_rets.mean() * td)
    ann_vol   = float(port_rets.std() * np.sqrt(td))

    # Indicator time series
    ind = pd.DataFrame({"Portfolio Value (€)": port,
                        "Daily Return": port.pct_change(),
                        "Cumulative Return": port / port.iloc[0] - 1,
                        "Drawdown": drawdown})
    for p in r["sma_periods"]:
        ind[f"SMA {p}"] = calc_sma(port, p)
    ind[f"EMA {r['ema_period']}"] = calc_ema(port, r["ema_period"])
    rsi = pd.Series(calc_rsi(port), index=port.index, dtype=float)
    rsi.iloc[:14] = np.nan                       # warm-up period, not a real value
    ind["RSI 14"] = rsi
    ind["Rolling Vol 21d (ann.)"] = port.pct_change().rolling(21).std() * np.sqrt(td)

    wb = Workbook()

    # ── Summary ───────────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    ws.cell(1, 1, "Portfolio Report").font = Font(bold=True, size=16)
    row = 3
    row = _xl_kv(ws, row, "PARAMETERS", [
        ("Source file",           os.path.basename(r["source"]),        None),
        ("Export date",           pd.Timestamp.now().to_pydatetime(),   "yyyy-mm-dd hh:mm"),
        ("First date",            port.index[0],                        XL_DATE),
        ("Last date",             port.index[-1],                       XL_DATE),
        ("Trading days",          len(port),                            XL_INT),
        ("Number of assets",      len(names),                           XL_INT),
        ("Allocation method",     r["method"],                          None),
        ("Budget",                r["total"],                           XL_EUR),
        ("Invested (whole shares)", invested.sum(),                     XL_EUR),
        ("Cash left",             r["total"] - invested.sum(),          XL_EUR),
        ("VaR confidence",        conf,                                 "0%"),
    ])
    row = _xl_kv(ws, row, "HISTORICAL PERFORMANCE", [
        ("Start value",             port.iloc[0],                XL_EUR),
        ("End value",               port.iloc[-1],               XL_EUR),
        ("Total return",            port.iloc[-1] / port.iloc[0] - 1, XL_PCT),
        ("Annualised return",       ann_ret,                     XL_PCT),
        ("Annualised volatility",   ann_vol,                     XL_PCT),
        ("Sharpe ratio (rf = 0)",   ann_ret / ann_vol if ann_vol else 0, XL_RATIO),
        ("Max drawdown",            drawdown.min(),              XL_PCT),
        (f"VaR {conf_lbl} (1 day)", var_d,                       XL_PCT),
        (f"VaR {conf_lbl} (1 day, €)", var_d * port.iloc[-1],    XL_EUR),
        (f"CVaR {conf_lbl} (1 day)", cvar_d,                     XL_PCT),
        ("Best day",                port_rets.max(),             XL_PCT),
        ("Worst day",               port_rets.min(),             XL_PCT),
        ("% positive days",         (port_rets > 0).mean(),      XL_PCT),
    ])
    last = ind.iloc[-1]
    row = _xl_kv(ws, row, "LATEST INDICATORS", [
        (col, last[col], XL_EUR if col.startswith(("SMA", "EMA", "Portfolio")) else
                         XL_RATIO if col == "RSI 14" else XL_PCT)
        for col in ind.columns if col != "Daily Return"])

    n_days = r["dash_days"]
    m = dashboard_metrics(mu, cov, w_target, n_days, conf, port_rets, td)
    row = _xl_kv(ws, row, f"DASHBOARD  (horizon {n_days} trading days)", [
        ("Est. performance",        m["est_perf"],               XL_PCT),
        ("Est. performance (€)",    m["est_perf"] * r["total"],  XL_EUR),
        ("Est. volatility (ann.)",  m["est_vol"],                XL_PCT),
        ("Current volatility (21d)", m["cur_vol"],               XL_PCT),
        (f"VaR {conf_lbl} (horizon)", abs(m["var_horizon"]),     XL_PCT),
        ("Sharpe ratio",            m["sharpe"],                 XL_RATIO),
        ("Risk score (0-10)",       m["risk_score"],             "0.0"),
        ("Risk level",              m["risk_label"],             None),
    ])
    if r.get("ucits"):
        u = r["ucits"]
        _xl_kv(ws, row, "UCITS 5/10/40", [
            ("Compliant",               "YES" if u["ok"] else "NO", None),
            ("Largest position",        u["max_w"],                 XL_PCT),
            ("Sum of positions > 5%",   u["big_sum"],               XL_PCT),
        ])
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 26

    # ── Weights ───────────────────────────────────────────────────────────────
    ws = wb.create_sheet("Weights")
    order = np.argsort(w_actual)[::-1]
    rows = [(names[i], w_target[i], w_target[i] * r["total"], p0[i], int(qty[i]),
             invested[i], w_actual[i], p1[i], current[i],
             current[i] / current.sum() if current.sum() else 0,
             current[i] - invested[i],
             current[i] / invested[i] - 1 if invested[i] else None)
            for i in order]
    rows.append(("TOTAL", w_target.sum(), r["total"], None, int(qty.sum()),
                 invested.sum(), w_actual.sum(), None, current.sum(), 1.0,
                 current.sum() - invested.sum(),
                 current.sum() / invested.sum() - 1 if invested.sum() else None))
    _xl_block(ws, 1,
              ["Asset", "Target Weight", "Target Amount (€)", "Start Price", "Quantity",
               "Invested (€)", "Actual Weight", "Last Price", "Current Value (€)",
               "Current Weight", "P&L (€)", "P&L (%)"],
              rows,
              [None, XL_PCT, XL_EUR, XL_NUM, XL_INT, XL_EUR, XL_PCT, XL_NUM,
               XL_EUR, XL_PCT, XL_EUR, XL_PCT], bold_last=True)
    ws.freeze_panes = "B2"
    _xl_autofit(ws)

    # ── Indicators (daily time series) ────────────────────────────────────────
    ws = wb.create_sheet("Indicators")
    fmts = [XL_DATE] + [XL_EUR if c.startswith(("SMA", "EMA", "Portfolio")) else
                        XL_RATIO if c == "RSI 14" else XL_PCT for c in ind.columns]
    _xl_block(ws, 1, ["Date", *ind.columns],
              [(d, *vals) for d, vals in zip(ind.index, ind.itertuples(index=False))], fmts)
    ws.freeze_panes = "B2"
    _xl_autofit(ws)

    # ── Asset statistics ──────────────────────────────────────────────────────
    ws = wb.create_sheet("Asset Stats")
    sig_p = float(np.sqrt(w_actual @ cov @ w_actual))
    risk_contrib = w_actual * (cov @ w_actual) / sig_p**2 if sig_p > 0 else np.zeros(len(names))
    asset_dd = (df_close / df_close.cummax() - 1).min().values
    rows = []
    for i, n in enumerate(names):
        a = rets.iloc[:, i]
        a_var = float(np.percentile(a, (1 - conf) * 100))
        a_ret, a_vol = mu[i] * td, a.std() * np.sqrt(td)
        rows.append((n, w_actual[i], p1[i] / p0[i] - 1, a_ret, a_vol,
                     a_ret / a_vol if a_vol else 0, a_var, a[a <= a_var].mean(),
                     asset_dd[i], risk_contrib[i]))
    _xl_block(ws, 1,
              ["Asset", "Weight", "Total Return", "Ann. Return", "Ann. Volatility",
               "Sharpe", f"VaR {conf_lbl} (1d)", f"CVaR {conf_lbl} (1d)",
               "Max Drawdown", "Risk Contribution"],
              rows, [None, XL_PCT, XL_PCT, XL_PCT, XL_PCT, XL_RATIO,
                     XL_PCT, XL_PCT, XL_PCT, XL_PCT])
    ws.freeze_panes = "B2"
    _xl_autofit(ws)

    # ── Correlation matrix ────────────────────────────────────────────────────
    ws = wb.create_sheet("Correlation")
    corr = rets.corr().values
    _xl_block(ws, 1, ["", *names],
              [(names[i], *corr[i]) for i in range(len(names))],
              [None] + [XL_RATIO] * len(names))
    last_cell = ws.cell(len(names) + 1, len(names) + 1).coordinate
    ws.conditional_formatting.add(f"B2:{last_cell}", ColorScaleRule(
        start_type="num", start_value=-1, start_color="EF4444",
        mid_type="num", mid_value=0, mid_color="FFFFFF",
        end_type="num", end_value=1, end_color="0EA5E9"))
    ws.freeze_panes = "B2"
    _xl_autofit(ws, min_w=9)

    # ── Efficient frontier ────────────────────────────────────────────────────
    ws = wb.create_sheet("Efficient Frontier")
    w_ms, w_mv = max_sharpe(mu, cov, len(mu)), min_variance(mu, cov, len(mu))
    ports = [("Your Portfolio", w_actual), ("Max Sharpe", w_ms), ("Min Variance", w_mv)]
    row = _xl_block(ws, 1, ["Portfolio", "Ann. Return", "Ann. Volatility", "Sharpe"],
                    [(lbl, *portfolio_perf(w, mu, cov, td)) for lbl, w in ports],
                    [None, XL_PCT, XL_PCT, XL_RATIO], title="KEY PORTFOLIOS")
    row = _xl_block(ws, row, ["Asset", *[lbl for lbl, _ in ports]],
                    [(names[i], *[w[i] for _, w in ports]) for i in order],
                    [None, XL_PCT, XL_PCT, XL_PCT], title="WEIGHTS COMPARISON")
    if r.get("frontier") is not None:
        fr_v, fr_r = r["frontier"]
        _xl_block(ws, row, ["Ann. Volatility", "Ann. Return", "Sharpe"],
                  [(v, ret, ret / v if v else 0) for v, ret in zip(fr_v, fr_r)],
                  [XL_PCT, XL_PCT, XL_RATIO], title="FRONTIER CURVE")
    _xl_autofit(ws)

    # ── GBM Monte Carlo ───────────────────────────────────────────────────────
    if r.get("gbm") is not None:
        g  = r["gbm"]
        ws = wb.create_sheet("GBM Projection")
        pct = g["percentiles"]
        row = _xl_kv(ws, 1, f"FINAL VALUE AFTER {len(pct) - 1} DAYS  "
                            f"({g['n_sims']} simulations)",
                     [(c, pct[c].iloc[-1], XL_EUR) for c in pct.columns])
        if g.get("shocks"):
            row = _xl_block(ws, row,
                            ["Scenario", "Description", "Trough (median)", "Trough Day",
                             "Final Median (€)", "Final 5th pct (€)"],
                            g["shocks"], [None, None, XL_PCT, XL_INT, XL_EUR, XL_EUR],
                            title="SHOCK SCENARIOS")
        _xl_block(ws, row, ["Day", *pct.columns],
                  [(d, *vals) for d, vals in zip(pct.index, pct.itertuples(index=False))],
                  [XL_INT] + [XL_EUR] * len(pct.columns), title="PERCENTILE PATHS")
        _xl_autofit(ws)
        ws.column_dimensions["A"].width = 26

    wb.save(path)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _entry(parent, var, width=24, ph=""):
    f = tk.Frame(parent, bg=C["card"], highlightbackground=C["border"], highlightthickness=1)
    e = tk.Entry(f, textvariable=var, width=width, bg=C["card"], fg=C["text"],
                 insertbackground=C["accent"], relief="flat", font=FM, bd=0)
    e.pack(padx=8, pady=5)
    if ph:
        def fi(_):
            if e.get()==ph: e.delete(0,"end"); e.config(fg=C["text"])
        def fo(_):
            if not e.get(): e.insert(0,ph); e.config(fg=C["sub"])
        e.insert(0,ph); e.config(fg=C["sub"])
        e.bind("<FocusIn>",fi); e.bind("<FocusOut>",fo)
    f.bind("<Enter>", lambda _: f.config(highlightbackground=C["accent"]))
    f.bind("<Leave>", lambda _: f.config(highlightbackground=C["border"]))
    e.bind("<Enter>", lambda _: f.config(highlightbackground=C["accent"]))
    e.bind("<Leave>", lambda _: f.config(highlightbackground=C["border"]))
    return f, e

def _lbl(parent, text, fg=None, font=None, **kw):
    return tk.Label(parent, text=text, fg=fg or C["text"],
                    bg=kw.pop("bg", C["panel"]), font=font or FB, **kw)

def _sep(parent):
    tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=8)

def _sec(parent, text):
    _lbl(parent, text, fg=C["accent"], font=FL).pack(anchor="w", pady=(8,2))

class Toggle(tk.Frame):
    def __init__(self, parent, label, var, on_change=None, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self._v  = var
        self._cb = on_change
        self._lbl_text = label
        self._btn = tk.Label(self, font=FL, cursor="hand2", padx=12, pady=6)
        self._btn.pack()
        self._btn.bind("<Button-1>", self._toggle)
        self._draw()

    def _toggle(self, _=None):
        self._v.set(not self._v.get())
        self._draw()
        if self._cb: self._cb()

    def _draw(self):
        if self._v.get():
            self._btn.config(text=f"✓  {self._lbl_text}", fg=C["bg"],
                              bg=C["accent2"])
            self.config(highlightbackground=C["accent2"], highlightthickness=1)
        else:
            self._btn.config(text=f"○  {self._lbl_text}", fg=C["sub"],
                              bg=C["muted"])
            self.config(highlightbackground=C["muted"], highlightthickness=1)


def _mpl_canvas(parent, fig):
    """Embed a matplotlib figure in a tk frame, with toolbar."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    toolbar.config(bg=C["panel"])
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


def _tag(ax, label):
    ax.text(0.01, 0.98, f" {label} ", transform=ax.transAxes,
            fontsize=7, fontweight="bold", color=MP["bg"], va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=MP["accent"],
                      edgecolor="none", alpha=0.9))

def _date_axis(ax):
    loc = mdates.AutoDateLocator()
    fmt = mdates.AutoDateFormatter(loc)
    fmt.scaled[1.0] = "%d %b"
    fmt.scaled[30.] = "%b '%y"
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Portfolio Manager")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1100x720")
        self._center()

        # ── Config vars ───────────────────────────────────────────────────────
        self.v_path      = tk.StringVar()
        self.v_amount    = tk.StringVar(value="10000")
        self.v_uniform   = tk.BooleanVar(value=False)
        self.v_optimize  = tk.BooleanVar(value=True)
        self.v_custom    = tk.StringVar()
        self.v_conf      = tk.StringVar(value="0.95")
        self.v_var       = tk.BooleanVar(value=True)
        self.v_sma       = tk.BooleanVar(value=True)
        self.v_ema       = tk.BooleanVar(value=True)
        self.v_rsi       = tk.BooleanVar(value=True)
        self.v_distrib   = tk.BooleanVar(value=True)
        self.v_frontier  = tk.BooleanVar(value=True)
        self.v_sma_p      = tk.StringVar(value="12, 26")
        self.v_ema_p      = tk.StringVar(value="12")
        self.v_min_assets = tk.StringVar(value="5")
        self.v_gbm_mc     = tk.BooleanVar(value=True)
        self.v_gbm_sims   = tk.StringVar(value="200")
        self.v_gbm_days   = tk.StringVar(value="252")
        self.v_shocks     = tk.BooleanVar(value=True)
        self.v_dashboard  = tk.BooleanVar(value=True)
        self.v_dash_range = tk.StringVar(value="63")   # trading days (≈3 months)
        self.v_ucits      = tk.BooleanVar(value=False)
        self._results     = None   # everything the Excel export needs, set by _pipeline

        # ── Build layout ──────────────────────────────────────────────────────
        self._build_sidebar()
        self._build_main()
        self._show_panel("config")

    def _center(self):
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"1100x720+{(sw-1100)//2}+{(sh-720)//2}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self.root, bg=C["panel"], width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        tk.Label(self.sidebar, text="PORTFOLIO", font=("Courier New",13,"bold"),
                 fg=C["accent"], bg=C["panel"]).pack(pady=(20,0))
        tk.Label(self.sidebar, text="MANAGER",   font=("Courier New",13,"bold"),
                 fg=C["text"],   bg=C["panel"]).pack()
        tk.Frame(self.sidebar, bg=C["border"], height=1).pack(fill="x", padx=16, pady=12)

        self._nav_btns = {}
        nav_items = [
            ("config",   "⚙  Configuration"),
            ("var",      "📊  Value at Risk"),
            ("sma",      "📈  SMA"),
            ("ema",      "📉  EMA"),
            ("rsi",      "🔄  RSI"),
            ("distrib",  "🗂  Distribution"),
            ("frontier", "🌐  Efficient Frontier"),
            ("gbm",      "🎲  GBM Monte Carlo"),
            ("dashboard","🏠  Dashboard"),
        ]
        for key, label in nav_items:
            btn = tk.Label(self.sidebar, text=label, font=FB, fg=C["sub"],
                           bg=C["panel"], anchor="w", padx=20, pady=9, cursor="hand2")
            btn.pack(fill="x")
            btn.bind("<Button-1>", lambda _, k=key: self._show_panel(k))
            btn.bind("<Enter>",    lambda _, b=btn: b.config(fg=C["text"], bg=C["card"]))
            btn.bind("<Leave>",    lambda _, b=btn, k2=key: (
                b.config(fg=C["accent"] if self._active==k2 else C["sub"],
                         bg=C["border"] if self._active==k2 else C["panel"])))
            self._nav_btns[key] = btn

        # Status bar at bottom of sidebar
        tk.Frame(self.sidebar, bg=C["border"], height=1).pack(
            fill="x", padx=16, side="bottom", pady=8)
        self._status = tk.Label(self.sidebar, text="Ready", font=FS,
                                 fg=C["sub"], bg=C["panel"], wraplength=170)
        self._status.pack(side="bottom", padx=10, pady=4)

        self._active = "config"

    def _show_panel(self, key):
        # Deactivate old
        if hasattr(self, "_active"):
            old = self._nav_btns.get(self._active)
            if old: old.config(fg=C["sub"], bg=C["panel"])
        self._active = key
        btn = self._nav_btns.get(key)
        if btn: btn.config(fg=C["accent"], bg=C["border"])

        # Show panel
        for k, frame in self._panels.items():
            frame.pack_forget()
        self._panels[key].pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN AREA
    # ══════════════════════════════════════════════════════════════════════════
    def _build_main(self):
        self.main = tk.Frame(self.root, bg=C["bg"])
        self.main.pack(side="right", fill="both", expand=True)
        self._panels = {
            "config":   self._build_config_panel(),
            "var":      self._build_chart_panel("var"),
            "sma":      self._build_chart_panel("sma"),
            "ema":      self._build_chart_panel("ema"),
            "rsi":      self._build_chart_panel("rsi"),
            "distrib":  self._build_chart_panel("distrib"),
            "frontier": self._build_chart_panel("frontier"),
            "gbm":      self._build_chart_panel("gbm"),
            "dashboard": self._build_dashboard_panel(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  CONFIG PANEL
    # ══════════════════════════════════════════════════════════════════════════
    def _build_config_panel(self):
        outer = tk.Frame(self.main, bg=C["bg"])

        # Scrollable content
        canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C["bg"])
        win   = canvas.create_window((0,0), window=inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win, width=e.width)
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        inner.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120),"units"))

        pad = tk.Frame(inner, bg=C["bg"])
        pad.pack(fill="both", expand=True, padx=32, pady=24)

        # Title
        tk.Label(pad, text="Portfolio Configuration", font=FT,
                 fg=C["text"], bg=C["bg"]).pack(anchor="w")
        tk.Label(pad, text="Configure your data, allocation and indicators, then click Run.",
                 font=FB, fg=C["sub"], bg=C["bg"]).pack(anchor="w", pady=(2,12))

        # ── Section: Data ─────────────────────────────────────────────────────
        sec = self._card(pad, "DATA SOURCE")
        _sec(sec, "CSV FILE")
        row = tk.Frame(sec, bg=C["card"])
        row.pack(fill="x", pady=(0,4))
        ef, ee = _entry(row, self.v_path, width=38)
        ef.pack(side="left")
        def browse():
            p = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
            if p: self.v_path.set(p); ee.config(fg=C["text"])
        tk.Button(row, text="Browse…", font=FL, fg=C["bg"], bg=C["accent"],
                  relief="flat", padx=12, pady=5, cursor="hand2",
                  command=browse).pack(side="left", padx=8)
        self._cac_btn = tk.Button(row, text="CAC 40 · 1Y (Yahoo)", font=FL, fg=C["bg"],
                                  bg=C["accent2"], relief="flat", padx=12, pady=5,
                                  cursor="hand2", command=self._load_cac40)
        self._cac_btn.pack(side="left")
        self._path_entry = ee

        _sec(sec, "TOTAL INVESTMENT (€)")
        ef2, _ = _entry(sec, self.v_amount, width=16)
        ef2.pack(anchor="w")

        # ── Section: Allocation ───────────────────────────────────────────────
        sec2 = self._card(pad, "ALLOCATION STRATEGY")
        row2 = tk.Frame(sec2, bg=C["card"])
        row2.pack(anchor="w", pady=4)
        self._tb_uni = Toggle(row2, "Uniform split",       self.v_uniform,  self._on_dist)
        self._tb_opt = Toggle(row2, "Markowitz Optimize",  self.v_optimize, self._on_dist)
        self._tb_uni.pack(side="left", padx=(0,8))
        self._tb_opt.pack(side="left")

        self._custom_row = tk.Frame(sec2, bg=C["card"])
        _lbl(self._custom_row, "Custom amounts per asset (comma-separated):",
             fg=C["sub"], bg=C["card"]).pack(anchor="w", pady=(4,2))
        ef3, _ = _entry(self._custom_row, self.v_custom, width=50,
                        ph="e.g.  500, 1000, 750, …")
        ef3.pack(anchor="w")

        # ── Section: VaR confidence ───────────────────────────────────────────
        sec3 = self._card(pad, "VAR CONFIDENCE LEVEL")
        conf_row = tk.Frame(sec3, bg=C["card"])
        conf_row.pack(anchor="w")
        self._conf_btns = {}
        for val, lbl in [("0.90","90%"),("0.95","95%"),("0.99","99%")]:
            b = tk.Label(conf_row, text=lbl, font=FL, fg=C["bg"], bg=C["accent"],
                         padx=16, pady=6, cursor="hand2")
            b.pack(side="left", padx=4)
            b.bind("<Button-1>", lambda _, v=val: self._select_conf(v))
            self._conf_btns[val] = b
        self._select_conf("0.95")

        # ── Section: Indicators ───────────────────────────────────────────────
        sec4 = self._card(pad, "INDICATORS & CHARTS")
        r1 = tk.Frame(sec4, bg=C["card"]); r1.pack(anchor="w", pady=4)
        Toggle(r1, "Value at Risk",          self.v_var).pack(side="left", padx=(0,8))
        Toggle(r1, "RSI 14",                 self.v_rsi).pack(side="left", padx=(0,8))
        Toggle(r1, "Distribution Table",     self.v_distrib).pack(side="left")

        r2 = tk.Frame(sec4, bg=C["card"]); r2.pack(anchor="w", pady=(8,4))
        Toggle(r2, "SMA", self.v_sma).pack(side="left", padx=(0,8))
        ef4, _ = _entry(r2, self.v_sma_p, width=14)
        ef4.pack(side="left", padx=(4,20))
        Toggle(r2, "EMA", self.v_ema).pack(side="left", padx=(0,8))
        ef5, _ = _entry(r2, self.v_ema_p, width=8)
        ef5.pack(side="left")

        r3 = tk.Frame(sec4, bg=C["card"]); r3.pack(anchor="w", pady=(8,0))
        Toggle(r3, "Efficient Frontier (Markowitz)", self.v_frontier).pack(side="left")
        _lbl(r3, "  ⚠ may take 15–30s", fg=C["sub"], bg=C["card"],
             font=FS).pack(side="left", pady=4)

        r4 = tk.Frame(sec4, bg=C["card"]); r4.pack(anchor="w", pady=(8,0))
        Toggle(r4, "Performance Dashboard", self.v_dashboard).pack(side="left", padx=(0,16))
        _lbl(r4, "Horizon (trading days):", fg=C["sub"], bg=C["card"]).pack(side="left")
        ef_dr, _ = _entry(r4, self.v_dash_range, width=6); ef_dr.pack(side="left", padx=6)
        _lbl(r4, "  e.g. 21=1M  63=3M  126=6M  252=1Y", fg=C["sub"], bg=C["card"],
             font=FS).pack(side="left")

        # ── Section: Optimization constraints ────────────────────────────────
        sec5 = self._card(pad, "OPTIMIZATION CONSTRAINTS")
        _sec(sec5, "MINIMUM NUMBER OF ASSETS IN PORTFOLIO")
        min_row = tk.Frame(sec5, bg=C["card"]); min_row.pack(anchor="w", pady=(0,8))
        _lbl(min_row, "Min assets:", fg=C["sub"], bg=C["card"]).pack(side="left", padx=(0,8))
        ef_min, _ = _entry(min_row, self.v_min_assets, width=6)
        ef_min.pack(side="left")
        _lbl(min_row, "  (optimizer ensures at least this many assets have weight > 1%)",
             fg=C["sub"], bg=C["card"], font=FS).pack(side="left", padx=8)

        # ── Section: GBM Monte Carlo ──────────────────────────────────────────
        sec6 = self._card(pad, "GBM MONTE CARLO SIMULATION")
        r6a = tk.Frame(sec6, bg=C["card"]); r6a.pack(anchor="w", pady=4)
        Toggle(r6a, "GBM Portfolio Projection", self.v_gbm_mc).pack(side="left", padx=(0,16))
        _lbl(r6a, "Simulations:", fg=C["sub"], bg=C["card"]).pack(side="left")
        ef_gs, _ = _entry(r6a, self.v_gbm_sims, width=6); ef_gs.pack(side="left", padx=6)
        _lbl(r6a, "Days:", fg=C["sub"], bg=C["card"]).pack(side="left", padx=(12,0))
        ef_gd, _ = _entry(r6a, self.v_gbm_days, width=6); ef_gd.pack(side="left", padx=6)
        r6b = tk.Frame(sec6, bg=C["card"]); r6b.pack(anchor="w", pady=(4,0))
        Toggle(r6b, "Show 4 Shock Scenarios (2008 crash / COVID / Dot-com / Black Monday)",
               self.v_shocks).pack(side="left")

        # ── Section: UCITS regulatory rule ────────────────────────────────────
        sec7 = self._card(pad, "REGULATORY CONSTRAINTS (UCITS FUNDS)")
        r7 = tk.Frame(sec7, bg=C["card"]); r7.pack(anchor="w", pady=4)
        Toggle(r7, "Apply UCITS 5/10/40 rule", self.v_ucits).pack(side="left")
        _lbl(sec7, "Max 10% per asset · assets above 5% capped at 40% in total.",
             fg=C["sub"], bg=C["card"], font=FS).pack(anchor="w", pady=(4,0))
        _lbl(sec7, f"Markowitz: enforced in the optimizer (needs ≥ {UCITS_MIN_ASSETS} assets).  "
                   "Uniform / custom: compliance is checked and reported.",
             fg=C["sub"], bg=C["card"], font=FS).pack(anchor="w")

        # ── Run button ────────────────────────────────────────────────────────
        run_row = tk.Frame(pad, bg=C["bg"])
        run_row.pack(anchor="w", pady=(20,0))
        self._run_btn = tk.Button(
            run_row, text="🚀  Run Analysis", font=("Courier New",11,"bold"),
            fg=C["bg"], bg=C["accent2"], relief="flat",
            padx=24, pady=10, cursor="hand2", command=self._run)
        self._run_btn.pack(side="left")

        self._export_btn = tk.Button(
            run_row, text="📥  Export to Excel", font=("Courier New",11,"bold"),
            fg=C["bg"], bg=C["accent"], relief="flat", state="disabled",
            padx=24, pady=10, cursor="hand2", command=self._export_excel)
        self._export_btn.pack(side="left", padx=(12,0))

        self._pbar = ttk.Progressbar(run_row, mode="indeterminate", length=300)
        style = ttk.Style(); style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=C["muted"],
                         background=C["accent"], thickness=4)

        return outer

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=C["card"], highlightbackground=C["border"],
                          highlightthickness=1)
        outer.pack(fill="x", pady=(0,12))
        tk.Label(outer, text=title, font=FL, fg=C["accent"],
                 bg=C["card"]).pack(anchor="w", padx=14, pady=(10,4))
        tk.Frame(outer, bg=C["border"], height=1).pack(fill="x", padx=14)
        inner = tk.Frame(outer, bg=C["card"])
        inner.pack(fill="x", padx=14, pady=(6,12))
        return inner

    def _on_dist(self):
        if not self.v_uniform.get() and not self.v_optimize.get():
            self._custom_row.pack(fill="x", pady=(4,0))
        else:
            self._custom_row.pack_forget()

    def _select_conf(self, val):
        self.v_conf.set(val)
        for v, b in self._conf_btns.items():
            b.config(fg=C["bg"] if v==val else C["sub"],
                     bg=C["accent"] if v==val else C["muted"])

    # ══════════════════════════════════════════════════════════════════════════
    #  CHART PANELS (placeholders until Run)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_chart_panel(self, key):
        f = tk.Frame(self.main, bg=C["bg"])
        lbl = tk.Label(f, text="Run the analysis to display this chart.",
                       font=FB, fg=C["sub"], bg=C["bg"])
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        return f

    def _replace_chart(self, key, fig):
        """Clear the panel and embed a new matplotlib figure."""
        panel = self._panels[key]
        for w in panel.winfo_children():
            w.destroy()
        _mpl_canvas(panel, fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  RUN PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    def _load_cac40(self):
        """Download the last year of CAC 40 data in the background and select it."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"CAC40_1y_{pd.Timestamp.now():%Y%m%d}.csv")
        self._cac_btn.config(state="disabled")
        self._set_status("Downloading CAC 40 from Yahoo Finance…")

        def work():
            try:
                kept, skipped = download_cac40(path)
            except Exception as ex:
                self.root.after(0, lambda m=str(ex): (
                    self._cac_btn.config(state="normal"),
                    self._set_status("✗ CAC 40 download failed"),
                    messagebox.showerror("CAC 40 download", m)))
                return

            def done():
                self._cac_btn.config(state="normal")
                self.v_path.set(path)
                self._path_entry.config(fg=C["text"])
                self._set_status(f"✔ CAC 40 loaded · {len(kept)} stocks")
                if skipped:
                    messagebox.showinfo("CAC 40 download",
                                        f"{len(kept)} stocks loaded.\n"
                                        f"No data for: {', '.join(skipped)}")
            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _run(self):
        if not os.path.isfile(self.v_path.get()):
            messagebox.showerror("Missing file", "Please select a valid CSV file.")
            return
        try: float(self.v_amount.get())
        except ValueError:
            messagebox.showerror("Invalid amount", "Investment must be a number."); return

        self._run_btn.config(state="disabled")
        self._export_btn.config(state="disabled")
        self._results = None
        self._pbar.pack(side="left", padx=12)
        self._pbar.start(10)
        self._set_status("Loading…")
        threading.Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        try:
            # 1. Load CSV
            self._set_status("Importing data…")
            df_raw = pd.read_csv(self.v_path.get(), index_col=0)
            close_cols  = [c for c in df_raw.columns if "Close"  in c]
            open_cols   = [c for c in df_raw.columns if "Open"   in c]
            high_cols   = [c for c in df_raw.columns if "High"   in c]
            low_cols    = [c for c in df_raw.columns if "Low"    in c]

            df_close = df_raw[close_cols] if close_cols else df_raw
            df_open  = df_raw[open_cols]  if open_cols  else None
            df_high  = df_raw[high_cols]  if high_cols  else None
            df_low   = df_raw[low_cols]   if low_cols   else None

            # 2. Allocation
            self._set_status("Computing allocation…")
            total = float(self.v_amount.get())
            n_assets = df_close.shape[1]

            if self.v_uniform.get():
                amounts = [total / n_assets] * n_assets
            elif self.v_optimize.get():
                min_a = int(self.v_min_assets.get()) if self.v_min_assets.get().isdigit() else 1
                amounts = markowitz_optimize(df_close, total, min_assets=min_a,
                                             ucits=self.v_ucits.get())
            else:
                raw = self.v_custom.get()
                amounts = [float(x.strip()) for x in raw.split(",") if x.strip()]
                if len(amounts) != n_assets:
                    raise ValueError(f"Expected {n_assets} values, got {len(amounts)}")

            # 3. Build portfolio series
            self._set_status("Building portfolio…")
            prices0     = df_close.iloc[0].values
            quantities  = [int(a / p) if p > 0 else 0
                           for a, p in zip(amounts, prices0)]
            if self.v_ucits.get() and self.v_optimize.get() and not self.v_uniform.get():
                quantities = ucits_whole_shares(prices0, amounts, total)
            portfolio_close = (df_close * quantities).sum(axis=1)
            portfolio_close.name = "Portfolio"
            df_portfolio = pd.DataFrame({"Portfolio": portfolio_close})
            df_portfolio.index = pd.to_datetime(df_portfolio.index, dayfirst=False)

            # Asset names
            names = [c.replace("_Close","") for c in df_close.columns]
            df_distrib = pd.DataFrame({"Asset": names, "Qty": quantities,
                                        "Amount (€)": [round(q*p,2) for q,p in
                                                         zip(quantities, df_close.iloc[0])]})
            df_distrib = df_distrib.set_index("Asset")

            # UCITS compliance check on the amounts actually invested
            # (whole-share rounding can shift weights slightly)
            ucits_note = ucits_info = None
            if self.v_ucits.get():
                ok, max_w, big_sum = ucits_check(df_distrib["Amount (€)"].values)
                ucits_note = "UCITS ✔" if ok else "UCITS ✗"
                ucits_info = {"ok": ok, "max_w": max_w, "big_sum": big_sum}
                if not ok:
                    msg = (f"Portfolio is NOT UCITS 5/10/40 compliant.\n\n"
                           f"Largest position: {max_w:.2%}  (limit 10%)\n"
                           f"Sum of positions > 5%: {big_sum:.2%}  (limit 40%)")
                    self.root.after(0, lambda m=msg: messagebox.showwarning("UCITS rule", m))

            # 4. Render charts
            self._set_status("Rendering charts…")
            conf = float(self.v_conf.get())
            periods_sma = [int(x.strip()) for x in self.v_sma_p.get().split(",")
                            if x.strip().isdigit()]
            raw_ema = self.v_ema_p.get().strip()
            period_ema = int(raw_ema) if raw_ema.isdigit() else 12

            # Keep everything the Excel export needs
            if self.v_uniform.get():    method = "Uniform split"
            elif self.v_optimize.get(): method = "Markowitz (max Sharpe)" + (
                                            " + UCITS 5/10/40" if self.v_ucits.get() else "")
            else:                       method = "Custom amounts"
            df_close_dt = df_close.copy()
            df_close_dt.index = df_portfolio.index
            df_close_dt.columns = names
            self._results = {
                "source": self.v_path.get(), "total": total, "method": method,
                "conf": conf, "names": names, "df_close": df_close_dt,
                "portfolio": df_portfolio["Portfolio"], "amounts": amounts,
                "quantities": quantities, "sma_periods": periods_sma,
                "ema_period": period_ema, "ucits": ucits_info,
                "dash_days": int(self.v_dash_range.get())
                             if self.v_dash_range.get().isdigit() else 63,
            }

            if self.v_var.get():
                self.root.after(0, lambda: self._plot_var(df_portfolio, conf))

            if self.v_sma.get():
                self.root.after(0, lambda p=periods_sma: self._plot_sma(df_portfolio, p))

            if self.v_ema.get():
                self.root.after(0, lambda p=period_ema: self._plot_ema(df_portfolio, p))

            if self.v_rsi.get():
                self.root.after(0, lambda: self._plot_rsi(df_portfolio))

            if self.v_distrib.get():
                self.root.after(0, lambda: self._plot_distrib(df_distrib, names, quantities,
                                                                df_close))

            if self.v_frontier.get():
                self._set_status("Computing Efficient Frontier…")
                w_arr   = np.array(amounts, dtype=float)
                w_arr  /= w_arr.sum()
                ret_df  = df_close.pct_change().dropna()
                mu      = ret_df.mean().values
                cov     = ret_df.cov().values
                mc_r, mc_v, mc_s = self._monte_carlo(mu, cov, n_assets)
                fr_v, fr_r       = self._frontier_curve(mu, cov, n_assets)
                self._results["frontier"] = (fr_v, fr_r)
                self.root.after(0, lambda: self._plot_frontier(
                    mc_r, mc_v, mc_s, fr_v, fr_r, w_arr, mu, cov, names))

            if self.v_gbm_mc.get():
                self._set_status("Running GBM Monte Carlo…")
                w_gbm   = np.array(amounts, dtype=float)
                w_gbm  /= w_gbm.sum()
                ret_df2 = df_close.pct_change().dropna()
                mu_gbm  = ret_df2.mean().values
                cov_gbm = ret_df2.cov().values
                n_sims  = int(self.v_gbm_sims.get()) if self.v_gbm_sims.get().isdigit() else 200
                n_days  = int(self.v_gbm_days.get()) if self.v_gbm_days.get().isdigit() else 252
                show_shocks = self.v_shocks.get()
                initial_val = float(self.v_amount.get())
                self.root.after(0, lambda: self._plot_gbm(
                    mu_gbm, cov_gbm, w_gbm, n_sims, n_days, initial_val, show_shocks, names))

            if self.v_dashboard.get():
                self._set_status("Building Dashboard…")
                w_dash  = np.array(amounts, dtype=float)
                w_dash /= w_dash.sum()
                ret_dash = df_close.pct_change().dropna()
                mu_dash  = ret_dash.mean().values
                cov_dash = ret_dash.cov().values
                n_days_d = int(self.v_dash_range.get()) if self.v_dash_range.get().isdigit() else 63
                init_d   = float(self.v_amount.get())
                conf_d   = float(self.v_conf.get())
                port_rets = df_portfolio["Portfolio"].pct_change().dropna()
                self.root.after(0, lambda: self._build_dashboard_data(
                    mu_dash, cov_dash, w_dash, n_days_d, init_d, conf_d,
                    port_rets, names, df_close))

            self.root.after(0, lambda: self._done(ucits_note))

        except Exception as ex:
            self.root.after(0, lambda: self._on_error(str(ex)))

    def _monte_carlo(self, mu, cov, n, n_sim=4000):
        rets, vols, sharpes = [], [], []
        for _ in tqdm(range(n_sim), desc="Monte-Carlo", ncols=60):
            w = np.random.dirichlet(np.ones(n))
            r,v,s = portfolio_perf(w, mu, cov)
            rets.append(r); vols.append(v); sharpes.append(s)
        return np.array(rets), np.array(vols), np.array(sharpes)

    def _frontier_curve(self, mu, cov, n, n_pts=100):
        w0 = min_variance(mu, cov, n)
        r_min,_,_ = portfolio_perf(w0, mu, cov)
        r_max = mu.max() * 252 * 0.98
        fv, fr = [], []
        for t in tqdm(np.linspace(r_min, r_max, n_pts), desc="Frontier", ncols=60):
            res = min_var_target(t/252, mu, cov, n)
            if res:
                v = np.sqrt(res.x @ cov @ res.x) * np.sqrt(252)
                fv.append(v); fr.append(t)
        return np.array(fv), np.array(fr)

    def _set_status(self, msg):
        self.root.after(0, lambda: self._status.config(text=msg))

    def _done(self, ucits_note=None):
        self._pbar.stop()
        self._pbar.pack_forget()
        self._run_btn.config(state="normal")
        self._export_btn.config(state="normal")
        self._set_status("✔ Analysis complete" + (f"  ·  {ucits_note}" if ucits_note else ""))
        # Auto-navigate to first enabled chart
        for key in ["dashboard","var","sma","ema","rsi","distrib","frontier","gbm"]:
            v = {"var":self.v_var,"sma":self.v_sma,"ema":self.v_ema,
                 "rsi":self.v_rsi,"distrib":self.v_distrib,"frontier":self.v_frontier,
                 "gbm":self.v_gbm_mc,"dashboard":self.v_dashboard}
            if v[key].get():
                self._show_panel(key); break

    def _on_error(self, msg):
        self._pbar.stop(); self._pbar.pack_forget()
        self._run_btn.config(state="normal")
        self._set_status(f"✗ Error")
        messagebox.showerror("Error", msg)

    def _export_excel(self):
        if not self._results:
            messagebox.showinfo("Nothing to export", "Run the analysis first."); return
        stem = os.path.splitext(os.path.basename(self._results["source"]))[0]
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel workbook", "*.xlsx")],
            initialfile=f"portfolio_{stem}_{pd.Timestamp.now():%Y%m%d}.xlsx")
        if not path:
            return
        self._set_status("Exporting to Excel…")
        try:
            export_to_excel(path, self._results)
        except PermissionError:
            messagebox.showerror("Export failed",
                                 "Cannot write the file — is it open in Excel?")
            self._set_status("✗ Export failed"); return
        except Exception as ex:
            messagebox.showerror("Export failed", str(ex))
            self._set_status("✗ Export failed"); return
        self._set_status("✔ Exported to Excel")
        if hasattr(os, "startfile") and messagebox.askyesno(
                "Export complete", f"Saved to:\n{path}\n\nOpen it now?"):
            os.startfile(path)

    # ══════════════════════════════════════════════════════════════════════════
    #  CHART BUILDERS
    # ══════════════════════════════════════════════════════════════════════════
    def _fig(self, w=10, h=5):
        mpl_style()
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(MP["bg"])
        return fig, ax

    # ── VaR ───────────────────────────────────────────────────────────────────
    def _plot_var(self, df, conf):
        fig, ax = self._fig(10, 5)
        var_val = calc_var(df["Portfolio"], conf)
        returns = df["Portfolio"].pct_change().dropna()

        n, bins, patches = ax.hist(returns, bins="auto",
                                    color=MP["accent"], alpha=0.35, edgecolor="none")
        for patch, left in zip(patches, bins[:-1]):
            if left < var_val:
                patch.set_facecolor(MP["danger"]); patch.set_alpha(0.55)

        ax.axvline(var_val, color=MP["danger"], lw=1.8, ls="--")
        ax.text(var_val, n.max()*0.88, f"  VaR {conf:.0%}\n  {var_val:.4%}",
                color=MP["danger"], fontsize=8, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.2%}"))
        ax.set_title("Value at Risk — Historical Simulation")
        ax.set_xlabel("Daily Return"); ax.set_ylabel("Frequency")
        _tag(ax, f"VaR {conf:.0%}")
        fig.tight_layout()
        self._replace_chart("var", fig)

    # ── SMA ───────────────────────────────────────────────────────────────────
    def _plot_sma(self, df, periods):
        fig, ax = self._fig(10, 5)
        colors  = [MP["sma1"], MP["sma2"], MP["accent3"]]
        ax.plot(df.index, df["Portfolio"], color=MP["accent"], lw=1.4, label="Portfolio")
        for i, p in enumerate(periods):
            s = calc_sma(df["Portfolio"], p)
            ax.plot(df.index, s, color=colors[i%len(colors)], lw=1.1,
                    ls="--", label=f"SMA {p}")
        _date_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x:,.0f}"))
        ax.set_title("Simple Moving Average"); ax.legend(loc="upper left")
        _tag(ax, "SMA"); fig.tight_layout()
        self._replace_chart("sma", fig)

    # ── EMA ───────────────────────────────────────────────────────────────────
    def _plot_ema(self, df, period):
        fig, ax = self._fig(10, 5)
        ax.plot(df.index, df["Portfolio"], color=MP["accent"], lw=1.4, label="Portfolio")
        e = calc_ema(df["Portfolio"], period)
        ax.plot(df.index, e, color=MP["ema1"], lw=1.1, ls="--", label=f"EMA {period}")
        _date_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x:,.0f}"))
        ax.set_title("Exponential Moving Average"); ax.legend(loc="upper left")
        _tag(ax, "EMA"); fig.tight_layout()
        self._replace_chart("ema", fig)

    # ── RSI ───────────────────────────────────────────────────────────────────
    def _plot_rsi(self, df):
        fig, ax = self._fig(10, 5)
        rsi_vals = calc_rsi(df["Portfolio"])
        idx  = df.index
        rsi_s = pd.Series(rsi_vals, index=idx)
        ax.fill_between(idx, rsi_s, 70, where=(rsi_s>=70),
                         color=MP["danger"],  alpha=0.18, interpolate=True)
        ax.fill_between(idx, rsi_s, 30, where=(rsi_s<=30),
                         color=MP["pos"],     alpha=0.18, interpolate=True)
        ax.plot(idx, rsi_s, color=MP["accent3"], lw=1.4, label="RSI 14")
        ax.axhline(70, color=MP["danger"], lw=0.9, ls="--", label="Overbought (70)")
        ax.axhline(30, color=MP["pos"],    lw=0.9, ls="--", label="Oversold (30)")
        ax.axhline(50, color=MP["sub"],    lw=0.6, ls=":")
        ax.set_ylim(0, 100); _date_axis(ax)
        ax.set_title("Relative Strength Index"); ax.legend(loc="upper left")
        _tag(ax, "RSI 14"); fig.tight_layout()
        self._replace_chart("rsi", fig)

    # ── Distribution ──────────────────────────────────────────────────────────
    def _plot_distrib(self, df_distrib, names, quantities, df_close):
        mpl_style()
        fig = plt.figure(figsize=(10, 5.5))
        fig.patch.set_facecolor(MP["bg"])

        gs = gridspec.GridSpec(1, 2, figure=fig, left=0.04, right=0.97,
                               top=0.90, bottom=0.10, wspace=0.35)
        ax_tbl = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])

        # ── Tableau propre ────────────────────────────────────────────────────
        ax_tbl.set_facecolor(MP["panel"]); ax_tbl.axis("off")
        col_labels = ["Qty", "Amount (€)"]
        cell_vals  = [[str(r["Qty"]), f"€{r['Amount (€)']:,.2f}"]
                       for _, r in df_distrib.iterrows()]
        tbl = ax_tbl.table(cellText=cell_vals, colLabels=col_labels,
                            rowLabels=df_distrib.index.tolist(),
                            loc="center", cellLoc="right")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.55)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(MP["border"]); cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor(MP["accent"])
                cell.set_text_props(color=MP["bg"], fontweight="bold")
            elif col == -1:
                cell.set_facecolor(MP["panel"])
                cell.set_text_props(color=MP["sub"])
            else:
                cell.set_facecolor(MP["bg"] if row%2==0 else MP["panel"])
                cell.set_text_props(color=MP["text"])
        ax_tbl.set_title("Allocation Table", color=MP["text"], fontsize=10,
                          fontweight="semibold", pad=10)
        _tag(ax_tbl, "ALLOC")

        # ── Bar chart ─────────────────────────────────────────────────────────
        ax_bar.set_facecolor(MP["panel"])
        prices0  = df_close.iloc[0].values
        vals     = [q*p for q,p in zip(quantities, prices0)]
        total    = sum(vals) or 1
        weights  = [v/total for v in vals]
        idx_sort = np.argsort(weights)[::-1][:15]  # top 15

        n_s = [names[i]   for i in idx_sort]
        w_s = [weights[i] for i in idx_sort]
        cols = [MP["accent"] if w >= np.mean(weights) else MP["sub"] for w in w_s]

        bars = ax_bar.barh(n_s, w_s, color=cols, height=0.6, alpha=0.85)
        for bar, val in zip(bars, w_s):
            ax_bar.text(bar.get_width()+0.002,
                        bar.get_y()+bar.get_height()/2,
                        f"{val:.1%}", va="center", fontsize=7, color=MP["text"])

        ax_bar.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.0%}"))
        ax_bar.invert_yaxis()
        ax_bar.set_title("Weight per Asset (top 15)", color=MP["text"],
                          fontsize=10, fontweight="semibold", pad=10)
        ax_bar.grid(axis="x", color=MP["grid"], lw=0.5)
        ax_bar.set_axisbelow(True)
        _tag(ax_bar, "WEIGHTS")

        fig.suptitle("Portfolio Distribution", color=MP["text"],
                      fontsize=12, fontweight="bold", y=0.97)
        self._replace_chart("distrib", fig)

    # ── Efficient Frontier ────────────────────────────────────────────────────
    def _plot_frontier(self, mc_r, mc_v, mc_s, fr_v, fr_r,
                        weights, mu, cov, names):
        mpl_style()
        fig = plt.figure(figsize=(10, 5.5))
        fig.patch.set_facecolor(MP["bg"])
        gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                                top=0.88, bottom=0.10, wspace=0.30)
        ax  = fig.add_subplot(gs[0, 0])
        axb = fig.add_subplot(gs[0, 1])

        # Scatter MC
        sc = ax.scatter(mc_v, mc_r, c=mc_s, cmap="plasma",
                        alpha=0.30, s=5, linewidths=0, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
        cbar.set_label("Sharpe", color=MP["sub"], fontsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MP["sub"], fontsize=6)
        cbar.outline.set_edgecolor(MP["border"])
        cbar.ax.yaxis.set_tick_params(color=MP["sub"])

        # Frontier curve
        if len(fr_v) > 2:
            ax.plot(fr_v, fr_r, color=MP["accent"], lw=2.2, zorder=4,
                    label="Efficient Frontier")

        # Key portfolios
        ws  = max_sharpe(mu, cov, len(mu))
        wm  = min_variance(mu, cov, len(mu))
        sr,sv,ss = portfolio_perf(ws, mu, cov)
        mr,mv,ms = portfolio_perf(wm, mu, cov)
        pr,pv,ps = portfolio_perf(weights, mu, cov)
        p_color  = MP["pos"] if ps >= ss*0.75 else MP["danger"]

        for (vol,ret,lbl,color,marker,sz,sharpe) in [
            (sv,sr,"Max Sharpe",   MP["accent3"],"*", 200, ss),
            (mv,mr,"Min Variance", MP["accent2"],"D", 80,  ms),
            (pv,pr,"Your Portfolio",p_color,     "P", 160, ps),
        ]:
            ax.scatter(vol, ret, marker=marker, s=sz, color=color,
                       zorder=6, label=f"{lbl}  ({sharpe:.2f})")
            ax.annotate(f" {lbl}\n {ret:.1%} / {vol:.1%}",
                        xy=(vol,ret), xytext=(vol+0.005, ret+0.005),
                        color=color, fontsize=6.5,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1%}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.1%}"))
        ax.set_xlabel("Annualised Volatility"); ax.set_ylabel("Annualised Return")
        ax.set_title("Risk / Return Space")
        ax.legend(loc="lower right", fontsize=7)
        _tag(ax, "FRONTIER")

        # Bar chart des poids
        axb.set_facecolor(MP["panel"])
        idx_s  = np.argsort(weights)[::-1][:15]
        ns     = [names[i] for i in idx_s]
        ws_arr = [weights[i] for i in idx_s]
        mean_w = np.mean(weights)
        bcols  = [MP["accent"] if w>=mean_w else MP["sub"] for w in ws_arr]
        bars   = axb.barh(ns, ws_arr, color=bcols, height=0.6, alpha=0.85)
        for bar,val in zip(bars,ws_arr):
            axb.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                     f"{val:.1%}", va="center", fontsize=7, color=MP["text"])
        axb.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.0%}"))
        axb.invert_yaxis()
        axb.set_title("Your Portfolio Weights", color=MP["text"],
                       fontsize=10, fontweight="semibold", pad=10)
        axb.grid(axis="x", color=MP["grid"], lw=0.5); axb.set_axisbelow(True)
        _tag(axb, "WEIGHTS")

        fig.suptitle("Markowitz Efficient Frontier", color=MP["text"],
                      fontsize=12, fontweight="bold", y=0.97)
        self._replace_chart("frontier", fig)


    # ── GBM Monte Carlo ───────────────────────────────────────────────────────
    def _plot_gbm(self, mu, cov, weights, n_sims, n_days, initial_val, show_shocks, names):
        """
        Geometric Brownian Motion projection of the optimised portfolio.
        Optionally overlays 4 panels for historical shock scenarios.
        """
        mpl_style()
        L = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-10)

        # ── GBM simulation ────────────────────────────────────────────────────
        port_mu  = float(np.dot(weights, mu))          # daily drift
        port_var = float(weights @ cov @ weights)      # daily variance
        port_sig = float(np.sqrt(port_var))

        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = initial_val
        for t in range(1, n_days + 1):
            z = np.random.standard_normal(n_sims)
            paths[:, t] = paths[:, t-1] * np.exp(
                (port_mu - 0.5 * port_var) + port_sig * z)

        days_arr = np.arange(n_days + 1)
        pct_5    = np.percentile(paths, 5,  axis=0)
        pct_25   = np.percentile(paths, 25, axis=0)
        pct_50   = np.percentile(paths, 50, axis=0)
        pct_75   = np.percentile(paths, 75, axis=0)
        pct_95   = np.percentile(paths, 95, axis=0)
        gbm_export = {"n_sims": n_sims, "shocks": [], "percentiles": pd.DataFrame(
            {"5th pct": pct_5, "25th pct": pct_25, "Median": pct_50,
             "75th pct": pct_75, "95th pct": pct_95},
            index=pd.Index(days_arr, name="Day"))}
        if self._results is not None:
            self._results["gbm"] = gbm_export

        # ── Shock scenarios (approximate GBM with shocked drift/vol) ──────────
        SHOCKS = [
            {
                "name":  "2008 Financial Crisis",
                "color": MP["danger"],
                "drop":  -0.50,   # peak-to-trough ~50%
                "vol":    0.040,  # daily vol spike
                "days":   375,    # duration in trading days
                "desc":  "Lehman collapse · -50% peak-to-trough · 375 days",
            },
            {
                "name":  "COVID-19 Crash (2020)",
                "color": MP["accent3"],
                "drop":  -0.34,
                "vol":    0.050,
                "days":   23,
                "desc":  "Fastest 30%+ drop in history · -34% · 23 days",
            },
            {
                "name":  "Dot-com Bust (2000-02)",
                "color": "#C084FC",
                "drop":  -0.49,
                "vol":    0.025,
                "days":   628,
                "desc":  "NASDAQ -78% · Broad market -49% · 628 days",
            },
            {
                "name":  "Black Monday (1987)",
                "color": "#FB7185",
                "drop":  -0.34,
                "vol":    0.080,
                "days":   1,
                "desc":  "Single-day -22.6% · Highest 1-day vol ever",
            },
        ]

        def shock_paths(shock, n_sim, horizon, start_val):
            """Return simulated paths through a shock scenario."""
            drop_factor = 1 + shock["drop"]
            shock_days  = min(shock["days"], horizon)
            heal_days   = horizon - shock_days
            sig_s       = shock["vol"]
            mu_s        = np.log(drop_factor) / shock_days - 0.5 * sig_s**2

            out = np.zeros((n_sim, horizon + 1))
            out[:, 0] = start_val
            for t in range(1, shock_days + 1):
                z = np.random.standard_normal(n_sim)
                out[:, t] = out[:, t-1] * np.exp(mu_s + sig_s * z)
            # Recovery phase: drift back at normal portfolio drift
            for t in range(shock_days, horizon):
                z = np.random.standard_normal(n_sim)
                out[:, t+1] = out[:, t] * np.exp(
                    (port_mu - 0.5*port_var) + port_sig * z)
            return out

        if not show_shocks:
            # ── Single panel layout ───────────────────────────────────────────
            fig = plt.figure(figsize=(10, 5.5))
            fig.patch.set_facecolor(MP["bg"])
            ax = fig.add_subplot(111)
            ax.set_facecolor(MP["panel"])

            # Draw a sample of paths
            sample = min(80, n_sims)
            for i in range(sample):
                ax.plot(days_arr, paths[i], color=MP["accent"], alpha=0.04, lw=0.6)

            ax.fill_between(days_arr, pct_5,  pct_95, color=MP["accent"], alpha=0.12,
                            label="5th–95th pct")
            ax.fill_between(days_arr, pct_25, pct_75, color=MP["accent"], alpha=0.22,
                            label="25th–75th pct")
            ax.plot(days_arr, pct_50, color=MP["accent2"], lw=2.0,
                    label=f"Median  €{pct_50[-1]:,.0f}")
            ax.plot(days_arr, pct_5,  color=MP["danger"],  lw=1.2, ls="--",
                    label=f"5th pct  €{pct_5[-1]:,.0f}")
            ax.plot(days_arr, pct_95, color=MP["pos"],     lw=1.2, ls="--",
                    label=f"95th pct €{pct_95[-1]:,.0f}")
            ax.axhline(initial_val, color=MP["sub"], lw=0.8, ls=":", label="Initial value")

            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x:,.0f}"))
            ax.set_xlabel("Trading Days"); ax.set_ylabel("Portfolio Value (€)")
            ax.set_title(f"GBM Portfolio Projection  ·  {n_sims} simulations  ·  {n_days} days")
            ax.legend(loc="upper left", fontsize=7)
            _tag(ax, "GBM MC")
            fig.tight_layout()

        else:
            # ── 2×3 layout: main GBM + 4 shock panels ────────────────────────
            fig = plt.figure(figsize=(14, 9))
            fig.patch.set_facecolor(MP["bg"])
            gs  = gridspec.GridSpec(2, 3, figure=fig,
                                    left=0.06, right=0.97, top=0.93, bottom=0.07,
                                    wspace=0.30, hspace=0.40)

            ax_main = fig.add_subplot(gs[0, :])   # full top row
            ax_main.set_facecolor(MP["panel"])

            sample = min(60, n_sims)
            for i in range(sample):
                ax_main.plot(days_arr, paths[i], color=MP["accent"], alpha=0.04, lw=0.6)
            ax_main.fill_between(days_arr, pct_5,  pct_95, color=MP["accent"], alpha=0.12)
            ax_main.fill_between(days_arr, pct_25, pct_75, color=MP["accent"], alpha=0.22)
            ax_main.plot(days_arr, pct_50, color=MP["accent2"], lw=2.0,
                         label=f"Median €{pct_50[-1]:,.0f}")
            ax_main.plot(days_arr, pct_5,  color=MP["danger"],  lw=1.1, ls="--",
                         label=f"5th pct €{pct_5[-1]:,.0f}")
            ax_main.plot(days_arr, pct_95, color=MP["pos"],     lw=1.1, ls="--",
                         label=f"95th pct €{pct_95[-1]:,.0f}")
            ax_main.axhline(initial_val, color=MP["sub"], lw=0.8, ls=":")
            ax_main.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x:,.0f}"))
            ax_main.set_title(f"GBM Portfolio Projection  ·  {n_sims} sims  ·  {n_days} days")
            ax_main.legend(loc="upper left", fontsize=7)
            _tag(ax_main, "GBM MC")

            shock_axes = [
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2]),
                # 4th shock needs its own row — add below gs
            ]
            # We have 3 slots in row 1; embed 4th inside a tight extra axis
            # Rebuild: use 3 rows with last row split
            # Simpler: use 2×2 for shocks in a nested gs
            plt.close(fig)

            fig = plt.figure(figsize=(14, 10))
            fig.patch.set_facecolor(MP["bg"])
            outer_gs = gridspec.GridSpec(2, 1, figure=fig,
                                         left=0.06, right=0.97,
                                         top=0.94, bottom=0.05,
                                         hspace=0.38)
            top_ax  = fig.add_subplot(outer_gs[0])
            top_ax.set_facecolor(MP["panel"])

            for i in range(sample):
                top_ax.plot(days_arr, paths[i], color=MP["accent"], alpha=0.04, lw=0.6)
            top_ax.fill_between(days_arr, pct_5,  pct_95, color=MP["accent"], alpha=0.12)
            top_ax.fill_between(days_arr, pct_25, pct_75, color=MP["accent"], alpha=0.22)
            top_ax.plot(days_arr, pct_50, color=MP["accent2"], lw=2.0,
                        label=f"Median €{pct_50[-1]:,.0f}")
            top_ax.plot(days_arr, pct_5,  color=MP["danger"],  lw=1.1, ls="--",
                        label=f"5th pct €{pct_5[-1]:,.0f}")
            top_ax.plot(days_arr, pct_95, color=MP["pos"],     lw=1.1, ls="--",
                        label=f"95th pct €{pct_95[-1]:,.0f}")
            top_ax.axhline(initial_val, color=MP["sub"], lw=0.8, ls=":",
                           label=f"Initial €{initial_val:,.0f}")
            top_ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x:,.0f}"))
            top_ax.set_xlabel("Trading Days"); top_ax.set_ylabel("Portfolio Value (€)")
            top_ax.set_title(
                f"GBM Portfolio Projection  ·  {n_sims} simulations  ·  {n_days} days",
                fontsize=11, fontweight="semibold")
            top_ax.legend(loc="upper left", fontsize=7)
            _tag(top_ax, "GBM MC")

            shock_gs = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=outer_gs[1], wspace=0.32)

            n_shock_sims = min(n_sims, 150)
            horizon      = n_days

            for col, shock in enumerate(SHOCKS):
                sax = fig.add_subplot(shock_gs[col])
                sax.set_facecolor(MP["panel"])
                sp  = shock_paths(shock, n_shock_sims, horizon, initial_val)

                sp5  = np.percentile(sp, 5,  axis=0)
                sp50 = np.percentile(sp, 50, axis=0)
                sp95 = np.percentile(sp, 95, axis=0)
                t_arr = np.arange(horizon + 1)

                # Sample paths
                for i in range(min(30, n_shock_sims)):
                    sax.plot(t_arr, sp[i], color=shock["color"], alpha=0.06, lw=0.5)

                sax.fill_between(t_arr, sp5, sp95, color=shock["color"], alpha=0.18)
                sax.plot(t_arr, sp50,  color=shock["color"], lw=1.8,
                         label=f"Median €{sp50[-1]:,.0f}")
                sax.plot(t_arr, sp5,   color=shock["color"], lw=0.9, ls="--",
                         label=f"5th  €{sp5[-1]:,.0f}")
                sax.axhline(initial_val, color=MP["sub"], lw=0.7, ls=":")

                # Mark the trough
                trough_idx = np.argmin(sp50)
                sax.axvline(trough_idx, color=shock["color"], lw=0.8, ls=":",
                            alpha=0.6)
                sax.scatter([trough_idx], [sp50[trough_idx]],
                            color=shock["color"], s=30, zorder=6)

                drop_pct = (sp50[trough_idx] / initial_val - 1) * 100
                gbm_export["shocks"].append((shock["name"], shock["desc"], drop_pct / 100,
                                             int(trough_idx), sp50[-1], sp5[-1]))
                sax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"€{x/1e3:.0f}k"))
                sax.set_title(shock["name"], fontsize=7.5, fontweight="semibold",
                              color=shock["color"])
                sax.set_xlabel("Days", fontsize=7)
                sax.legend(loc="lower right", fontsize=6)
                _tag(sax, f"{drop_pct:+.0f}% trough")

                # Desc annotation
                sax.text(0.02, 0.97, shock["desc"], transform=sax.transAxes,
                         fontsize=5.5, color=MP["sub"], va="top", wrap=True)

        fig.suptitle("GBM Monte Carlo — Optimised Portfolio", color=MP["text"],
                     fontsize=12, fontweight="bold", y=0.99)
        self._replace_chart("gbm", fig)


    # ══════════════════════════════════════════════════════════════════════════
    #  DASHBOARD PANEL
    # ══════════════════════════════════════════════════════════════════════════
    def _build_dashboard_panel(self):
        """Build the skeleton Tkinter dashboard (populated later by _build_dashboard_data)."""
        outer = tk.Frame(self.main, bg=C["bg"])

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(outer, bg=C["bg"])
        hdr.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(hdr, text="Performance Dashboard", font=FT,
                 fg=C["text"], bg=C["bg"]).pack(side="left")
        self._dash_range_lbl = tk.Label(hdr, text="", font=FB,
                                         fg=C["sub"], bg=C["bg"])
        self._dash_range_lbl.pack(side="left", padx=16)

        # ── KPI row ───────────────────────────────────────────────────────────
        kpi_row = tk.Frame(outer, bg=C["bg"])
        kpi_row.pack(fill="x", padx=24, pady=(8, 0))
        self._kpi_frames = {}
        kpi_keys = [
            ("est_perf",    "Est. Performance",    C["accent2"]),
            ("est_vol",     "Est. Volatility",     C["accent"]),
            ("cur_vol",     "Current Volatility",  C["accent3"]),
            ("var_kpi",     "Value at Risk (95%)", C["danger"]),
            ("sharpe",      "Sharpe Ratio",        C["pos"]),
            ("risk_ind",    "Risk Indicator",      C["accent3"]),
        ]
        for i, (key, label, color) in enumerate(kpi_keys):
            card = tk.Frame(kpi_row, bg=C["card"],
                            highlightbackground=color,
                            highlightthickness=1)
            card.pack(side="left", expand=True, fill="both", padx=(0, 10 if i < 5 else 0))
            tk.Label(card, text=label, font=FS, fg=C["sub"],
                     bg=C["card"]).pack(anchor="w", padx=12, pady=(10, 2))
            val_lbl = tk.Label(card, text="—", font=("Courier New", 18, "bold"),
                               fg=color, bg=C["card"])
            val_lbl.pack(anchor="w", padx=12)
            sub_lbl = tk.Label(card, text="", font=FS, fg=C["sub"], bg=C["card"])
            sub_lbl.pack(anchor="w", padx=12, pady=(0, 10))
            self._kpi_frames[key] = (val_lbl, sub_lbl, color)

        # ── Risk gauge bar ────────────────────────────────────────────────────
        gauge_row = tk.Frame(outer, bg=C["bg"])
        gauge_row.pack(fill="x", padx=24, pady=(14, 0))
        tk.Label(gauge_row, text="RISK LEVEL", font=FL,
                 fg=C["sub"], bg=C["bg"]).pack(anchor="w")
        gauge_bg = tk.Frame(gauge_row, bg=C["muted"], height=14,
                            highlightbackground=C["border"], highlightthickness=1)
        gauge_bg.pack(fill="x", pady=(4, 2))
        gauge_bg.pack_propagate(False)
        self._gauge_fill = tk.Frame(gauge_bg, bg=C["accent2"], height=14)
        self._gauge_fill.place(x=0, y=0, relheight=1, relwidth=0.0)
        self._gauge_lbl = tk.Label(gauge_row, text="", font=FS, fg=C["sub"], bg=C["bg"])
        self._gauge_lbl.pack(anchor="w")

        # ── Separator ─────────────────────────────────────────────────────────
        tk.Frame(outer, bg=C["border"], height=1).pack(fill="x", padx=24, pady=12)

        # ── Chart area (matplotlib projection) ───────────────────────────────
        self._dash_chart_frame = tk.Frame(outer, bg=C["bg"])
        self._dash_chart_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        tk.Label(self._dash_chart_frame,
                 text="Run the analysis to display the performance projection.",
                 font=FB, fg=C["sub"], bg=C["bg"]).place(relx=0.5, rely=0.5, anchor="center")

        self._dash_panel = outer
        return outer

    def _kpi_set(self, key, value, sub=""):
        """Update a KPI card's value and subtitle labels."""
        val_lbl, sub_lbl, _ = self._kpi_frames[key]
        val_lbl.config(text=value)
        sub_lbl.config(text=sub)

    def _build_dashboard_data(self, mu, cov, weights, n_days, initial_val,
                               conf, port_rets, names, df_close):
        """Compute all dashboard metrics and render the panel."""
        # ── 1-7. Metrics (shared with the Excel export) ───────────────────────
        m = dashboard_metrics(mu, cov, weights, n_days, conf, port_rets)
        est_perf_pct, est_vol_ann, cur_vol_ann = m["est_perf"], m["est_vol"], m["cur_vol"]
        var_horiz, sharpe_ann = m["var_horizon"], m["sharpe"]
        risk_score, risk_label = m["risk_score"], m["risk_label"]
        est_perf_eur = initial_val * est_perf_pct
        var_eur      = initial_val * abs(var_horiz)
        risk_color   = {"LOW": C["pos"], "MEDIUM": C["accent3"], "HIGH": C["danger"]}[risk_label]

        # ── 8. Update range label ─────────────────────────────────────────────
        months_approx = round(n_days / 21, 1)
        self._dash_range_lbl.config(
            text=f"Horizon: {n_days} trading days  ≈  {months_approx} months")

        # ── 9. Update KPI cards ───────────────────────────────────────────────
        perf_sign = "+" if est_perf_pct >= 0 else ""
        self._kpi_set("est_perf",
                      f"{perf_sign}{est_perf_pct:.2%}",
                      f"{perf_sign}€{est_perf_eur:,.0f}  over {n_days}d")
        self._kpi_set("est_vol",
                      f"{est_vol_ann:.2%}",
                      "Annualised (model)")
        self._kpi_set("cur_vol",
                      f"{cur_vol_ann:.2%}",
                      "Annualised (last 21d realised)")
        self._kpi_set("var_kpi",
                      f"{abs(var_horiz):.2%}",
                      f"≈ €{var_eur:,.0f}  at {int(conf*100)}% conf. over {n_days}d")
        sharpe_sign = "+" if sharpe_ann >= 0 else ""
        self._kpi_set("sharpe",
                      f"{sharpe_sign}{sharpe_ann:.2f}",
                      "Annualised (rf = 0%)")
        self._kpi_set("risk_ind",
                      f"{risk_score:.1f} / 10",
                      risk_label)
        # colour-code risk card dynamically
        val_lbl, sub_lbl, _ = self._kpi_frames["risk_ind"]
        val_lbl.config(fg=risk_color)
        sub_lbl.config(fg=risk_color)

        # ── 10. Update gauge ──────────────────────────────────────────────────
        gauge_pct = risk_score / 10
        self._gauge_fill.place(relwidth=gauge_pct)
        self._gauge_fill.config(bg=risk_color)
        self._gauge_lbl.config(
            text=f"Risk score {risk_score:.1f}/10  —  {risk_label}  "
                 f"(vol {est_vol_ann:.1%}  |  VaR{int(conf*100)} {abs(var_horiz):.1%}  "
                 f"|  Sharpe {sharpe_ann:.2f})",
            fg=risk_color)

        # ── 11. GBM projection chart ──────────────────────────────────────────
        self._plot_dashboard_chart(
            mu, cov, weights, n_days, initial_val, est_perf_pct,
            var_horiz, est_vol_ann, cur_vol_ann, sharpe_ann, risk_score,
            risk_label, risk_color, names, df_close)

    def _plot_dashboard_chart(self, mu, cov, weights, n_days, initial_val,
                               est_perf_pct, var_horiz, est_vol_ann,
                               cur_vol_ann, sharpe_ann, risk_score,
                               risk_label, risk_color, names, df_close):
        """Render the 3-panel dashboard chart: projection / top assets / rolling vol."""
        mpl_style()

        port_mu_d  = float(np.dot(weights, mu))
        port_var_d = float(weights @ cov @ weights)
        port_sig_d = float(np.sqrt(port_var_d))
        TD = 252

        # GBM paths
        N_SIM = 300
        paths = np.zeros((N_SIM, n_days + 1))
        paths[:, 0] = initial_val
        for t in range(1, n_days + 1):
            z = np.random.standard_normal(N_SIM)
            paths[:, t] = paths[:, t-1] * np.exp(
                (port_mu_d - 0.5 * port_var_d) + port_sig_d * z)

        t_arr = np.arange(n_days + 1)
        p5    = np.percentile(paths,  5, axis=0)
        p25   = np.percentile(paths, 25, axis=0)
        p50   = np.percentile(paths, 50, axis=0)
        p75   = np.percentile(paths, 75, axis=0)
        p95   = np.percentile(paths, 95, axis=0)

        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor(MP["bg"])
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                left=0.07, right=0.97,
                                top=0.91, bottom=0.09,
                                wspace=0.34, hspace=0.42)

        # ── Panel A: GBM performance projection (spans 2 columns) ─────────────
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor(MP["panel"])

        for i in range(min(60, N_SIM)):
            ax1.plot(t_arr, paths[i], color=MP["accent"], alpha=0.04, lw=0.5)
        ax1.fill_between(t_arr, p5,  p95, color=MP["accent"], alpha=0.10, label="5–95th pct")
        ax1.fill_between(t_arr, p25, p75, color=MP["accent"], alpha=0.20, label="25–75th pct")
        ax1.plot(t_arr, p50, color=MP["accent2"], lw=2.2,
                 label=f"Median  €{p50[-1]:,.0f}")
        ax1.plot(t_arr, p5,  color=MP["danger"],  lw=1.1, ls="--",
                 label=f"5th pct  €{p5[-1]:,.0f}")
        ax1.plot(t_arr, p95, color=MP["pos"],     lw=1.1, ls="--",
                 label=f"95th pct €{p95[-1]:,.0f}")
        ax1.axhline(initial_val, color=MP["sub"], lw=0.8, ls=":")

        # VaR loss boundary
        var_line = initial_val * (1 - abs(var_horiz))
        ax1.axhline(var_line, color=MP["danger"], lw=1.0, ls="-.",
                    label=f"VaR floor  €{var_line:,.0f}")

        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax1.set_xlabel("Trading Days")
        ax1.set_ylabel("Portfolio Value (€)")
        ax1.set_title("Est. Performance Projection (GBM)", fontweight="semibold")
        ax1.legend(loc="upper left", fontsize=6.5)
        _tag(ax1, "PROJECTION")

        # ── Panel B: Top asset weights donut ─────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor(MP["panel"])
        n_top = min(8, len(names))
        idx_top = np.argsort(weights)[::-1][:n_top]
        w_top   = [weights[i] for i in idx_top]
        n_top_names = [names[i][:6] for i in idx_top]
        other   = max(0, 1 - sum(w_top))
        if other > 0.001:
            w_top.append(other); n_top_names.append("Other")
        cmap_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(w_top)))
        wedges, texts, autotexts = ax2.pie(
            w_top, labels=n_top_names, autopct="%1.1f%%",
            colors=cmap_colors, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=MP["bg"], linewidth=1.2),
            textprops=dict(fontsize=6, color=MP["text"]))
        for at in autotexts:
            at.set_fontsize(5.5); at.set_color(MP["bg"])
        ax2.set_title("Weight Distribution (top assets)", fontweight="semibold")
        _tag(ax2, "WEIGHTS")

        # ── Panel C: Rolling 21-day volatility ──────────────────────────────
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.set_facecolor(MP["panel"])
        port_rets_series = df_close.pct_change().dropna() @ weights
        roll_vol = port_rets_series.rolling(21).std() * np.sqrt(TD)
        ax3.plot(port_rets_series.index, roll_vol,
                 color=MP["accent3"], lw=1.4, label="21-day rolling vol (ann.)")
        ax3.axhline(est_vol_ann, color=MP["accent"], lw=1.0, ls="--",
                    label=f"Model est. vol {est_vol_ann:.1%}")
        ax3.axhline(cur_vol_ann, color=MP["accent2"], lw=1.0, ls=":",
                    label=f"Current vol {cur_vol_ann:.1%}")
        ax3.fill_between(port_rets_series.index, roll_vol, est_vol_ann,
                         where=(roll_vol > est_vol_ann),
                         color=MP["danger"], alpha=0.12, label="Above model est.")
        ax3.fill_between(port_rets_series.index, roll_vol, est_vol_ann,
                         where=(roll_vol <= est_vol_ann),
                         color=MP["pos"], alpha=0.10, label="Below model est.")
        _date_axis(ax3)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax3.set_title("Rolling 21-Day Realised Volatility vs Model Estimate",
                       fontweight="semibold")
        ax3.legend(loc="upper left", fontsize=6.5)
        _tag(ax3, "VOLATILITY")

        # ── Panel D: Risk scorecard ─────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(MP["panel"])
        ax4.axis("off")

        metrics = [
            ("Est. Performance",   f"{'+' if est_perf_pct>=0 else ''}{est_perf_pct:.2%}",
             MP["accent2"] if est_perf_pct >= 0 else MP["danger"]),
            ("Est. Volatility",    f"{est_vol_ann:.2%}",     MP["accent"]),
            ("Current Volatility", f"{cur_vol_ann:.2%}",     MP["accent3"]),
            ("VaR (horizon)",      f"{abs(var_horiz):.2%}",  MP["danger"]),
            ("Sharpe Ratio",       f"{sharpe_ann:+.2f}",
             MP["pos"] if sharpe_ann >= 0 else MP["danger"]),
            ("Risk Score",         f"{risk_score:.1f}/10",   risk_color),
            ("Risk Level",         risk_label,                risk_color),
        ]

        y = 0.96
        ax4.text(0.5, y, "Risk Summary", ha="center", va="top",
                 fontsize=9, fontweight="bold", color=MP["text"],
                 transform=ax4.transAxes)
        y -= 0.10
        ax4.plot([0.05, 0.95], [y, y], color=MP["border"],
                 lw=0.8, transform=ax4.transAxes)
        y -= 0.04

        for label, value, color in metrics:
            ax4.text(0.05, y, label, ha="left", va="top",
                     fontsize=7.5, color=MP["sub"], transform=ax4.transAxes)
            ax4.text(0.95, y, value, ha="right", va="top",
                     fontsize=8, fontweight="bold", color=color,
                     transform=ax4.transAxes)
            y -= 0.115
            ax4.plot([0.05, 0.95], [y + 0.04, y + 0.04],
                     color=MP["grid"], lw=0.4, transform=ax4.transAxes)

        # Risk gauge bar inside the scorecard
        ax4.text(0.05, y - 0.02, "Risk Gauge", ha="left", va="top",
                 fontsize=7, color=MP["sub"], transform=ax4.transAxes)
        bar_y  = y - 0.12
        bar_h  = 0.06
        # Background bar
        bar_bg = plt.Rectangle((0.05, bar_y), 0.90, bar_h,
                                transform=ax4.transAxes,
                                facecolor=MP["muted"], edgecolor=MP["border"],
                                linewidth=0.5, clip_on=False)
        ax4.add_patch(bar_bg)
        # Filled portion
        bar_fill = plt.Rectangle((0.05, bar_y), 0.90 * (risk_score / 10), bar_h,
                                  transform=ax4.transAxes,
                                  facecolor=risk_color, alpha=0.85,
                                  clip_on=False)
        ax4.add_patch(bar_fill)
        _tag(ax4, "SCORECARD")

        fig.suptitle("Portfolio Performance Dashboard", color=MP["text"],
                     fontsize=13, fontweight="bold", y=0.99)

        # ── Embed in the dashboard chart frame ────────────────────────────────
        for w in self._dash_chart_frame.winfo_children():
            w.destroy()
        _mpl_canvas(self._dash_chart_frame, fig)

    # ══════════════════════════════════════════════════════════════════════════
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
