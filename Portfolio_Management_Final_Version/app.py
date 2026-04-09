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
    "pos": "#34D399", "grid": "#1C2535",
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


#FINANCIAL CALCULATION

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

def markowitz_optimize(df_close, total):
    n   = df_close.shape[1]
    ret = df_close.pct_change().dropna()
    cov = ret.cov().values
    try:
        import cvxpy as cp
        w   = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, cov))
        prob= cp.Problem(obj, [cp.sum(w)<=1, w>=0.01, w<=0.10])
        prob.solve()
        return (np.round(w.value, 4) * total).astype(float).tolist()
    except Exception:
        # fallback : min-variance scipy
        mu  = ret.mean().values
        wv  = min_variance(mu, cov, n)
        return (wv * total).tolist()


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
        self.v_sma_p     = tk.StringVar(value="12, 26")
        self.v_ema_p     = tk.StringVar(value="12")

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

        # ── Run button ────────────────────────────────────────────────────────
        run_row = tk.Frame(pad, bg=C["bg"])
        run_row.pack(anchor="w", pady=(20,0))
        self._run_btn = tk.Button(
            run_row, text="🚀  Run Analysis", font=("Courier New",11,"bold"),
            fg=C["bg"], bg=C["accent2"], relief="flat",
            padx=24, pady=10, cursor="hand2", command=self._run)
        self._run_btn.pack(side="left")

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
    def _run(self):
        if not os.path.isfile(self.v_path.get()):
            messagebox.showerror("Missing file", "Please select a valid CSV file.")
            return
        try: float(self.v_amount.get())
        except ValueError:
            messagebox.showerror("Invalid amount", "Investment must be a number."); return

        self._run_btn.config(state="disabled")
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
                amounts = markowitz_optimize(df_close, total)
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

            # 4. Render charts
            self._set_status("Rendering charts…")
            conf = float(self.v_conf.get())

            if self.v_var.get():
                self.root.after(0, lambda: self._plot_var(df_portfolio, conf))

            if self.v_sma.get():
                periods_sma = [int(x.strip()) for x in self.v_sma_p.get().split(",")
                                if x.strip().isdigit()]
                self.root.after(0, lambda p=periods_sma: self._plot_sma(df_portfolio, p))

            if self.v_ema.get():
                raw_ema = self.v_ema_p.get().strip()
                period_ema = int(raw_ema) if raw_ema.isdigit() else 12
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
                self.root.after(0, lambda: self._plot_frontier(
                    mc_r, mc_v, mc_s, fr_v, fr_r, w_arr, mu, cov, names))

            self.root.after(0, self._done)

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

    def _done(self):
        self._pbar.stop()
        self._pbar.pack_forget()
        self._run_btn.config(state="normal")
        self._set_status("✔ Analysis complete")
        # Auto-navigate to first enabled chart
        for key in ["var","sma","ema","rsi","distrib","frontier"]:
            v = {"var":self.v_var,"sma":self.v_sma,"ema":self.v_ema,
                 "rsi":self.v_rsi,"distrib":self.v_distrib,"frontier":self.v_frontier}
            if v[key].get():
                self._show_panel(key); break

    def _on_error(self, msg):
        self._pbar.stop(); self._pbar.pack_forget()
        self._run_btn.config(state="normal")
        self._set_status(f"✗ Error")
        messagebox.showerror("Error", msg)

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

        # ── DataFrame────────────────────────────────────────────────────
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
        mr,mv,_  = portfolio_perf(wm, mu, cov)
        pr,pv,ps = portfolio_perf(weights, mu, cov)
        p_color  = MP["pos"] if ps >= ss*0.75 else MP["danger"]

        for (vol,ret,lbl,color,marker,sz) in [
            (sv,sr,"Max Sharpe",  MP["accent3"],"*", 200),
            (mv,mr,"Min Variance",MP["accent2"],"D", 80),
            (pv,pr,"Your Portfolio",p_color,   "P", 160),
        ]:
            ax.scatter(vol, ret, marker=marker, s=sz, color=color,
                       zorder=6, label=f"{lbl}  ({ps if lbl=='Your Portfolio' else ss if lbl=='Max Sharpe' else mv:.2f})")
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

    # ══════════════════════════════════════════════════════════════════════════
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
