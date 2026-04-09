import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings("ignore")

import config


# ── Palette (identique à Data_Visualization.py) ───────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "border":   "#21262D",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
    "accent":   "#58A6FF",
    "accent2":  "#3FB950",
    "accent3":  "#D29922",
    "danger":   "#F85149",
    "positive": "#56D364",
    "grid":     "#1C2128",
}

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["subtext"],
        "axes.titlecolor":   PALETTE["text"],
        "axes.titlesize":    11,
        "axes.titleweight":  "semibold",
        "axes.titlepad":     12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["border"],
        "legend.labelcolor": PALETTE["text"],
        "legend.fontsize":   8,
        "lines.linewidth":   1.5,
        "text.color":        PALETTE["text"],
        "font.family":       "monospace",
    })

def _fmt_pct(x, _):
    return f"{x:.1%}"

def _tag(ax, label):
    ax.text(
        0.01, 0.98, f" {label} ",
        transform=ax.transAxes,
        fontsize=7, fontweight="bold",
        color=PALETTE["bg"], va="top",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor=PALETTE["accent"],
                  edgecolor="none", alpha=0.85),
    )


# ── Calculs Markowitz ─────────────────────────────────────────────────────────

def _portfolio_perf(weights, mean_returns, cov_matrix, trading_days=252):
    ret    = np.dot(weights, mean_returns) * trading_days
    vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days)
    sharpe = ret / vol if vol != 0 else 0.0
    return ret, vol, sharpe


def _min_variance_for_target(target_return, mean_returns, cov_matrix, n):
    def portfolio_vol(w):
        return np.sqrt(w @ cov_matrix @ w)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},
    ]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    result = minimize(portfolio_vol, np.ones(n) / n,
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})
    return result if result.success else None


def _max_sharpe(mean_returns, cov_matrix, n, rf=0.0):
    def neg_sharpe(w):
        r, v, _ = _portfolio_perf(w, mean_returns, cov_matrix)
        return -(r - rf) / v if v != 0 else 0.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    result = minimize(neg_sharpe, np.ones(n) / n,
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})
    return result.x if result.success else np.ones(n) / n


def _min_global_variance(mean_returns, cov_matrix, n):
    def portfolio_vol(w):
        return np.sqrt(w @ cov_matrix @ w)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    result = minimize(portfolio_vol, np.ones(n) / n,
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})
    return result.x if result.success else np.ones(n) / n


# ── Fenêtre déroulante Tkinter ────────────────────────────────────────────────

def _show_scrollable_table(asset_names, weights, p_color_hex):
    """Fenêtre Tkinter avec tableau déroulant, filtrable et trié par poids."""

    root = tk.Tk()
    root.title("Portfolio — Asset Weights")
    root.configure(bg=PALETTE["bg"])
    root.geometry("500x540")
    root.resizable(True, True)

    # ── En-tête ───────────────────────────────────────────────────────────────
    header = tk.Frame(root, bg=PALETTE["bg"])
    header.pack(fill="x", padx=16, pady=(14, 0))

    tk.Label(header, text="PORTFOLIO ANALYTICS",
             font=("Courier", 13, "bold"),
             fg=PALETTE["text"], bg=PALETTE["bg"]).pack(side="left")
    tk.Label(header, text=f"  {len(asset_names)} assets",
             font=("Courier", 9),
             fg=PALETTE["subtext"], bg=PALETTE["bg"]).pack(side="left", padx=6)

    tk.Label(root, text="Asset Weights — Your Portfolio",
             font=("Courier", 10, "bold"),
             fg=PALETTE["accent"], bg=PALETTE["bg"]).pack(anchor="w", padx=16, pady=(6, 2))

    tk.Frame(root, bg=PALETTE["border"], height=1).pack(fill="x", padx=16, pady=(0, 6))

    # ── Barre de recherche ────────────────────────────────────────────────────
    search_frame = tk.Frame(root, bg=PALETTE["panel"],
                             highlightbackground=PALETTE["border"],
                             highlightthickness=1)
    search_frame.pack(fill="x", padx=16, pady=(0, 8))

    tk.Label(search_frame, text="🔍", bg=PALETTE["panel"],
             fg=PALETTE["subtext"], font=("Courier", 10)).pack(side="left", padx=(8, 4))

    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var,
                             bg=PALETTE["panel"], fg=PALETTE["subtext"],
                             insertbackground=PALETTE["text"],
                             relief="flat", font=("Courier", 9), bd=0)
    search_entry.pack(side="left", fill="x", expand=True, pady=6, padx=(0, 8))
    search_entry.insert(0, "Filter asset...")

    def _focus_in(e):
        if search_entry.get() == "Filter asset...":
            search_entry.delete(0, "end")
            search_entry.config(fg=PALETTE["text"])

    def _focus_out(e):
        if search_entry.get() == "":
            search_entry.insert(0, "Filter asset...")
            search_entry.config(fg=PALETTE["subtext"])

    search_entry.bind("<FocusIn>",  _focus_in)
    search_entry.bind("<FocusOut>", _focus_out)

    # ── Tableau ───────────────────────────────────────────────────────────────
    table_frame = tk.Frame(root, bg=PALETTE["bg"])
    table_frame.pack(fill="both", expand=True, padx=16)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.Treeview",
                     background=PALETTE["panel"],
                     foreground=PALETTE["text"],
                     fieldbackground=PALETTE["panel"],
                     rowheight=26,
                     font=("Courier", 9),
                     borderwidth=0)
    style.configure("Dark.Treeview.Heading",
                     background=PALETTE["accent"],
                     foreground=PALETTE["bg"],
                     font=("Courier", 9, "bold"),
                     relief="flat")
    style.map("Dark.Treeview",
              background=[("selected", PALETTE["border"])],
              foreground=[("selected", PALETTE["accent"])])

    columns = ("rank", "asset", "weight", "bar")
    tree = ttk.Treeview(table_frame, columns=columns,
                         show="headings", style="Dark.Treeview",
                         selectmode="browse")

    tree.heading("rank",   text="#",            anchor="center")
    tree.heading("asset",  text="Asset",         anchor="w")
    tree.heading("weight", text="Weight",        anchor="center")
    tree.heading("bar",    text="Distribution",  anchor="w")

    tree.column("rank",   width=36,  anchor="center", stretch=False)
    tree.column("asset",  width=130, anchor="w",      stretch=True)
    tree.column("weight", width=70,  anchor="center", stretch=False)
    tree.column("bar",    width=180, anchor="w",      stretch=True)

    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    tree.pack(side="left", fill="both", expand=True)

    # Données pré-triées
    sorted_idx     = np.argsort(weights)[::-1]
    names_sorted   = [asset_names[i] for i in sorted_idx]
    weights_sorted = [weights[i]     for i in sorted_idx]
    max_w          = max(weights_sorted) if weights_sorted else 1.0

    all_rows = []
    for rank, (name, w) in enumerate(zip(names_sorted, weights_sorted), start=1):
        bar_len = int((w / max_w) * 22)
        bar_str = "█" * bar_len + "░" * (22 - bar_len)
        all_rows.append((str(rank), name, f"{w:.2%}", bar_str))

    def _populate(filter_text=""):
        for item in tree.get_children():
            tree.delete(item)
        ft = filter_text.lower().strip()
        for i, row in enumerate(all_rows):
            if ft and ft not in row[1].lower():
                continue
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end", values=row, tags=(tag,))
        tree.tag_configure("even", background=PALETTE["panel"])
        tree.tag_configure("odd",  background=PALETTE["bg"])

    _populate()

    def _on_search(*_):
        q = search_var.get()
        _populate("" if q == "Filter asset..." else q)

    search_var.trace_add("write", _on_search)

    # ── Pied de page ──────────────────────────────────────────────────────────
    tk.Frame(root, bg=PALETTE["border"], height=1).pack(fill="x", padx=16, pady=(6, 0))
    footer = tk.Frame(root, bg=PALETTE["bg"])
    footer.pack(fill="x", padx=16, pady=(4, 10))

    tk.Label(footer,
             text=f"Total: {sum(weights_sorted):.2%}  ·  {len(weights_sorted)} assets",
             font=("Courier", 8), fg=PALETTE["subtext"], bg=PALETTE["bg"]).pack(side="left")

    tk.Button(footer, text="✕  Close", command=root.destroy,
              bg=PALETTE["danger"], fg=PALETTE["bg"],
              font=("Courier", 8, "bold"),
              relief="flat", padx=10, pady=3, cursor="hand2").pack(side="right")

    root.mainloop()


# ── Classe principale ─────────────────────────────────────────────────────────

class Efficient_Frontier:
    """
    Trace la frontière efficiente de Markowitz et positionne
    le portefeuille courant sur le graphique.

    Paramètres
    ----------
    df_close          : pd.DataFrame  — prix de clôture (colonnes = actifs)
    portfolio_weights : list/ndarray  — poids du portefeuille (somme = 1)
    n_simulations     : int           — portfolios Monte-Carlo (défaut 4000)
    trading_days      : int           — jours/an pour annualisation (défaut 252)
    rf                : float         — taux sans risque annualisé (défaut 0)
    """

    def __init__(self,
                 df_close: pd.DataFrame,
                 portfolio_weights,
                 n_simulations: int = 4000,
                 trading_days:  int = 252,
                 rf:            float = 0.0):

        self.df_close      = df_close
        self.trading_days  = trading_days
        self.rf            = rf
        self.n_simulations = n_simulations
        self.asset_names   = [c.replace("_Close", "") for c in df_close.columns]
        self.n             = len(self.asset_names)

        self.returns    = df_close.pct_change().dropna()
        self.mean_ret   = self.returns.mean().values
        self.cov_matrix = self.returns.cov().values

        w = np.array(portfolio_weights, dtype=float)
        self.portfolio_weights = w / w.sum()

    # ── Monte-Carlo avec barre de progression ─────────────────────────────────
    def _simulate(self):
        rets, vols, sharpes = [], [], []
        for _ in tqdm(range(self.n_simulations),
                      desc="  Monte-Carlo   ",
                      bar_format="  {l_bar}{bar:35}{r_bar}",
                      colour="blue",
                      ncols=72):
            w = np.random.dirichlet(np.ones(self.n))
            r, v, s = _portfolio_perf(w, self.mean_ret, self.cov_matrix,
                                       self.trading_days)
            rets.append(r)
            vols.append(v)
            sharpes.append(s)
        return np.array(rets), np.array(vols), np.array(sharpes)

    # ── Frontière efficiente avec barre de progression ────────────────────────
    def _compute_frontier(self, n_points=120):
        r_min_w = _min_global_variance(self.mean_ret, self.cov_matrix, self.n)
        r_min, _, _ = _portfolio_perf(r_min_w, self.mean_ret,
                                       self.cov_matrix, self.trading_days)
        r_max   = np.max(self.mean_ret) * self.trading_days
        targets = np.linspace(r_min, r_max * 0.98, n_points)

        frontier_vols, frontier_rets = [], []
        for t in tqdm(targets,
                      desc="  Frontier      ",
                      bar_format="  {l_bar}{bar:35}{r_bar}",
                      colour="cyan",
                      ncols=72):
            res = _min_variance_for_target(
                t / self.trading_days, self.mean_ret, self.cov_matrix, self.n)
            if res is not None:
                v = np.sqrt(res.x @ self.cov_matrix @ res.x) * np.sqrt(self.trading_days)
                frontier_vols.append(v)
                frontier_rets.append(t)

        return np.array(frontier_vols), np.array(frontier_rets)

    # ── Plot ──────────────────────────────────────────────────────────────────
    def plot(self):
        _apply_style()

        print("\n╔══════════════════════════════════════════╗")
        print("║      MARKOWITZ EFFICIENT FRONTIER        ║")
        print(f"║  {self.n} assets  ·  {self.n_simulations} simulations{' ' * max(0, 11 - len(str(self.n_simulations)))}   ║")
        print("╚══════════════════════════════════════════╝\n")

        # Calculs
        mc_rets, mc_vols, mc_sharpes = self._simulate()
        frontier_vols, frontier_rets = self._compute_frontier()

        p_ret, p_vol, p_sharpe = _portfolio_perf(
            self.portfolio_weights, self.mean_ret, self.cov_matrix, self.trading_days)

        w_sharpe = _max_sharpe(self.mean_ret, self.cov_matrix, self.n, self.rf)
        s_ret, s_vol, s_sharpe = _portfolio_perf(
            w_sharpe, self.mean_ret, self.cov_matrix, self.trading_days)

        w_mvp = _min_global_variance(self.mean_ret, self.cov_matrix, self.n)
        m_ret, m_vol, _ = _portfolio_perf(
            w_mvp, self.mean_ret, self.cov_matrix, self.trading_days)

        p_color = PALETTE["positive"] if p_sharpe >= s_sharpe * 0.75 else PALETTE["danger"]

        print(f"\n  ✔  Computation complete")
        print(f"     Your portfolio  →  Return {p_ret:.2%}  |  Vol {p_vol:.2%}  |  Sharpe {p_sharpe:.3f}")
        print(f"     Max Sharpe      →  Return {s_ret:.2%}  |  Vol {s_vol:.2%}  |  Sharpe {s_sharpe:.3f}\n")

        # Figure
        fig = plt.figure(figsize=(13, 6.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        fig.text(0.02, 0.97, "MARKOWITZ EFFICIENT FRONTIER",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], va="top")
        fig.text(0.02, 0.925,
                 f"Monte-Carlo: {self.n_simulations:,} portfolios  ·  "
                 f"{self.n} assets  ·  Annualised ({self.trading_days}d)",
                 fontsize=8, color=PALETTE["subtext"], va="top")

        gs = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.06, right=0.97,
                               top=0.87, bottom=0.10, wspace=0.30)
        ax_main  = fig.add_subplot(gs[0, 0])
        ax_table = fig.add_subplot(gs[0, 1])

        # Scatter Monte-Carlo
        sc = ax_main.scatter(mc_vols, mc_rets, c=mc_sharpes, cmap="plasma",
                             alpha=0.35, s=6, linewidths=0, zorder=2)
        cbar = fig.colorbar(sc, ax=ax_main, pad=0.02, fraction=0.035)
        cbar.set_label("Sharpe Ratio", color=PALETTE["subtext"], fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"], fontsize=7)
        cbar.outline.set_edgecolor(PALETTE["border"])

        # Frontière
        if len(frontier_vols) > 2:
            ax_main.plot(frontier_vols, frontier_rets,
                         color=PALETTE["accent"], linewidth=2.2,
                         zorder=4, label="Efficient Frontier")

        # Max Sharpe
        ax_main.scatter(s_vol, s_ret, marker="*", s=220, color=PALETTE["accent3"],
                        zorder=6, label=f"Max Sharpe  ({s_sharpe:.2f})")
        ax_main.annotate(f" Max Sharpe\n {s_ret:.1%} / {s_vol:.1%}",
                         xy=(s_vol, s_ret), xytext=(s_vol + 0.005, s_ret + 0.005),
                         color=PALETTE["accent3"], fontsize=7,
                         arrowprops=dict(arrowstyle="->", color=PALETTE["accent3"], lw=0.8))

        # Min Variance
        ax_main.scatter(m_vol, m_ret, marker="D", s=80, color=PALETTE["accent2"],
                        zorder=6, label=f"Min Variance  ({m_vol:.1%})")
        ax_main.annotate(f" Min Variance\n {m_ret:.1%} / {m_vol:.1%}",
                         xy=(m_vol, m_ret), xytext=(m_vol + 0.005, m_ret - 0.015),
                         color=PALETTE["accent2"], fontsize=7,
                         arrowprops=dict(arrowstyle="->", color=PALETTE["accent2"], lw=0.8))

        # Portefeuille courant
        ax_main.scatter(p_vol, p_ret, marker="P", s=180, color=p_color,
                        zorder=7, edgecolors="white", linewidths=0.6,
                        label=f"Your Portfolio  ({p_sharpe:.2f})")
        ax_main.annotate(f" Your Portfolio\n {p_ret:.1%} / {p_vol:.1%}",
                         xy=(p_vol, p_ret), xytext=(p_vol + 0.005, p_ret + 0.010),
                         color=p_color, fontsize=7, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=p_color, lw=0.8))

        ax_main.set_xlabel("Annualised Volatility (Std)")
        ax_main.set_ylabel("Annualised Return")
        ax_main.set_title("Risk / Return Space")
        ax_main.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax_main.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax_main.legend(loc="lower right", fontsize=7.5)
        _tag(ax_main, "FRONTIER")

        # Table comparaison
        ax_table.set_facecolor(PALETTE["panel"])
        ax_table.axis("off")
        rows = [
            ["Your Portfolio", f"{p_ret:.2%}", f"{p_vol:.2%}", f"{p_sharpe:.3f}"],
            ["Max Sharpe",     f"{s_ret:.2%}", f"{s_vol:.2%}", f"{s_sharpe:.3f}"],
            ["Min Variance",   f"{m_ret:.2%}", f"{m_vol:.2%}", "—"],
        ]
        table = ax_table.table(cellText=rows,
                                colLabels=["Portfolio", "Return", "Volatility", "Sharpe"],
                                loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.4, 2.0)

        row_colors = [p_color, PALETTE["accent3"], PALETTE["accent2"]]
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(PALETTE["border"])
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor(PALETTE["accent"])
                cell.set_text_props(color=PALETTE["bg"], fontweight="bold")
            else:
                cell.set_facecolor(PALETTE["bg"] if r % 2 == 0 else PALETTE["panel"])
                cell.set_text_props(
                    color=row_colors[r - 1] if c == 0 else PALETTE["text"],
                    fontweight="bold" if c == 0 else "normal")

        ax_table.set_title("Performance Comparison", pad=14)

        # Bouton "View Asset Weights" dans la figure
        btn_ax = fig.add_axes([0.595, 0.10, 0.30, 0.075])
        for spine in btn_ax.spines.values():
            spine.set_visible(False)
        btn_ax.set_facecolor(PALETTE["accent"])
        btn_ax.set_xticks([])
        btn_ax.set_yticks([])
        btn_ax.text(0.5, 0.5, "☰  View Asset Weights",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=PALETTE["bg"],
                    transform=btn_ax.transAxes)

        btn_pos = btn_ax.get_position()

        def _on_click(event):
            if event.inaxes is None:
                return
            # Coordonnées normalisées dans la figure
            x_fig = event.x / (fig.get_size_inches()[0] * fig.dpi)
            y_fig = event.y / (fig.get_size_inches()[1] * fig.dpi)
            if (btn_pos.x0 <= x_fig <= btn_pos.x1 and
                    btn_pos.y0 <= y_fig <= btn_pos.y1):
                _show_scrollable_table(self.asset_names,
                                       self.portfolio_weights, p_color)

        fig.canvas.mpl_connect("button_press_event", _on_click)

        plt.savefig("efficient_frontier.png", dpi=150,
                    bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.show()

        # Tableau déroulant ouvert automatiquement après le graphique
        _show_scrollable_table(self.asset_names, self.portfolio_weights, p_color)
