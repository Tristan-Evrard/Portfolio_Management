import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np

from Financial_Calculation import *


# ── Global Style ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":        "#0D1117",
    "panel":     "#161B22",
    "border":    "#21262D",
    "text":      "#E6EDF3",
    "subtext":   "#8B949E",
    "accent":    "#58A6FF",
    "accent2":   "#3FB950",
    "accent3":   "#D29922",
    "danger":    "#F85149",
    "positive":  "#56D364",
    "sma1":      "#79C0FF",
    "sma2":      "#FFA657",
    "ema1":      "#D2A8FF",
    "ema2":      "#FF7B72",
    "grid":      "#1C2128",
}

def apply_global_style():
    plt.rcParams.update({
        "figure.facecolor":     PALETTE["bg"],
        "axes.facecolor":       PALETTE["panel"],
        "axes.edgecolor":       PALETTE["border"],
        "axes.labelcolor":      PALETTE["subtext"],
        "axes.titlecolor":      PALETTE["text"],
        "axes.titlesize":       11,
        "axes.titleweight":     "semibold",
        "axes.titlepad":        12,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.grid":            True,
        "grid.color":           PALETTE["grid"],
        "grid.linewidth":       0.6,
        "grid.alpha":           1.0,
        "xtick.color":          PALETTE["subtext"],
        "ytick.color":          PALETTE["subtext"],
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.facecolor":     PALETTE["panel"],
        "legend.edgecolor":     PALETTE["border"],
        "legend.labelcolor":    PALETTE["text"],
        "legend.fontsize":      8,
        "legend.framealpha":    0.9,
        "lines.linewidth":      1.5,
        "text.color":           PALETTE["text"],
        "font.family":          "monospace",
    })


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_pct(x, _):
    return f"{x:.2%}"

def _format_currency(x, _):
    return f"€{x:,.0f}"

def _tag(ax, label):
    """Small pill-shaped tag in the top-left corner of an axes."""
    ax.text(
        0.01, 0.98, f" {label} ",
        transform=ax.transAxes,
        fontsize=7, fontweight="bold",
        color=PALETTE["bg"],
        backgroundcolor=PALETTE["accent"],
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=PALETTE["accent"],
            edgecolor="none",
            alpha=0.85,
        ),
    )

def _style_date_axis(ax):
    locator   = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    formatter.scaled[1/(24.0)] = "%d %b"
    formatter.scaled[1.0]      = "%d %b"
    formatter.scaled[30.]      = "%b '%y"
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")


# ── Plot functions ────────────────────────────────────────────────────────────

class Data_Visualization:

    def __init__(self, DataFrame, DataFrame_Amount_Each_Value, **kwarg):
        self.DataFrame = DataFrame
        self.DataFrame_Amount_Each_Value = DataFrame_Amount_Each_Value
        self.indicators = kwarg

    # ── VaR ──────────────────────────────────────────────────────────────────
    def Portfolio_Value_At_Risk_plot(self, DataFrame, Confidence_Level, ax):
        VaR = round(Calc_VaR_Single(DataFrame["Portfolio"], Confidence_Level), 5)
        returns = Normalize_Distribution_Single(DataFrame["Portfolio"])

        n, bins, patches = ax.hist(
            returns, bins="auto",
            color=PALETTE["accent"], alpha=0.35, edgecolor="none",
        )
        # Colour bars beyond VaR in red
        for patch, left in zip(patches, bins[:-1]):
            if left < VaR:
                patch.set_facecolor(PALETTE["danger"])
                patch.set_alpha(0.55)

        ax.axvline(VaR, color=PALETTE["danger"], linewidth=1.8, ls="--", zorder=5)
        ax.text(
            VaR, ax.get_ylim()[1] * 0.92,
            f"  VaR {Confidence_Level:.0%}\n  {VaR:.4%}",
            color=PALETTE["danger"], fontsize=8, fontweight="bold",
        )

        ax.xaxis.set_major_formatter(FuncFormatter(_format_pct))
        ax.set_title("Value at Risk — Historical Simulation")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        _tag(ax, f"VaR {Confidence_Level:.0%}")

    # ── SMA ──────────────────────────────────────────────────────────────────
    def Portfolio_SMA_plot(self, DataFrame, periods, ax):
        sma_colors = [PALETTE["sma1"], PALETTE["sma2"], PALETTE["accent3"]]

        ax.plot(
            DataFrame.index, DataFrame["Portfolio"],
            color=PALETTE["accent"], linewidth=1.4, alpha=0.85, label="Portfolio",
        )

        if isinstance(periods, (int, float)):
            periods = [periods]

        for idx, p in enumerate(periods):
            col = f"Portfolio-SMA{p}"
            DataFrame[col] = SMA(DataFrame["Portfolio"], p)
            color = sma_colors[idx % len(sma_colors)]
            ax.plot(
                DataFrame.index, DataFrame[col],
                color=color, linewidth=1.2, ls="--", alpha=0.9, label=f"SMA {int(p)}",
            )

        _style_date_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_currency))
        ax.set_title("Simple Moving Average")
        ax.legend(loc="upper left")
        _tag(ax, "SMA")

    # ── EMA ──────────────────────────────────────────────────────────────────
    def Portfolio_EMA_plot(self, DataFrame, periods, ax):
        ema_colors = [PALETTE["ema1"], PALETTE["ema2"], PALETTE["accent2"]]

        ax.plot(
            DataFrame.index, DataFrame["Portfolio"],
            color=PALETTE["accent"], linewidth=1.4, alpha=0.85, label="Portfolio",
        )

        if isinstance(periods, (int, float)):
            periods = [periods]

        for idx, p in enumerate(periods):
            col = f"Portfolio-EMA{p}"
            DataFrame[col] = EMA(DataFrame["Portfolio"], p)
            color = ema_colors[idx % len(ema_colors)]
            ax.plot(
                DataFrame.index, DataFrame[col],
                color=color, linewidth=1.2, ls="--", alpha=0.9, label=f"EMA {int(p)}",
            )

        _style_date_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_currency))
        ax.set_title("Exponential Moving Average")
        ax.legend(loc="upper left")
        _tag(ax, "EMA")

    # ── RSI ──────────────────────────────────────────────────────────────────
    def Portfolio_RSI_plot(self, DataFrame, ax):
        RSI_portfolio(DataFrame)
        rsi_data = DataFrame["RSI14"].iloc[14:]
        idx_data = DataFrame.index[14:]

        # Gradient fill between overbought/oversold zones
        ax.fill_between(idx_data, rsi_data, 70,
                         where=(rsi_data >= 70),
                         color=PALETTE["danger"], alpha=0.20, interpolate=True)
        ax.fill_between(idx_data, rsi_data, 30,
                         where=(rsi_data <= 30),
                         color=PALETTE["positive"], alpha=0.20, interpolate=True)

        ax.plot(idx_data, rsi_data,
                color=PALETTE["accent3"], linewidth=1.4, label="RSI 14")
        ax.axhline(70, color=PALETTE["danger"],  linewidth=0.9, ls="--", alpha=0.7, label="Overbought (70)")
        ax.axhline(30, color=PALETTE["positive"], linewidth=0.9, ls="--", alpha=0.7, label="Oversold (30)")
        ax.axhline(50, color=PALETTE["subtext"],  linewidth=0.6, ls=":",  alpha=0.5)
        ax.set_ylim(0, 100)

        _style_date_axis(ax)
        ax.set_title("Relative Strength Index")
        ax.legend(loc="upper left")
        _tag(ax, "RSI 14")

    # ── Distribution table ───────────────────────────────────────────────────
    def Portfolio_distribution(self, DataFrame_Amount_Each_Value, ax):
        ax.set_facecolor(PALETTE["panel"])
        ax.axis("off")

        col_labels = DataFrame_Amount_Each_Value.columns.tolist()
        row_labels = DataFrame_Amount_Each_Value.index.tolist()
        cell_data  = DataFrame_Amount_Each_Value.values

        table = ax.table(
            cellText=cell_data,
            colLabels=col_labels,
            rowLabels=row_labels,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.3, 1.6)

        # Style every cell
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(PALETTE["border"])
            cell.set_linewidth(0.5)
            if row == 0:  # Header row
                cell.set_facecolor(PALETTE["accent"])
                cell.set_text_props(color=PALETTE["bg"], fontweight="bold")
            elif col == -1:  # Row labels
                cell.set_facecolor(PALETTE["border"])
                cell.set_text_props(color=PALETTE["subtext"], fontstyle="italic")
            else:
                cell.set_facecolor(PALETTE["panel"] if row % 2 == 0 else PALETTE["bg"])
                cell.set_text_props(color=PALETTE["text"])

        ax.set_title("Portfolio Distribution", pad=14)
        _tag(ax, "ALLOC")

    # ── Main plot ─────────────────────────────────────────────────────────────
    def plot_graph(self):
        apply_global_style()

        self.DataFrame.index = pd.to_datetime(self.DataFrame.index, format="%m/%d/%Y")

        active = {k: v for k, v in self.indicators.items() if v is not None and v != "n"}
        nb = len(active)

        # Determine grid size
        if nb <= 1:
            rows, cols = 1, 1
        elif nb == 2:
            rows, cols = 1, 2
        elif nb <= 4:
            rows, cols = 2, 2
        else:
            cols = 3
            rows = (nb + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 7, rows * 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        # ── Header banner ────────────────────────────────────────────────────
        fig.text(
            0.02, 0.98,
            "PORTFOLIO ANALYTICS",
            fontsize=15, fontweight="bold",
            color=PALETTE["text"],
            va="top",
        )
        latest_val = self.DataFrame["Portfolio"].iloc[-1]
        first_val  = self.DataFrame["Portfolio"].iloc[0]
        perf       = (latest_val - first_val) / first_val
        perf_color = PALETTE["positive"] if perf >= 0 else PALETTE["danger"]
        perf_sign  = "▲" if perf >= 0 else "▼"
        fig.text(
            0.02, 0.944,
            f"Portfolio value: €{latest_val:,.2f}   {perf_sign} {abs(perf):.2%} overall",
            fontsize=9, color=perf_color, va="top",
        )

        gs = gridspec.GridSpec(rows, cols, figure=fig,
                               hspace=0.45, wspace=0.30,
                               left=0.06, right=0.97,
                               top=0.88, bottom=0.08)

        axes_flat = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]
        ax_idx = 0

        if self.indicators.get("Confidence_level") not in (None, "n"):
            self.Portfolio_Value_At_Risk_plot(
                self.DataFrame, self.indicators["Confidence_level"], axes_flat[ax_idx]
            )
            ax_idx += 1

        if self.indicators.get("periods_SMA") not in (None, "n"):
            self.Portfolio_SMA_plot(
                self.DataFrame, self.indicators["periods_SMA"], axes_flat[ax_idx]
            )
            ax_idx += 1

        if self.indicators.get("periods_EMA") not in (None, "n"):
            self.Portfolio_EMA_plot(
                self.DataFrame, self.indicators["periods_EMA"], axes_flat[ax_idx]
            )
            ax_idx += 1

        if self.indicators.get("RSI") == "y":
            self.Portfolio_RSI_plot(self.DataFrame, axes_flat[ax_idx])
            ax_idx += 1

        if self.indicators.get("Portfolio_Distribution") == "y":
            self.Portfolio_distribution(self.DataFrame_Amount_Each_Value, axes_flat[ax_idx])
            ax_idx += 1

        # Hide unused subplots
        for ax in axes_flat[ax_idx:]:
            ax.set_visible(False)

        plt.savefig("portfolio_output.png", dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        plt.show()