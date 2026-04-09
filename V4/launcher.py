import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0A0E17",
    "panel":     "#111827",
    "card":      "#1A2332",
    "border":    "#1E3A5F",
    "accent":    "#0EA5E9",
    "accent2":   "#10B981",
    "accent3":   "#F59E0B",
    "danger":    "#EF4444",
    "text":      "#F1F5F9",
    "sub":       "#64748B",
    "muted":     "#334155",
    "highlight": "#0EA5E920",
}

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_STEP   = ("Courier New", 11, "bold")
FONT_LABEL  = ("Courier New", 9,  "bold")
FONT_BODY   = ("Courier New", 9)
FONT_MONO   = ("Courier New", 10)
FONT_BIG    = ("Courier New", 28, "bold")
FONT_SMALL  = ("Courier New", 7)


# ══════════════════════════════════════════════════════════════════════════════
#  Widgets helpers
# ══════════════════════════════════════════════════════════════════════════════

def styled_entry(parent, textvariable=None, width=28, placeholder="", **kw):
    frame = tk.Frame(parent, bg=C["card"], highlightbackground=C["border"],
                     highlightthickness=1)
    e = tk.Entry(frame, textvariable=textvariable, width=width,
                 bg=C["card"], fg=C["text"], insertbackground=C["accent"],
                 relief="flat", font=FONT_MONO,
                 disabledbackground=C["muted"], disabledforeground=C["sub"],
                 **kw)
    e.pack(padx=8, pady=6)

    if placeholder:
        def _in(ev):
            if e.get() == placeholder:
                e.delete(0, "end")
                e.config(fg=C["text"])
        def _out(ev):
            if e.get() == "":
                e.insert(0, placeholder)
                e.config(fg=C["sub"])
        e.insert(0, placeholder)
        e.config(fg=C["sub"])
        e.bind("<FocusIn>",  _in)
        e.bind("<FocusOut>", _out)

    def _hover_in(ev):  frame.config(highlightbackground=C["accent"])
    def _hover_out(ev): frame.config(highlightbackground=C["border"])
    frame.bind("<Enter>", _hover_in)
    frame.bind("<Leave>", _hover_out)
    e.bind("<Enter>",  _hover_in)
    e.bind("<Leave>",  _hover_out)

    return frame, e


def section_label(parent, text):
    tk.Label(parent, text=text, font=FONT_LABEL,
             fg=C["accent"], bg=C["panel"]).pack(anchor="w", pady=(10, 2))


def divider(parent):
    tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=8)


def pill_tag(parent, text, color):
    tk.Label(parent, text=f" {text} ", font=FONT_SMALL,
             fg=C["bg"], bg=color,
             padx=4, pady=1).pack(side="left", padx=3)


class ToggleButton(tk.Frame):
    """Push Button ON/OFF"""
    def __init__(self, parent, label, variable, command=None, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self._var = variable
        self._cmd = command
        self._label = label
        self._btn = tk.Label(self, font=FONT_LABEL, cursor="hand2",
                              padx=14, pady=7)
        self._btn.pack()
        self._btn.bind("<Button-1>", self._toggle)
        self._refresh()

    def _toggle(self, _=None):
        self._var.set(not self._var.get())
        self._refresh()
        if self._cmd:
            self._cmd()

    def _refresh(self):
        if self._var.get():
            self._btn.config(text=f"✓  {self._label}",
                              fg=C["bg"], bg=C["accent2"],
                              relief="flat")
            self.config(highlightbackground=C["accent2"], highlightthickness=1)
        else:
            self._btn.config(text=f"○  {self._label}",
                              fg=C["sub"], bg=C["muted"],
                              relief="flat")
            self.config(highlightbackground=C["muted"], highlightthickness=1)


class StepDot(tk.Canvas):
    """Progressions indicators on the top"""
    def __init__(self, parent, n_steps, **kw):
        super().__init__(parent, bg=C["bg"], highlightthickness=0,
                         height=24, **kw)
        self.n = n_steps
        self._current = 0
        self.bind("<Configure>", lambda e: self._draw())

    def set_step(self, step):
        self._current = step
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        if w < 10:
            return
        r = 6
        gap = 32
        total = self.n * (2 * r) + (self.n - 1) * gap
        x0 = (w - total) / 2
        y  = 12
        for i in range(self.n):
            cx = x0 + i * (2 * r + gap) + r
            if i < self._current:
                self.create_oval(cx-r, y-r, cx+r, y+r,
                                 fill=C["accent2"], outline="")
                self.create_text(cx, y, text="✓", fill=C["bg"],
                                 font=("Courier New", 7, "bold"))
            elif i == self._current:
                self.create_oval(cx-r-2, y-r-2, cx+r+2, y+r+2,
                                 outline=C["accent"], width=1, fill="")
                self.create_oval(cx-r, y-r, cx+r, y+r,
                                 fill=C["accent"], outline="")
                self.create_text(cx, y, text=str(i+1), fill=C["bg"],
                                 font=("Courier New", 7, "bold"))
            else:
                self.create_oval(cx-r, y-r, cx+r, y+r,
                                 fill=C["muted"], outline="")
                self.create_text(cx, y, text=str(i+1), fill=C["sub"],
                                 font=("Courier New", 7))
            # Ligne de connexion
            if i < self.n - 1:
                lx = cx + r + 1
                rx = x0 + (i+1) * (2*r + gap) - r - 1
                color = C["accent2"] if i < self._current else C["muted"]
                self.create_line(lx, y, rx, y, fill=color, width=1)


class AnimatedFrame(tk.Frame):
    """Slide Frame for transition"""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["panel"], **kw)

    def slide_in(self, from_right=True):
        w = self.winfo_reqwidth() or 600
        start_x = w if from_right else -w
        self.place(x=start_x, y=0, relwidth=1, relheight=1)
        self._animate_x(start_x, 0, 12)

    def slide_out(self, to_left=True):
        w = self.winfo_reqwidth() or 600
        end_x = -w if to_left else w
        self._animate_x(0, end_x, 12, on_done=self.place_forget)

    def _animate_x(self, start, end, steps, on_done=None, step=0):
        if step >= steps:
            self.place(x=end, y=0, relwidth=1, relheight=1)
            if on_done:
                on_done()
            return
        t = step / steps
        # Ease out cubic
        t2 = 1 - (1 - t) ** 3
        x = int(start + (end - start) * t2)
        self.place(x=x, y=0, relwidth=1, relheight=1)
        self.after(16, lambda: self._animate_x(start, end, steps, on_done, step+1))


# ══════════════════════════════════════════════════════════════════════════════
#  Wizard principal
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioWizard:
    STEPS = ["Portfolio Setup", "Indicators", "Review & Launch"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Portfolio Manager")
        self.root.geometry("680x600")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)
        self._center()

        # ── State variables ───────────────────────────────────────────────────
        self.path_var        = tk.StringVar()
        self.amount_var      = tk.StringVar(value="10000")
        self.dist_uniform    = tk.BooleanVar(value=False)
        self.optimize        = tk.BooleanVar(value=True)
        self.custom_amounts  = tk.StringVar()
        self.confidence_var  = tk.StringVar(value="0.95")

        self.show_var        = tk.BooleanVar(value=True)
        self.show_sma        = tk.BooleanVar(value=True)
        self.show_ema        = tk.BooleanVar(value=True)
        self.show_rsi        = tk.BooleanVar(value=True)
        self.show_frontier   = tk.BooleanVar(value=True)
        self.show_distrib    = tk.BooleanVar(value=True)

        self.sma_periods_var = tk.StringVar(value="12, 26")
        self.ema_periods_var = tk.StringVar(value="12")

        self.current_step = 0
        self.frames       = []

        # ── Layout racine ─────────────────────────────────────────────────────
        self._build_header()
        self.container = tk.Frame(self.root, bg=C["panel"])
        self.container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.container.pack_propagate(False)
        self._build_footer()

        #Building of every frames
        self.frames = [
            self._build_step0(),
            self._build_step1(),
            self._build_step2(),
        ]
        self._show_step(0, animate=False)

    def _center(self):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - 680) // 2
        y  = (sh - 600) // 2
        self.root.geometry(f"680x600+{x}+{y}")

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg"])
        hdr.pack(fill="x", padx=20, pady=(16, 8))

        left = tk.Frame(hdr, bg=C["bg"])
        left.pack(side="left")
        tk.Label(left, text="PORTFOLIO", font=FONT_TITLE,
                 fg=C["text"], bg=C["bg"]).pack(side="left")
        tk.Label(left, text=" MANAGER", font=FONT_TITLE,
                 fg=C["accent"], bg=C["bg"]).pack(side="left")

        # Dots de progression
        self.dots = StepDot(hdr, len(self.STEPS), width=200)
        self.dots.pack(side="right", pady=4)

    # ── Footer ────────────────────────────────────────────────────────────────
    def _build_footer(self):
        foot = tk.Frame(self.root, bg=C["bg"])
        foot.pack(fill="x", padx=20, pady=(0, 12))

        self.btn_back = tk.Button(
            foot, text="← Back", font=FONT_LABEL,
            fg=C["sub"], bg=C["muted"], activeforeground=C["text"],
            activebackground=C["border"], relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=self._go_back)
        self.btn_back.pack(side="left")

        self.step_label = tk.Label(foot, font=FONT_SMALL,
                                    fg=C["sub"], bg=C["bg"])
        self.step_label.pack(side="left", padx=16)

        self.btn_next = tk.Button(
            foot, text="Next  →", font=FONT_LABEL,
            fg=C["bg"], bg=C["accent"], activeforeground=C["bg"],
            activebackground="#38BDF8", relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=self._go_next)
        self.btn_next.pack(side="right")

    def _refresh_footer(self):
        step = self.current_step
        self.btn_back.config(state="normal" if step > 0 else "disabled",
                              fg=C["sub"] if step > 0 else C["muted"])
        if step == len(self.STEPS) - 1:
            self.btn_next.config(text="🚀  Launch", bg=C["accent2"])
        else:
            self.btn_next.config(text="Next  →", bg=C["accent"])
        self.step_label.config(
            text=f"Step {step+1} of {len(self.STEPS)}  ·  {self.STEPS[step]}")

    # ── Navigation ────────────────────────────────────────────────────────────
    def _show_step(self, step, animate=True, direction=1):
        for f in self.frames:
            f.place_forget()
        self.current_step = step
        self.dots.set_step(step)
        self._refresh_footer()
        frame = self.frames[step]
        if animate:
            frame.slide_in(from_right=(direction > 0))
        else:
            frame.place(x=0, y=0, relwidth=1, relheight=1)

    def _go_next(self):
        if not self._validate():
            return
        if self.current_step < len(self.STEPS) - 1:
            old = self.frames[self.current_step]
            old.slide_out(to_left=True)
            self.root.after(150, lambda: self._show_step(self.current_step + 1, direction=1))
        else:
            self._launch()

    def _go_back(self):
        if self.current_step > 0:
            old = self.frames[self.current_step]
            old.slide_out(to_left=False)
            self.root.after(150, lambda: self._show_step(self.current_step - 1, direction=-1))

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 0 — Portfolio Setup
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step0(self):
        f = AnimatedFrame(self.container)

        inner = tk.Frame(f, bg=C["panel"])
        inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9)

        # Title
        tk.Label(inner, text="Portfolio Setup", font=FONT_STEP,
                 fg=C["accent"], bg=C["panel"]).pack(anchor="w")
        tk.Label(inner, text="Configure your data source and investment parameters.",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(anchor="w", pady=(2, 6))
        divider(inner)

        # ── CSV path ──────────────────────────────────────────────────────────
        section_label(inner, "DATA FILE  (.csv)")
        path_row = tk.Frame(inner, bg=C["panel"])
        path_row.pack(fill="x")
        pf, pe = styled_entry(path_row, textvariable=self.path_var, width=42)
        pf.pack(side="left")
        self._path_entry = pe

        def browse():
            p = filedialog.askopenfilename(
                title="Select CSV data file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if p:
                self.path_var.set(p)
                pe.config(fg=C["text"])
        tk.Button(path_row, text="Browse…", font=FONT_LABEL,
                  fg=C["bg"], bg=C["accent"], relief="flat",
                  padx=12, pady=6, cursor="hand2", command=browse).pack(side="left", padx=8)

        # ── Amount ────────────────────────────────────────────────────────────
        section_label(inner, "TOTAL INVESTMENT (€)")
        af, ae = styled_entry(inner, textvariable=self.amount_var, width=18)
        af.pack(anchor="w")

        # ── Distribution ──────────────────────────────────────────────────────
        section_label(inner, "ALLOCATION STRATEGY")
        dist_row = tk.Frame(inner, bg=C["panel"])
        dist_row.pack(anchor="w", pady=4)

        self._tb_uniform  = ToggleButton(dist_row, "Uniform split",  self.dist_uniform,
                                          command=self._on_dist_change)
        self._tb_uniform.pack(side="left", padx=(0, 8))
        self._tb_optimize = ToggleButton(dist_row, "Optimize (Markowitz)",
                                          self.optimize, command=self._on_dist_change)
        self._tb_optimize.pack(side="left", padx=(0, 8))

        # Custom amounts (only visible when neither uniform nor optimize)
        self._custom_frame = tk.Frame(inner, bg=C["panel"])
        tk.Label(self._custom_frame,
                 text="Custom amounts per asset (comma-separated):",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(anchor="w", pady=(6, 2))
        cf, _ = styled_entry(self._custom_frame, textvariable=self.custom_amounts,
                              width=48, placeholder="e.g.  1000, 500, 1500, 200 …")
        cf.pack(anchor="w")
        # Default: optimize ON, custom hidden
        self.optimize.set(True)

        # ── Confidence level ──────────────────────────────────────────────────
        divider(inner)
        section_label(inner, "VAR CONFIDENCE LEVEL")
        conf_row = tk.Frame(inner, bg=C["panel"])
        conf_row.pack(anchor="w")

        for val, lbl in [("0.90", "90%"), ("0.95", "95%"), ("0.99", "99%")]:
            active = self.confidence_var.get() == val
            btn = tk.Label(conf_row, text=lbl, font=FONT_LABEL,
                            fg=C["bg"] if active else C["sub"],
                            bg=C["accent"] if active else C["muted"],
                            padx=14, pady=6, cursor="hand2")
            btn.pack(side="left", padx=4)
            btn.bind("<Button-1>", lambda e, v=val, b=btn, r=conf_row: self._select_conf(v, r))

        return f

    def _on_dist_change(self):
        u = self.dist_uniform.get()
        o = self.optimize.get()
        # Mutual exclusion
        if u and o:
            # whichever was just toggled, turn off the other
            pass  # let both coexist — user decides
        if not u and not o:
            self._custom_frame.pack(fill="x", pady=(4, 0))
        else:
            self._custom_frame.pack_forget()

    def _select_conf(self, val, row):
        self.confidence_var.set(val)
        labels = {"0.90": "90%", "0.95": "95%", "0.99": "99%"}
        for child in row.winfo_children():
            v = {v: k for k, v in labels.items()}.get(child.cget("text"))
            if child.cget("text") in labels.values():
                is_sel = (child.cget("text") == labels[val])
                child.config(fg=C["bg"] if is_sel else C["sub"],
                              bg=C["accent"] if is_sel else C["muted"])

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — Indicators
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step1(self):
        f = AnimatedFrame(self.container)

        inner = tk.Frame(f, bg=C["panel"])
        inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9)

        tk.Label(inner, text="Indicators & Charts", font=FONT_STEP,
                 fg=C["accent"], bg=C["panel"]).pack(anchor="w")
        tk.Label(inner, text="Select which analyses and charts to compute.",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(anchor="w", pady=(2, 6))
        divider(inner)

        # ── Indicators ───────────────────────────────────────────────────────
        section_label(inner, "TECHNICAL INDICATORS")
        row1 = tk.Frame(inner, bg=C["panel"])
        row1.pack(anchor="w", pady=4)

        ToggleButton(row1, "Value at Risk", self.show_var).pack(side="left", padx=(0,8))
        ToggleButton(row1, "RSI 14",        self.show_rsi).pack(side="left", padx=(0,8))
        ToggleButton(row1, "Distribution",  self.show_distrib).pack(side="left")

        divider(inner)
        section_label(inner, "MOVING AVERAGES")
        row2 = tk.Frame(inner, bg=C["panel"])
        row2.pack(anchor="w", pady=4)
        ToggleButton(row2, "SMA", self.show_sma,
                     command=lambda: self._toggle_period_row(
                         self.show_sma, sma_row)).pack(side="left", padx=(0,8))
        ToggleButton(row2, "EMA", self.show_ema,
                     command=lambda: self._toggle_period_row(
                         self.show_ema, ema_row)).pack(side="left")

        sma_row = tk.Frame(inner, bg=C["panel"])
        tk.Label(sma_row, text="SMA periods (comma-separated):",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(side="left", padx=(0, 6))
        sf, _ = styled_entry(sma_row, textvariable=self.sma_periods_var, width=14)
        sf.pack(side="left")
        sma_row.pack(anchor="w", pady=(4, 0))

        ema_row = tk.Frame(inner, bg=C["panel"])
        tk.Label(ema_row, text="EMA period:",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(side="left", padx=(0, 6))
        ef, _ = styled_entry(ema_row, textvariable=self.ema_periods_var, width=8)
        ef.pack(side="left")
        ema_row.pack(anchor="w", pady=(4, 0))

        divider(inner)
        section_label(inner, "PORTFOLIO ANALYSIS")
        row3 = tk.Frame(inner, bg=C["panel"])
        row3.pack(anchor="w", pady=4)
        ToggleButton(row3, "Efficient Frontier (Markowitz)",
                     self.show_frontier).pack(side="left")

        # Info card
        info = tk.Frame(inner, bg=C["card"], highlightbackground=C["border"],
                         highlightthickness=1)
        info.pack(fill="x", pady=(14, 0))
        tk.Label(info, text="ℹ  Efficient Frontier may take 10–30s depending on asset count.",
                 font=FONT_SMALL, fg=C["sub"], bg=C["card"]).pack(padx=10, pady=6, anchor="w")

        return f

    def _toggle_period_row(self, var, row):
        if var.get():
            row.pack(anchor="w", pady=(4, 0))
        else:
            row.pack_forget()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — Review & Launch
    # ══════════════════════════════════════════════════════════════════════════
    def _build_step2(self):
        f = AnimatedFrame(self.container)

        inner = tk.Frame(f, bg=C["panel"])
        inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9)

        tk.Label(inner, text="Review & Launch", font=FONT_STEP,
                 fg=C["accent"], bg=C["panel"]).pack(anchor="w")
        tk.Label(inner, text="Confirm your configuration before running the pipeline.",
                 font=FONT_BODY, fg=C["sub"], bg=C["panel"]).pack(anchor="w", pady=(2, 6))
        divider(inner)

        # Summary card (rebuilt dynamically each time this page is shown)
        self._summary_frame = tk.Frame(inner, bg=C["card"],
                                        highlightbackground=C["border"],
                                        highlightthickness=1)
        self._summary_frame.pack(fill="x")
        self._build_summary()

        divider(inner)

        # Progress / status
        self._status_label = tk.Label(inner, text="", font=FONT_BODY,
                                       fg=C["sub"], bg=C["panel"])
        self._status_label.pack(anchor="w")

        self._pbar = ttk.Progressbar(inner, mode="indeterminate", length=520)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar",
                         troughcolor=C["muted"],
                         background=C["accent"],
                         thickness=4)
        # pbar hidden until launch

        return f

    def _build_summary(self):
        for w in self._summary_frame.winfo_children():
            w.destroy()

        def row(label, value, color=None):
            r = tk.Frame(self._summary_frame, bg=C["card"])
            r.pack(fill="x", padx=12, pady=3)
            tk.Label(r, text=label, font=FONT_LABEL, fg=C["sub"],
                     bg=C["card"], width=22, anchor="w").pack(side="left")
            tk.Label(r, text=value, font=FONT_MONO,
                     fg=color or C["text"], bg=C["card"]).pack(side="left")

        p = self.path_var.get()
        row("CSV File",     os.path.basename(p) if p else "—", C["accent"])
        row("Investment",   f"€ {self.amount_var.get()}")
        row("Allocation",
            "Uniform" if self.dist_uniform.get() else
            "Markowitz Optimized" if self.optimize.get() else "Custom")
        row("VaR Confidence", f"{float(self.confidence_var.get()):.0%}")

        indicators = []
        if self.show_var.get():      indicators.append("VaR")
        if self.show_sma.get():      indicators.append(f"SMA({self.sma_periods_var.get()})")
        if self.show_ema.get():      indicators.append(f"EMA({self.ema_periods_var.get()})")
        if self.show_rsi.get():      indicators.append("RSI")
        if self.show_distrib.get():  indicators.append("Distribution")
        if self.show_frontier.get(): indicators.append("Frontier")

        row("Indicators", "  ".join(indicators) if indicators else "None",
            C["accent2"])

    # ══════════════════════════════════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════════════════════════════════
    def _validate(self):
        step = self.current_step

        if step == 0:
            p = self.path_var.get().strip()
            if not p or not os.path.isfile(p):
                self._shake(self._path_entry)
                messagebox.showerror("Missing file",
                                     "Please select a valid CSV data file.")
                return False
            try:
                float(self.amount_var.get())
            except ValueError:
                messagebox.showerror("Invalid amount",
                                     "Total investment must be a number.")
                return False
            if not self.dist_uniform.get() and not self.optimize.get():
                raw = self.custom_amounts.get()
                if not raw or raw.strip() == "e.g.  1000, 500, 1500, 200 …":
                    messagebox.showerror("Missing allocation",
                                         "Enter custom amounts per asset.")
                    return False

        if step == 2:
            self._build_summary()

        return True

    def _shake(self, widget):
        orig_x = widget.winfo_x()
        for i, dx in enumerate([6, -6, 4, -4, 2, -2, 0]):
            widget.after(i * 40, lambda d=dx: widget.place_configure(x=orig_x + d)
                          if widget.winfo_manager() == "place"
                          else None)

    # ══════════════════════════════════════════════════════════════════════════
    #  Launch pipeline
    # ══════════════════════════════════════════════════════════════════════════
    def _set_status(self, msg, color=None):
        self._status_label.config(text=msg, fg=color or C["sub"])
        self.root.update_idletasks()

    def _launch(self):
        self.btn_next.config(state="disabled")
        self.btn_back.config(state="disabled")
        self._pbar.pack(fill="x", pady=(8, 0))
        self._pbar.start(12)

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        try:
            self._set_status("⟳  Loading configuration…", C["accent"])
            self._apply_config()

            self._set_status("⟳  Importing data…", C["accent"])
            import Data_Importation as DI
            import Distribution_Portfolio as DP
            import Portfolio_Optimization as PO
            import Data_Visualization as DV
            import pandas as pd
            import numpy as np
            import config

            data = DI.load_data()
            has_ohlcv = (len(data) == 5)

            if has_ohlcv:
                df_open, df_high, df_low, df_close, df_volume = data
            else:
                df_open = df_high = df_low = df_volume = None
                df_close = data

            self._set_status("⟳  Computing allocation…", C["accent"])
            amounts = PO.compute_each_amount(df_close)
            df_all  = DP.Distribution_Portfolio(amounts, df_open, df_high, df_low, df_close)

            if has_ohlcv:
                df_portfolio_close = df_all[3]
            else:
                df_portfolio_close = df_all

            names = ([n.replace("_Close", "") for n in df_close.columns]
                     if has_ohlcv else list(df_close.columns))
            df_amounts = pd.DataFrame({
                "Names": names,
                "Quantity per asset": df_all[4],
            })

            self._set_status("⟳  Rendering charts…", C["accent2"])
            self.root.after(0, lambda: self._render_charts(
                df_portfolio_close, df_amounts, df_close,
                amounts, has_ohlcv, config))

        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _render_charts(self, df_portfolio_close, df_amounts, df_close,
                       amounts, has_ohlcv, config):
        import numpy as np
        import Data_Visualization as DV

        self._pbar.stop()
        self._set_status("✔  Launching charts…", C["accent2"])

        kwargs = {}
        if self.show_var.get():
            kwargs["Confidence_level"] = float(self.confidence_var.get())
        if self.show_sma.get():
            raw = self.sma_periods_var.get()
            periods = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            kwargs["periods_SMA"] = tuple(periods) if len(periods) > 1 else (periods[0] if periods else None)
        if self.show_ema.get():
            raw = self.ema_periods_var.get().strip()
            kwargs["periods_EMA"] = int(raw) if raw.isdigit() else None
        if self.show_rsi.get():
            kwargs["RSI"] = "y"
        if self.show_distrib.get():
            kwargs["Portfolio_Distribution"] = "y"

        DV.Data_Visualization(df_portfolio_close, df_amounts, **kwargs).plot_graph()

        if self.show_frontier.get():
            from Efficient_Fronter import Efficient_Frontier
            w = np.array(amounts, dtype=float)
            w = w / w.sum()
            Efficient_Frontier(
                df_close=df_close,
                portfolio_weights=w,
                n_simulations=4000,
                trading_days=252,
                rf=0.0,
            ).plot()

        self._set_status("✔  Done!", C["accent2"])
        self.btn_next.config(state="normal")
        self.btn_back.config(state="normal")

    def _on_error(self, msg):
        self._pbar.stop()
        self._pbar.pack_forget()
        self._set_status(f"✗  Error: {msg}", C["danger"])
        messagebox.showerror("Pipeline error", msg)
        self.btn_next.config(state="normal")
        self.btn_back.config(state="normal")

    # ══════════════════════════════════════════════════════════════════════════
    #  Apply config.*
    # ══════════════════════════════════════════════════════════════════════════
    def _apply_config(self):
        import config

        config.path_folder   = self.path_var.get().strip()
        config.total_amount  = float(self.amount_var.get())
        config.confidence_level = float(self.confidence_var.get())

        if self.dist_uniform.get():
            config.distribution       = "y"
            config.optimization_value = "n"
            config.distribution_value = None
        elif self.optimize.get():
            config.distribution       = "n"
            config.optimization_value = "y"
            config.distribution_value = None
        else:
            config.distribution       = "n"
            config.optimization_value = "n"
            raw = self.custom_amounts.get()
            config.distribution_value = [
                float(x.strip()) for x in raw.split(",") if x.strip()
            ]

        config.RSI = "y" if self.show_rsi.get() else "n"

        raw_sma = self.sma_periods_var.get()
        periods_sma = [int(x.strip()) for x in raw_sma.split(",")
                       if x.strip().isdigit()]
        config.periods_SMA = tuple(periods_sma) if periods_sma else None

        raw_ema = self.ema_periods_var.get().strip()
        config.periods_EMA = int(raw_ema) if raw_ema.isdigit() else None

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PortfolioWizard().run()
