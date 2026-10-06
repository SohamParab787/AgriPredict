import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from crop_yield_core import (
    FEATURE_LABELS,
    FEATURE_SPECS,
    FEATURES,
    TARGET,
    TARGET_RANGE,
    load_data,
    load_saved_bundle,
    predict_yield,
    save_artifacts,
    train_bundle,
)

# Color Palette - Professional Agricultural Theme
APP_BG = "#f4f2ec"
PANEL = "#ffffff"
PANEL_ALT = "#f8f7f4"
SIDEBAR = "#163a2d"
SIDEBAR_MUTED = "#cfe8d9"
PRIMARY = "#1f6b45"
PRIMARY_DARK = "#163b2d"
PRIMARY_LIGHT = "#e8f4ed"
ACCENT = "#d9a441"
ACCENT_LIGHT = "#fef8ea"
TEXT = "#1b2d24"
MUTED = "#52675f"
BORDER = "#dfe8e2"
CHART_GREEN = "#2e7d5d"
CHART_MUTED = "#6c8b7d"
DANGER = "#c0392b"
DANGER_LIGHT = "#fbeeed"

DEFAULTS = {feature: float(spec[4]) for feature, spec in FEATURE_SPECS.items()}
SECTION_GROUPS = {
    "Weather": ["rainfall_mm", "avg_temp_c", "sunlight_hours"],
    "Soil": ["soil_ph", "organic_carbon_pct", "nitrogen_kg_ha"],
    "Farm": ["irrigated", "farm_size_ha"],
}


class ScrollableFrame(tk.Frame):
    """Reusable scrollable container frame with mousewheel support."""

    def __init__(self, parent, bg=APP_BG, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_window = tk.Frame(self.canvas, bg=bg)

        self.scrollable_window.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window_id = self.canvas.create_window((0, 0), window=self.scrollable_window, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self._bind_mousewheel(self)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window_id, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel(child)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class FieldYieldApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AgriPredict: Smart Farm Yield Analytics and Prediction")
        self.geometry("1480x940")
        self.minsize(1200, 750)
        self.configure(bg=APP_BG)

        self.bundle = None
        self.dataset_stats = {}
        self.prediction_history = []
        self.current_prediction_val = None
        self.current_prediction_inputs = None

        self.pages = {}
        self.input_widgets = {}
        self.whatif_widgets = {}
        self.status_var = tk.StringVar(value="Initializing model…")
        self.current_page = tk.StringVar(value="Overview")

        # Canvases for dynamic prediction visualizations
        self.gauge_canvas = None
        self.profile_canvas = None
        self.history_chart_canvas = None
        self.insights_canvas = None
        self.model_chart = None
        self.feature_chart = None

        self._configure_style()
        self._build_layout()
        self.after(60, self._bootstrap_model)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=26, fieldbackground=PANEL, background=PANEL, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#e9efe9", foreground=TEXT, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#dfeee4")], foreground=[("selected", TEXT)])
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL_ALT, foreground=TEXT)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar navigation
        sidebar = tk.Frame(self, bg=SIDEBAR, width=255)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        tk.Label(sidebar, text="AgriPredict", bg=SIDEBAR, fg="white", font=("Segoe UI", 24, "bold"), anchor="w").grid(row=0, column=0, sticky="ew", padx=22, pady=(24, 2))
        tk.Label(sidebar, text="Smart Farm Yield Analytics", bg=SIDEBAR, fg=SIDEBAR_MUTED, font=("Segoe UI", 11), anchor="w").grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 18))

        nav_items = ["Overview", "Yield Prediction", "Data Insights", "Model Performance", "Feature Importance", "About Project"]
        self.side_buttons = {}
        for index, item in enumerate(nav_items, start=2):
            btn = tk.Button(
                sidebar,
                text=item,
                bg=SIDEBAR,
                fg="white",
                activebackground=PRIMARY,
                activeforeground="white",
                font=("Segoe UI", 11, "bold"),
                relief="flat",
                anchor="w",
                padx=16,
                pady=11,
                cursor="hand2",
                command=lambda page=item: self._show_page(page),
            )
            btn.grid(row=index, column=0, sticky="ew", padx=12, pady=3)
            self.side_buttons[item] = btn

        status_card = tk.Frame(sidebar, bg="#1e4638", highlightbackground="#2f604d", highlightthickness=1)
        status_card.grid(row=8, column=0, sticky="ew", padx=16, pady=(16, 12))
        tk.Label(status_card, text="SYSTEM STATUS", bg="#1e4638", fg=SIDEBAR_MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
        self.sidebar_status_label = tk.Label(status_card, text="● Loading…", bg="#1e4638", fg="#f7d986", font=("Segoe UI", 11, "bold"))
        self.sidebar_status_label.pack(anchor="w", padx=12, pady=(0, 10))

        tk.Button(
            sidebar,
            text="Retrain Models",
            bg="white",
            fg=SIDEBAR,
            activebackground="#eef6f0",
            activeforeground=SIDEBAR,
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=12,
            pady=10,
            cursor="hand2",
            command=self._retrain,
        ).grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 20))

        # Main content area
        main = tk.Frame(self, bg=APP_BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        self.topbar = tk.Frame(main, bg=APP_BG)
        self.topbar.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
        self.topbar.grid_columnconfigure(0, weight=1)
        self.topbar.grid_columnconfigure(1, weight=0)
        self.page_title = tk.Label(self.topbar, text="Overview", bg=APP_BG, fg=TEXT, font=("Segoe UI", 16, "bold"), anchor="w")
        self.page_title.grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(self.topbar, textvariable=self.status_var, bg=APP_BG, fg=MUTED, font=("Segoe UI", 9), anchor="e")
        self.status_label.grid(row=0, column=1, sticky="e")

        self.content = tk.Frame(main, bg=APP_BG)
        self.content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 18))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages = {
            "Overview": self._build_overview_page(),
            "Yield Prediction": self._build_prediction_page(),
            "Data Insights": self._build_data_insights_page(),
            "Model Performance": self._build_model_performance_page(),
            "Feature Importance": self._build_feature_importance_page(),
            "About Project": self._build_about_page(),
        }
        self._show_page("Overview")

    def _card(self, parent, **kwargs):
        return tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, **kwargs)

    def _kpi_card(self, parent, title, value, subtitle, accent=PRIMARY):
        card = self._card(parent, padx=14, pady=14)
        card.pack_propagate(False)
        tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        lbl_val = tk.Label(card, text=value, bg=PANEL, fg=accent, font=("Segoe UI", 26, "bold"), anchor="w")
        lbl_val.pack(fill="x", pady=(8, 4))
        tk.Label(card, text=subtitle, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), anchor="w", wraplength=220).pack(fill="x")
        return card, lbl_val

    # =========================================================================
    # OVERVIEW PAGE
    # =========================================================================
    def _build_overview_page(self):
        frame = tk.Frame(self.content, bg=APP_BG)
        frame.grid_columnconfigure(0, weight=1)

        hero = tk.Frame(frame, bg="#143f2d", highlightbackground="#234f3e", highlightthickness=1)
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)

        left = tk.Frame(hero, bg="#143f2d")
        left.grid(row=0, column=0, sticky="nsew", padx=22, pady=22)
        tk.Label(left, text="AgriPredict", bg="#143f2d", fg="#e7f7ee", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(left, text="AgriPredict: Smart Farm Yield Analytics and Prediction", bg="#143f2d", fg="white", font=("Segoe UI", 26, "bold"), justify="left").pack(anchor="w", pady=(4, 6))
        tk.Label(
            left,
            text="Predict crop productivity using machine learning and evaluate environmental, soil, and farm management factors.",
            bg="#143f2d",
            fg="#d9eee2",
            font=("Segoe UI", 12),
            wraplength=620,
            justify="left",
        ).pack(anchor="w")

        tk.Button(
            left,
            text="Go to Yield Prediction →",
            bg=ACCENT,
            fg="#183128",
            activebackground="#e8b854",
            activeforeground="#183128",
            font=("Segoe UI", 12, "bold"),
            padx=18,
            pady=10,
            relief="flat",
            cursor="hand2",
            command=lambda: self._show_page("Yield Prediction"),
        ).pack(anchor="w", pady=(16, 0))

        status = tk.Frame(hero, bg="#1d4d39", highlightbackground="#2c6649", highlightthickness=1)
        status.grid(row=0, column=1, sticky="n", padx=(0, 18), pady=18)
        tk.Label(status, text="Model Status", bg="#1d4d39", fg="#dff7e7", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        self.hero_model_status = tk.Label(status, text="● Ready", bg="#1d4d39", fg="#a8f6c6", font=("Segoe UI", 11, "bold"))
        self.hero_model_status.pack(anchor="w", padx=12, pady=(0, 14))

        kpis = tk.Frame(frame, bg=APP_BG)
        kpis.grid(row=1, column=0, sticky="ew")
        for i in range(5):
            kpis.grid_columnconfigure(i, weight=1)

        self.kpi_labels = {}
        cards = [
            ("records", "Dataset Records", "Cleaned samples", PRIMARY),
            ("yield", "Average Yield", "Dataset mean (t/ha)", CHART_GREEN),
            ("model", "Best Model", "Selected by CV MAE", ACCENT),
            ("r2", "Model R²", "Test set explanatory score", PRIMARY),
            ("mae", "Prediction MAE", "Test absolute error (t/ha)", CHART_GREEN),
        ]

        for col, (key, title, subtitle, accent) in enumerate(cards):
            card, lbl = self._kpi_card(kpis, title, "—", subtitle, accent)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10))
            self.kpi_labels[key] = lbl

        summary = self._card(frame, padx=20, pady=16)
        summary.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        tk.Label(summary, text="Project Snapshot & Architecture", bg=PANEL, fg=TEXT, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(
            summary,
            text=(
                "This system applies Gradient Boosting regression to predict crop yield from rainfall, temperature, sunlight, soil pH, "
                "organic carbon, nitrogen application, irrigation, and farm size. The training pipeline rigorously performs 5-fold cross-validation, "
                "missing-value imputation, and permutation importance. Dataset analytics remain strictly anchored to historical empirical data, "
                "while live predictions, gauges, what-if simulations, and history trends react dynamically to your farm inputs."
            ),
            bg=PANEL,
            fg=MUTED,
            justify="left",
            wraplength=1100,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 0))

        tk.Button(
            summary,
            text="Explore Cleaned Dataset Table",
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self._open_dataset_window,
        ).pack(anchor="w", pady=(12, 0))
        return frame

    def _populate_overview(self):
        if self.bundle is None:
            return
        df = self.bundle["df"]
        metrics = self.bundle["metrics_df"]
        model = self.bundle["best_name"]
        self.kpi_labels["records"].configure(text=f"{len(df):,}")
        self.kpi_labels["yield"].configure(text=f"{df[TARGET].mean():.2f}")
        self.kpi_labels["model"].configure(text=str(model))
        self.kpi_labels["r2"].configure(text=f"{metrics.loc[model, 'R2']:.3f}")
        self.kpi_labels["mae"].configure(text=f"{metrics.loc[model, 'MAE']:.3f}")

    # =========================================================================
    # DYNAMIC YIELD PREDICTION PAGE
    # =========================================================================
    def _build_prediction_page(self):
        scroll = ScrollableFrame(self.content, bg=APP_BG)
        container = scroll.scrollable_window

        # Header Title
        tk.Label(container, text="AgriPredict: Smart Farm Yield Analytics and Prediction", bg=APP_BG, fg=TEXT, font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Label(
            container,
            text="Configure farm parameters to generate model predictions, view dynamic yield gauges, benchmark against regional dataset statistics, run what-if scenarios, and track prediction history.",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=1100,
            justify="left",
        ).pack(anchor="w", pady=(0, 14))

        # -------------------------------------------------------------
        # 1. INPUT FORM SECTION (Weather, Soil, Farm)
        # -------------------------------------------------------------
        form = tk.Frame(container, bg=APP_BG)
        form.pack(fill="x", pady=(0, 10))
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=1)
        form.grid_columnconfigure(2, weight=1)

        self.input_widgets = {}
        for pos, section_name in enumerate(["Weather", "Soil", "Farm"]):
            section = self._card(form, padx=14, pady=12)
            section.grid(row=0, column=pos, sticky="nsew", padx=(0 if pos == 0 else 10))
            section.grid_columnconfigure(0, weight=1)
            tk.Label(section, text=section_name, bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
            for index, feature in enumerate(SECTION_GROUPS[section_name], start=1):
                self._make_input_row(section, feature, base_row=(index - 1) * 2 + 1)

        # Action Buttons
        action_bar = tk.Frame(container, bg=APP_BG)
        action_bar.pack(fill="x", pady=(10, 16))

        tk.Button(
            action_bar,
            text="🌾 PREDICT YIELD",
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            padx=22,
            pady=10,
            relief="flat",
            cursor="hand2",
            command=self._predict,
        ).pack(side="left")

        tk.Button(
            action_bar,
            text="↺ Reset Prediction",
            bg=PANEL_ALT,
            fg=TEXT,
            activebackground="#e4ede7",
            activeforeground=TEXT,
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=9,
            relief="flat",
            cursor="hand2",
            command=self._reset_inputs,
        ).pack(side="left", padx=10)

        tk.Button(
            action_bar,
            text="📊 Set to Dataset Medians",
            bg=PANEL_ALT,
            fg=MUTED,
            activebackground="#e4ede7",
            activeforeground=TEXT,
            font=("Segoe UI", 10),
            padx=12,
            pady=9,
            relief="flat",
            cursor="hand2",
            command=self._set_inputs_to_median,
        ).pack(side="left")

        # -------------------------------------------------------------
        # 2. DYNAMIC PREDICTION ANALYSIS SECTION (Result, Gauge & Profile)
        # -------------------------------------------------------------
        analysis_frame = tk.Frame(container, bg=APP_BG)
        analysis_frame.pack(fill="x", pady=(0, 16))
        analysis_frame.grid_columnconfigure(0, weight=4)
        analysis_frame.grid_columnconfigure(1, weight=5)

        # Left Card: Dynamic Result & Gauge
        gauge_card = self._card(analysis_frame, padx=16, pady=16)
        gauge_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(gauge_card, text="PREDICTED YIELD", bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        res_row = tk.Frame(gauge_card, bg=PANEL)
        res_row.pack(fill="x", pady=(4, 2))
        self.result_value_label = tk.Label(res_row, text="—", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 36, "bold"))
        self.result_value_label.pack(side="left")
        self.result_unit_label = tk.Label(res_row, text="t/ha", bg=PANEL, fg=MUTED, font=("Segoe UI", 14, "bold"))
        self.result_unit_label.pack(side="left", padx=(8, 0), pady=(14, 0))

        self.result_model_badge = tk.Label(gauge_card, text="Model: Gradient Boosting", bg=PRIMARY_LIGHT, fg=PRIMARY_DARK, font=("Segoe UI", 9, "bold"), padx=8, pady=3)
        self.result_model_badge.pack(anchor="w", pady=(0, 6))

        self.result_category_label = tk.Label(gauge_card, text="Awaiting farm input configuration", bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "italic"))
        self.result_category_label.pack(anchor="w", pady=(0, 6))

        # Gauge Host
        self.gauge_host = tk.Frame(gauge_card, bg=PANEL)
        self.gauge_host.pack(fill="both", expand=True, pady=(4, 0))

        # Right Card: Input vs Dataset Profile
        profile_card = self._card(analysis_frame, padx=16, pady=16)
        profile_card.grid(row=0, column=1, sticky="nsew")

        tk.Label(profile_card, text="INPUT VS DATASET PROFILE", bg=PANEL, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(
            profile_card,
            text="Compares your entered farm conditions against empirical dataset medians and interquartile ranges (IQR).",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 6))

        self.profile_host = tk.Frame(profile_card, bg=PANEL)
        self.profile_host.pack(fill="both", expand=True)

        # -------------------------------------------------------------
        # 3. DYNAMIC WHAT-IF ANALYSIS & SCENARIO COMPARISON
        # -------------------------------------------------------------
        whatif_card = self._card(container, padx=18, pady=16)
        whatif_card.pack(fill="x", pady=(0, 16))

        tk.Label(whatif_card, text="WHAT-IF ANALYSIS & SCENARIO COMPARISON", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(
            whatif_card,
            text="Modify individual factors to simulate alternative farming decisions and immediately quantify the yield delta generated by the model.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 12))

        whatif_cols = tk.Frame(whatif_card, bg=PANEL)
        whatif_cols.pack(fill="x")
        whatif_cols.grid_columnconfigure(0, weight=5)
        whatif_cols.grid_columnconfigure(1, weight=5)

        # Left: Scenario Controls
        builder_box = tk.Frame(whatif_cols, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
        builder_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(builder_box, text="Adjust Scenario Parameter", bg=PANEL_ALT, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w")

        sel_row = tk.Frame(builder_box, bg=PANEL_ALT)
        sel_row.pack(fill="x", pady=(6, 8))
        tk.Label(sel_row, text="Target Feature:", bg=PANEL_ALT, fg=TEXT, font=("Segoe UI", 9)).pack(side="left")
        self.whatif_feature_var = tk.StringVar(value="rainfall_mm")
        feature_options = [f for f in FEATURES]
        feature_menu = ttk.Combobox(sel_row, textvariable=self.whatif_feature_var, values=[FEATURE_LABELS[f] for f in feature_options], state="readonly", width=24)
        feature_menu.current(0)
        feature_menu.pack(side="left", padx=8)
        feature_menu.bind("<<ComboboxSelected>>", self._on_whatif_feature_selected)

        # Value adjustment row
        val_row = tk.Frame(builder_box, bg=PANEL_ALT)
        val_row.pack(fill="x", pady=(2, 6))
        tk.Label(val_row, text="New Value:", bg=PANEL_ALT, fg=TEXT, font=("Segoe UI", 9)).pack(side="left")
        self.whatif_val_var = tk.StringVar(value="1500")
        self.whatif_entry = tk.Entry(val_row, textvariable=self.whatif_val_var, width=10, font=("Segoe UI", 10), justify="center", bg=PANEL, fg=TEXT)
        self.whatif_entry.pack(side="left", padx=8)
        self.whatif_unit_lbl = tk.Label(val_row, text="mm", bg=PANEL_ALT, fg=MUTED, font=("Segoe UI", 9))
        self.whatif_unit_lbl.pack(side="left")

        self.whatif_slider = ttk.Scale(builder_box, from_=250, to=4000, orient="horizontal", value=1500)
        self.whatif_slider.pack(fill="x", pady=(2, 8))
        self.whatif_slider.bind("<Motion>", self._sync_whatif_slider)
        self.whatif_slider.bind("<ButtonRelease-1>", self._sync_whatif_slider)
        self.whatif_entry.bind("<FocusOut>", self._sync_whatif_entry)
        self.whatif_entry.bind("<Return>", self._sync_whatif_entry)

        # Quick preset buttons
        preset_row = tk.Frame(builder_box, bg=PANEL_ALT)
        preset_row.pack(fill="x", pady=(4, 6))
        tk.Label(preset_row, text="Quick Presets:", bg=PANEL_ALT, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(side="left")
        for p_label, p_feat, p_val in [
            ("+300mm Rain", "rainfall_mm", 300),
            ("+40kg N", "nitrogen_kg_ha", 40),
            ("Irrig = Yes", "irrigated", 1),
            ("Opt pH 6.8", "soil_ph", 6.8),
        ]:
            tk.Button(
                preset_row,
                text=p_label,
                bg=PANEL,
                fg=PRIMARY,
                font=("Segoe UI", 8, "bold"),
                relief="flat",
                cursor="hand2",
                padx=6,
                pady=2,
                command=lambda f=p_feat, v=p_val: self._apply_whatif_preset(f, v),
            ).pack(side="left", padx=3)

        eval_btn_row = tk.Frame(builder_box, bg=PANEL_ALT)
        eval_btn_row.pack(fill="x", pady=(8, 0))
        tk.Button(
            eval_btn_row,
            text="⚡ Evaluate Scenario",
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self._evaluate_whatif,
        ).pack(side="left")

        tk.Button(
            eval_btn_row,
            text="Apply to Main Form",
            bg=PANEL,
            fg=TEXT,
            activebackground="#e9efe9",
            activeforeground=TEXT,
            font=("Segoe UI", 9),
            padx=10,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self._apply_whatif_to_main,
        ).pack(side="left", padx=8)

        # Right: Side-by-Side Comparison
        comp_box = tk.Frame(whatif_cols, bg=PANEL_ALT, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
        comp_box.grid(row=0, column=1, sticky="nsew")
        comp_box.grid_columnconfigure(0, weight=1)
        comp_box.grid_columnconfigure(1, weight=1)

        # Current Scenario Card
        curr_card = tk.Frame(comp_box, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=10)
        curr_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(curr_card, text="CURRENT SCENARIO", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.whatif_curr_val = tk.Label(curr_card, text="—", bg=PANEL, fg=TEXT, font=("Segoe UI", 22, "bold"))
        self.whatif_curr_val.pack(anchor="w", pady=(6, 2))
        self.whatif_curr_details = tk.Label(curr_card, text="Base inputs", bg=PANEL, fg=MUTED, font=("Segoe UI", 9), justify="left")
        self.whatif_curr_details.pack(anchor="w")

        # Alternative Scenario Card
        alt_card = tk.Frame(comp_box, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=10)
        alt_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(alt_card, text="ALTERNATIVE SCENARIO", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.whatif_alt_val = tk.Label(alt_card, text="—", bg=PANEL, fg=ACCENT, font=("Segoe UI", 22, "bold"))
        self.whatif_alt_val.pack(anchor="w", pady=(6, 2))
        self.whatif_alt_details = tk.Label(alt_card, text="Modified inputs", bg=PANEL, fg=MUTED, font=("Segoe UI", 9), justify="left")
        self.whatif_alt_details.pack(anchor="w")

        # Delta Summary Badge
        delta_frame = tk.Frame(comp_box, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=8)
        delta_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        tk.Label(delta_frame, text="Yield Impact / Difference:", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.whatif_delta_lbl = tk.Label(delta_frame, text="—", bg=PRIMARY_LIGHT, fg=PRIMARY_DARK, font=("Segoe UI", 11, "bold"), padx=8, pady=2)
        self.whatif_delta_lbl.pack(side="left", padx=10)
        self.whatif_delta_note = tk.Label(delta_frame, text="Calculated using trained model", bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "italic"))
        self.whatif_delta_note.pack(side="right")

        # -------------------------------------------------------------
        # 4. PREDICTION HISTORY & DYNAMIC HISTORY TREND CHART
        # -------------------------------------------------------------
        history_section = tk.Frame(container, bg=APP_BG)
        history_section.pack(fill="x", pady=(0, 10))
        history_section.grid_columnconfigure(0, weight=5)
        history_section.grid_columnconfigure(1, weight=5)

        # Left: History Table
        hist_table_card = self._card(history_section, padx=14, pady=14)
        hist_table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        hist_head = tk.Frame(hist_table_card, bg=PANEL)
        hist_head.pack(fill="x", pady=(0, 8))
        tk.Label(hist_head, text="SESSION PREDICTION HISTORY", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 11, "bold")).pack(side="left")

        tk.Button(
            hist_head,
            text="🗑 Clear History",
            bg=DANGER_LIGHT,
            fg=DANGER,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=3,
            command=self._clear_history,
        ).pack(side="right")

        tk.Button(
            hist_head,
            text="📥 Export CSV",
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Segoe UI", 9),
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=3,
            command=self._export_history_csv,
        ).pack(side="right", padx=6)

        hist_cols = ("#", "Time", "Yield (t/ha)", "Rainfall", "Temp", "pH", "Nitrogen", "Irrig", "Farm Size")
        self.history_tree = ttk.Treeview(hist_table_card, columns=hist_cols, show="headings", height=7)
        col_widths = {"#": 35, "Time": 65, "Yield (t/ha)": 85, "Rainfall": 65, "Temp": 55, "pH": 45, "Nitrogen": 65, "Irrig": 50, "Farm Size": 70}
        for c in hist_cols:
            self.history_tree.heading(c, text=c)
            self.history_tree.column(c, width=col_widths.get(c, 60), anchor="center")

        tree_scroll = ttk.Scrollbar(hist_table_card, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scroll.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # Right: History Trend Chart
        hist_chart_card = self._card(history_section, padx=14, pady=14)
        hist_chart_card.grid(row=0, column=1, sticky="nsew")

        tk.Label(hist_chart_card, text="PREDICTION HISTORY TREND", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.history_chart_host = tk.Frame(hist_chart_card, bg=PANEL)
        self.history_chart_host.pack(fill="both", expand=True)

        # Initial renders
        self._reset_inputs()
        self._render_prediction_gauge(None)
        self._render_input_profile(DEFAULTS)
        self._render_history_chart()

        return scroll

    def _make_input_row(self, parent, feature, base_row):
        spec = FEATURE_SPECS[feature]
        lo, hi, step, default = spec[1], spec[2], spec[3], spec[4]
        value_var = tk.StringVar(value=str(default))
        unit_str = "ha" if feature.endswith("_ha") else "mm" if "rainfall" in feature else "°C" if "temp" in feature else "hrs/d" if "sunlight" in feature else "%" if "carbon" in feature else "kg/ha" if "nitrogen" in feature else ""

        widget = {
            "feature": feature,
            "lo": float(lo),
            "hi": float(hi),
            "step": float(step),
            "value_var": value_var,
            "label": tk.Label(parent, text=FEATURE_LABELS[feature], bg=PANEL, fg=TEXT, font=("Segoe UI", 9, "bold")),
            "entry": tk.Entry(parent, width=9, textvariable=value_var, justify="center", font=("Segoe UI", 9), bg=PANEL_ALT, fg=TEXT),
            "unit": tk.Label(parent, text=unit_str, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)),
        }

        widget["label"].grid(row=base_row, column=0, sticky="w", padx=(10, 4), pady=(2, 2))
        widget["entry"].grid(row=base_row, column=1, sticky="ew", padx=(0, 4), pady=(2, 2))
        widget["unit"].grid(row=base_row, column=2, sticky="w", padx=(0, 10), pady=(2, 2))

        if feature == "irrigated":
            combo = ttk.Combobox(parent, values=["No", "Yes"], state="readonly", width=8)
            combo.set("Yes" if float(default) >= 0.5 else "No")
            combo.grid(row=base_row, column=1, sticky="ew", padx=(0, 4), pady=(2, 2))
            combo.bind("<<ComboboxSelected>>", lambda event, f=feature: self._update_combobox_value(f, event.widget.get()))
            widget["combo"] = combo
            widget["entry"].grid_remove()
            widget["unit"].configure(text="")
        else:
            slider = ttk.Scale(parent, from_=lo, to=hi, value=float(default), orient="horizontal")
            slider.grid(row=base_row + 1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 6))
            slider.bind("<ButtonRelease-1>", lambda event, f=feature: self._sync_slider_value(f))
            slider.bind("<Motion>", lambda event, f=feature: self._sync_slider_value(f))
            widget["slider"] = slider
            widget["entry"].bind("<FocusOut>", lambda event, f=feature: self._sync_entry_value(f, event.widget.get()))
            widget["entry"].bind("<Return>", lambda event, f=feature: self._sync_entry_value(f, event.widget.get()))

        self.input_widgets[feature] = widget

    def _sync_slider_value(self, feature):
        widget = self.input_widgets[feature]
        if "slider" not in widget:
            return
        value = widget["slider"].get()
        widget["value_var"].set(f"{value:.{self._decimal_places(feature)}f}")

    def _sync_entry_value(self, feature, raw_value):
        widget = self.input_widgets[feature]
        try:
            value = float(raw_value)
        except ValueError:
            self.status_var.set("Invalid numeric value entered.")
            return
        lo, hi = widget["lo"], widget["hi"]
        if value < lo or value > hi:
            messagebox.showerror("Invalid input", f"{FEATURE_LABELS[feature]} must be between {lo} and {hi}.")
            value = min(max(value, lo), hi)
        widget["value_var"].set(f"{value:.{self._decimal_places(feature)}f}")
        if "slider" in widget:
            widget["slider"].set(value)

    def _update_combobox_value(self, feature, text):
        self.input_widgets[feature]["value_var"].set("1" if text == "Yes" else "0")

    def _decimal_places(self, feature):
        step = FEATURE_SPECS[feature][3]
        if step >= 1:
            return 0
        text = str(step)
        return len(text.split(".")[1]) if "." in text else 0

    def _set_inputs_to_median(self):
        if not self.dataset_stats:
            return
        medians = self.dataset_stats["medians"]
        for feat, val in medians.items():
            w = self.input_widgets[feat]
            if feat == "irrigated":
                choice = "Yes" if val >= 0.5 else "No"
                w["combo"].set(choice)
                w["value_var"].set("1" if val >= 0.5 else "0")
            else:
                w["value_var"].set(f"{val:.{self._decimal_places(feat)}f}")
                if "slider" in w:
                    w["slider"].set(val)
        self.status_var.set("Set all farm inputs to dataset median statistics.")

    def _reset_inputs(self):
        for feature, widgets in self.input_widgets.items():
            default = float(DEFAULTS[feature])
            if feature == "irrigated":
                widgets["combo"].set("Yes" if default >= 0.5 else "No")
                widgets["value_var"].set("1" if default >= 0.5 else "0")
            else:
                widgets["value_var"].set(f"{default:.{self._decimal_places(feature)}f}")
                if "slider" in widgets:
                    widgets["slider"].set(default)

        self.current_prediction_val = None
        self.current_prediction_inputs = None
        self.result_value_label.configure(text="—")
        self.result_category_label.configure(text="Awaiting farm input configuration", fg=MUTED)
        self.whatif_curr_val.configure(text="—")
        self.whatif_curr_details.configure(text="Base inputs")
        self.whatif_alt_val.configure(text="—")
        self.whatif_alt_details.configure(text="Modified inputs")
        self.whatif_delta_lbl.configure(text="—", bg=PANEL_ALT, fg=MUTED)

        self._render_prediction_gauge(None)
        self._render_input_profile(DEFAULTS)
        self.status_var.set("Prediction form reset to defaults.")

    def _collect_inputs(self):
        values = {}
        for feature in FEATURES:
            widget = self.input_widgets[feature]
            if feature == "irrigated":
                value = 1 if widget["combo"].get() == "Yes" else 0
            else:
                value = widget["value_var"].get()
                try:
                    value = float(value)
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric value for {FEATURE_LABELS[feature]}.") from exc
                lo, hi = widget["lo"], widget["hi"]
                if value < lo or value > hi:
                    raise ValueError(f"{FEATURE_LABELS[feature]} must be between {lo} and {hi}.")
            values[feature] = float(value)
        return values

    # -------------------------------------------------------------------------
    # CORE PREDICTION EXECUTION
    # -------------------------------------------------------------------------
    def _predict(self):
        if self.bundle is None:
            self.status_var.set("Model is still loading.")
            return
        try:
            values = self._collect_inputs()
            estimate = predict_yield(self.bundle["best_model"], self.bundle["bounds"], values)
            mae = float(self.bundle["metrics_df"].loc[self.bundle["best_name"], "MAE"])

            self.current_prediction_val = estimate
            self.current_prediction_inputs = values.copy()

            # Update Result Header
            self.result_value_label.configure(text=f"{estimate:.2f}")
            self.result_model_badge.configure(text=f"Model: {self.bundle['best_name']} (Test MAE ±{mae:.3f} t/ha)")

            # Categorize productivity based on yield level
            if estimate < 3.0:
                cat_text = "⚠️ Low Yield Potential — Environmental/Nutrient constraints detected"
                cat_color = DANGER
            elif estimate < 5.5:
                cat_text = "🌿 Moderate Yield Potential — Standard regional output"
                cat_color = "#b08826"
            elif estimate < 8.0:
                cat_text = "⭐ Optimal Yield Potential — Strong agronomic conditions"
                cat_color = PRIMARY
            else:
                cat_text = "🏆 Exceptional Yield Potential — Highly fertile & favorable conditions"
                cat_color = PRIMARY_DARK

            self.result_category_label.configure(text=cat_text, fg=cat_color)

            # Update Dynamic Visualizations
            self._render_prediction_gauge(estimate)
            self._render_input_profile(values)

            # Update What-If baseline
            self.whatif_curr_val.configure(text=f"{estimate:.2f} t/ha")
            self.whatif_curr_details.configure(
                text=f"Rain: {values['rainfall_mm']:.0f}mm | N: {values['nitrogen_kg_ha']:.0f}kg | pH: {values['soil_ph']:.1f}"
            )

            # Add to session history
            self._add_to_history(values, estimate)
            self.status_var.set(f"Predicted yield: {estimate:.2f} t/ha using {self.bundle['best_name']}.")

        except ValueError as exc:
            messagebox.showerror("Input Validation Error", str(exc))
            self.status_var.set("Prediction failed due to invalid input.")

    # -------------------------------------------------------------------------
    # DYNAMIC GAUGE VISUALIZATION
    # -------------------------------------------------------------------------
    def _render_prediction_gauge(self, predicted_yield=None, max_yield=12.0):
        if self.gauge_canvas is not None:
            self.gauge_canvas.get_tk_widget().destroy()
            self.gauge_canvas = None

        fig = Figure(figsize=(4.4, 2.7), dpi=100)
        fig.patch.set_facecolor(PANEL)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)

        # Gauge zones: (low, high, color)
        bands = [
            (0.0, 3.0, "#d97736"),
            (3.0, 5.5, "#cca43b"),
            (5.5, 8.5, "#2e7d5d"),
            (8.5, 12.0, "#163b2d"),
        ]

        # Draw background ring arcs
        for low, high, color in bands:
            theta1 = 180 - (low / max_yield) * 180
            theta2 = 180 - (high / max_yield) * 180
            theta = np.linspace(np.radians(theta1), np.radians(theta2), 100)
            r_inner = 0.64
            r_outer = 0.94
            x_inner = r_inner * np.cos(theta)
            y_inner = r_inner * np.sin(theta)
            x_outer = r_outer * np.cos(theta)
            y_outer = r_outer * np.sin(theta)
            x = np.concatenate([x_inner, x_outer[::-1]])
            y = np.concatenate([y_inner, y_outer[::-1]])
            ax.fill(x, y, color=color, alpha=0.88)

        # Zone tick labels
        ticks = [0, 3, 5.5, 8.5, 12]
        for t in ticks:
            angle = 180 - (t / max_yield) * 180
            rad = np.radians(angle)
            tx = 1.05 * np.cos(rad)
            ty = 1.05 * np.sin(rad)
            ax.text(tx, ty, f"{t:g}", ha="center", va="center", fontsize=8, fontweight="bold", color=MUTED)

        if predicted_yield is not None:
            val_clamped = min(max(predicted_yield, 0.0), max_yield)
            needle_angle = 180 - (val_clamped / max_yield) * 180
            needle_rad = np.radians(needle_angle)

            # Draw needle pointer
            ax.annotate(
                "",
                xy=(0.80 * np.cos(needle_rad), 0.80 * np.sin(needle_rad)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY_DARK, lw=3.2, mutation_scale=14),
                zorder=4,
            )

            # Center pivot
            pivot = plt.Circle((0, 0), 0.12, color=PRIMARY_DARK, zorder=5)
            ax.add_patch(pivot)
            pivot_inner = plt.Circle((0, 0), 0.05, color=PANEL, zorder=6)
            ax.add_patch(pivot_inner)

            ax.text(0, -0.20, f"{predicted_yield:.2f} t/ha", ha="center", va="center", fontsize=15, fontweight="bold", color=PRIMARY_DARK)
            ax.text(0, -0.40, "Active Model Output", ha="center", va="center", fontsize=9, color=MUTED)
        else:
            pivot = plt.Circle((0, 0), 0.10, color=MUTED, zorder=5)
            ax.add_patch(pivot)
            ax.text(0, -0.22, "— t/ha", ha="center", va="center", fontsize=14, fontweight="bold", color=MUTED)
            ax.text(0, -0.40, "Ready to Predict", ha="center", va="center", fontsize=9, color=MUTED)

        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-0.52, 1.15)
        ax.axis("off")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.gauge_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.gauge_canvas = canvas

    # -------------------------------------------------------------------------
    # INPUT VS DATASET PROFILE VISUALIZATION
    # -------------------------------------------------------------------------
    def _render_input_profile(self, user_inputs):
        if self.profile_canvas is not None:
            self.profile_canvas.get_tk_widget().destroy()
            self.profile_canvas = None

        fig = Figure(figsize=(6.4, 2.7), dpi=100)
        fig.patch.set_facecolor(PANEL)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)

        numeric_features = [
            ("rainfall_mm", "Rainfall"),
            ("avg_temp_c", "Temperature"),
            ("sunlight_hours", "Sunlight"),
            ("soil_ph", "Soil pH"),
            ("organic_carbon_pct", "Org Carbon"),
            ("nitrogen_kg_ha", "Nitrogen"),
            ("farm_size_ha", "Farm Size"),
        ]

        if not self.dataset_stats:
            ax.text(0.5, 0.5, "Dataset statistics loading…", ha="center", va="center", color=MUTED)
            ax.axis("off")
        else:
            stats = self.dataset_stats
            y_positions = np.arange(len(numeric_features))

            for idx, (feat, label) in enumerate(numeric_features):
                f_min = stats["mins"][feat]
                f_max = stats["maxs"][feat]
                f_med = stats["medians"][feat]
                f_q25 = stats["q25"][feat]
                f_q75 = stats["q75"][feat]
                u_val = user_inputs.get(feat, f_med)

                rng = f_max - f_min if f_max != f_min else 1.0
                norm_iqr_left = (f_q25 - f_min) / rng * 100
                norm_iqr_width = (f_q75 - f_q25) / rng * 100
                norm_med = (f_med - f_min) / rng * 100
                norm_user = min(max((u_val - f_min) / rng * 100, 0), 100)

                # Full range baseline
                ax.barh(idx, 100, left=0, height=0.34, color="#eaf0ec", edgecolor="none")
                # Interquartile range (IQR 25% - 75%)
                ax.barh(idx, norm_iqr_width, left=norm_iqr_left, height=0.34, color="#a9d6c2", edgecolor="none", label="Dataset IQR (25–75%)" if idx == 0 else "")
                # Median line
                ax.plot([norm_med, norm_med], [idx - 0.22, idx + 0.22], color=PRIMARY_DARK, lw=2, label="Dataset Median" if idx == 0 else "")
                # User Farm Input Marker
                ax.scatter([norm_user], [idx], color=ACCENT, s=80, edgecolors=PRIMARY_DARK, linewidth=1.4, zorder=5, label="Current Input" if idx == 0 else "")

            ax.set_yticks(y_positions)
            ax.set_yticklabels([lbl for _, lbl in numeric_features], fontsize=8, fontweight="bold", color=TEXT)
            ax.set_xlabel("Relative Feature Range: Min → 25% → Median → 75% → Max", fontsize=8, color=MUTED)
            ax.set_xlim(-2, 102)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["Min", "25%", "50%", "75%", "Max"], fontsize=8, color=MUTED)
            ax.grid(axis="x", alpha=0.15, linestyle=":")
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, facecolor=PANEL)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.profile_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.profile_canvas = canvas

    # -------------------------------------------------------------------------
    # WHAT-IF ANALYSIS & SCENARIO COMPARISON
    # -------------------------------------------------------------------------
    def _on_whatif_feature_selected(self, event=None):
        label = self.whatif_feature_var.get()
        # Find feature key from label
        feature = next((f for f, l in FEATURE_LABELS.items() if l == label), "rainfall_mm")
        spec = FEATURE_SPECS[feature]
        lo, hi, default = spec[1], spec[2], spec[4]

        unit_str = "ha" if feature.endswith("_ha") else "mm" if "rainfall" in feature else "°C" if "temp" in feature else "hrs/d" if "sunlight" in feature else "%" if "carbon" in feature else "kg/ha" if "nitrogen" in feature else ""
        self.whatif_unit_lbl.configure(text=unit_str)

        # Update slider range and entry
        curr_inputs = self.current_prediction_inputs or self._collect_inputs()
        current_val = curr_inputs.get(feature, default)

        self.whatif_slider.configure(from_=lo, to=hi)
        self.whatif_slider.set(current_val)
        self.whatif_val_var.set(f"{current_val:.{self._decimal_places(feature)}f}")

    def _sync_whatif_slider(self, event=None):
        label = self.whatif_feature_var.get()
        feature = next((f for f, l in FEATURE_LABELS.items() if l == label), "rainfall_mm")
        val = self.whatif_slider.get()
        self.whatif_val_var.set(f"{val:.{self._decimal_places(feature)}f}")

    def _sync_whatif_entry(self, event=None):
        label = self.whatif_feature_var.get()
        feature = next((f for f, l in FEATURE_LABELS.items() if l == label), "rainfall_mm")
        spec = FEATURE_SPECS[feature]
        lo, hi = spec[1], spec[2]
        try:
            val = float(self.whatif_val_var.get())
            val = min(max(val, lo), hi)
            self.whatif_slider.set(val)
            self.whatif_val_var.set(f"{val:.{self._decimal_places(feature)}f}")
        except ValueError:
            pass

    def _apply_whatif_preset(self, feature, val_or_delta):
        curr_inputs = self.current_prediction_inputs or self._collect_inputs()
        self.whatif_feature_var.set(FEATURE_LABELS[feature])
        spec = FEATURE_SPECS[feature]
        lo, hi = spec[1], spec[2]

        if feature == "irrigated":
            target_val = float(val_or_delta)
        elif feature in ["rainfall_mm", "nitrogen_kg_ha"]:
            target_val = min(curr_inputs[feature] + float(val_or_delta), hi)
        else:
            target_val = float(val_or_delta)

        self.whatif_slider.configure(from_=lo, to=hi)
        self.whatif_slider.set(target_val)
        self.whatif_val_var.set(f"{target_val:.{self._decimal_places(feature)}f}")
        self._on_whatif_feature_selected()
        self._evaluate_whatif()

    def _evaluate_whatif(self):
        if self.bundle is None:
            return
        try:
            # Baseline inputs
            base_inputs = self.current_prediction_inputs if self.current_prediction_inputs is not None else self._collect_inputs()
            base_yield = self.current_prediction_val if self.current_prediction_val is not None else predict_yield(self.bundle["best_model"], self.bundle["bounds"], base_inputs)

            # Scenario inputs
            scenario_inputs = base_inputs.copy()
            label = self.whatif_feature_var.get()
            feature = next((f for f, l in FEATURE_LABELS.items() if l == label), "rainfall_mm")
            new_val = float(self.whatif_val_var.get())
            scenario_inputs[feature] = new_val

            # Compute alternative prediction using actual model
            alt_yield = predict_yield(self.bundle["best_model"], self.bundle["bounds"], scenario_inputs)
            delta = alt_yield - base_yield
            pct_change = (delta / base_yield * 100) if base_yield != 0 else 0

            # Update UI Display
            self.whatif_curr_val.configure(text=f"{base_yield:.2f} t/ha")
            self.whatif_curr_details.configure(
                text=f"Base: {FEATURE_LABELS[feature].split('(')[0].strip()} = {base_inputs[feature]:g}"
            )

            self.whatif_alt_val.configure(text=f"{alt_yield:.2f} t/ha")
            self.whatif_alt_details.configure(
                text=f"Modified: {FEATURE_LABELS[feature].split('(')[0].strip()} = {new_val:g}"
            )

            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f} t/ha ({sign}{pct_change:.1f}%)"
            badge_bg = PRIMARY_LIGHT if delta >= 0 else DANGER_LIGHT
            badge_fg = PRIMARY_DARK if delta >= 0 else DANGER
            self.whatif_delta_lbl.configure(text=delta_str, bg=badge_bg, fg=badge_fg)

            self.status_var.set(f"What-If Scenario: {delta_str} yield difference ({self.bundle['best_name']}).")

        except Exception as exc:
            messagebox.showerror("Scenario Error", str(exc))

    def _apply_whatif_to_main(self):
        label = self.whatif_feature_var.get()
        feature = next((f for f, l in FEATURE_LABELS.items() if l == label), "rainfall_mm")
        new_val = self.whatif_val_var.get()
        w = self.input_widgets[feature]
        if feature == "irrigated":
            choice = "Yes" if float(new_val) >= 0.5 else "No"
            w["combo"].set(choice)
            w["value_var"].set("1" if float(new_val) >= 0.5 else "0")
        else:
            w["value_var"].set(new_val)
            if "slider" in w:
                w["slider"].set(float(new_val))
        self._predict()

    # -------------------------------------------------------------------------
    # PREDICTION SESSION HISTORY & TREND CHART
    # -------------------------------------------------------------------------
    def _add_to_history(self, inputs, predicted_val):
        rec_id = len(self.prediction_history) + 1
        record = {
            "id": rec_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "yield": round(predicted_val, 2),
            "rainfall": inputs["rainfall_mm"],
            "temp": inputs["avg_temp_c"],
            "ph": inputs["soil_ph"],
            "nitrogen": inputs["nitrogen_kg_ha"],
            "irrigated": "Yes" if inputs["irrigated"] == 1 else "No",
            "farm_size": inputs["farm_size_ha"],
            "sunlight": inputs["sunlight_hours"],
            "organic_carbon": inputs["organic_carbon_pct"],
        }
        self.prediction_history.append(record)
        self.history_tree.insert(
            "",
            "end",
            values=(
                record["id"],
                record["time"],
                f"{record['yield']:.2f}",
                f"{record['rainfall']:.0f}",
                f"{record['temp']:.1f}",
                f"{record['ph']:.1f}",
                f"{record['nitrogen']:.0f}",
                record["irrigated"],
                f"{record['farm_size']:.1f}",
            ),
        )
        self._render_history_chart()

    def _render_history_chart(self):
        if self.history_chart_canvas is not None:
            self.history_chart_canvas.get_tk_widget().destroy()
            self.history_chart_canvas = None

        fig = Figure(figsize=(6.0, 2.5), dpi=100)
        fig.patch.set_facecolor(PANEL)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)

        if not self.prediction_history:
            ax.text(0.5, 0.5, "No prediction history yet.\nMake predictions to view dynamic session trend.", ha="center", va="center", color=MUTED, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        elif len(self.prediction_history) == 1:
            rec = self.prediction_history[0]
            ax.scatter([1], [rec["yield"]], color=PRIMARY, s=120, zorder=5, label="Prediction 1")
            ax.annotate(
                f"#{rec['id']}: {rec['yield']:.2f} t/ha",
                (1, rec["yield"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontweight="bold",
                color=PRIMARY_DARK,
                fontsize=9,
            )
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([1])
            ax.set_xticklabels(["Prediction 1"])
            ax.set_ylabel("Yield (t/ha)", fontsize=9, color=MUTED)
            ax.grid(alpha=0.15)
            ax.set_title("Single Session Prediction", fontsize=9, fontweight="bold", color=PRIMARY_DARK)
        else:
            ids = [r["id"] for r in self.prediction_history]
            yields = [r["yield"] for r in self.prediction_history]

            # Line + Marker Trend
            ax.plot(ids, yields, marker="o", markersize=6, color=PRIMARY, lw=2.2, label="Predicted Yield", zorder=4)
            ax.fill_between(ids, yields, alpha=0.12, color=PRIMARY)

            # Benchmark dataset mean line if available
            if self.dataset_stats:
                mean_y = self.dataset_stats["yield_mean"]
                ax.axhline(mean_y, color=ACCENT, linestyle="--", lw=1.3, alpha=0.8, label=f"Dataset Mean ({mean_y:.2f})")

            # Point callout for latest
            latest_id, latest_y = ids[-1], yields[-1]
            ax.annotate(
                f"{latest_y:.2f}",
                (latest_id, latest_y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontweight="bold",
                color=PRIMARY_DARK,
                fontsize=8.5,
            )

            ax.set_xticks(ids)
            ax.set_xlabel("Prediction #", fontsize=8, color=MUTED)
            ax.set_ylabel("Yield (t/ha)", fontsize=8, color=MUTED)
            ax.grid(alpha=0.15)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9, facecolor=PANEL)
            ax.set_title("Prediction Sequence Trend", fontsize=9, fontweight="bold", color=PRIMARY_DARK)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.history_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.history_chart_canvas = canvas

    def _clear_history(self):
        if not self.prediction_history:
            return
        if messagebox.askyesno("Clear Prediction History", "Are you sure you want to clear all prediction history for this session?"):
            self.prediction_history.clear()
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            self._render_history_chart()
            self.status_var.set("Session prediction history cleared.")

    def _export_history_csv(self):
        if not self.prediction_history:
            messagebox.showinfo("Export CSV", "No prediction history to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Prediction History",
        )
        if path:
            keys = self.prediction_history[0].keys()
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.prediction_history)
            self.status_var.set(f"Prediction history exported to {Path(path).name}.")

    # =========================================================================
    # STATIC DATASET ANALYTICS: DATA INSIGHTS
    # =========================================================================
    def _build_data_insights_page(self):
        frame = tk.Frame(self.content, bg=APP_BG)
        frame.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text="Dataset Insights & Ground Truth Analytics", bg=APP_BG, fg=TEXT, font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            text="Exploratory bivariate relationships based strictly on the cleaned empirical training dataset (2,988 records). These ground-truth distributions remain static and are not altered by live prediction queries.",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, sticky="w")

        self.insights_canvas_host = tk.Frame(frame, bg=APP_BG)
        self.insights_canvas_host.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        self.insights_canvas_host.grid_columnconfigure(0, weight=1)
        self.insights_canvas = None
        return frame

    def _render_data_insights(self):
        if self.bundle is None:
            return
        if self.insights_canvas is not None:
            self.insights_canvas.get_tk_widget().destroy()
            self.insights_canvas = None

        df = self.bundle["df"]
        fig = Figure(figsize=(14, 7.8), dpi=100)
        fig.patch.set_facecolor(APP_BG)
        axes = fig.subplots(2, 3)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.38, wspace=0.32)

        plots = [
            ("rainfall_mm", "Rainfall (mm)"),
            ("avg_temp_c", "Average temperature (°C)"),
            ("soil_ph", "Soil pH"),
            ("nitrogen_kg_ha", "Nitrogen (kg/ha)"),
            ("farm_size_ha", "Farm size (ha)"),
            ("irrigated", "Irrigation"),
        ]

        for ax, (feature, label) in zip(axes.flat, plots):
            ax.set_facecolor(PANEL)
            if feature == "irrigated":
                box = ax.boxplot([df.loc[df[feature] == 0, TARGET], df.loc[df[feature] == 1, TARGET]], patch_artist=True, widths=0.45)
                colors = ["#cca43b", CHART_GREEN]
                for patch, color in zip(box["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.85)
                ax.set_title("Irrigation vs Yield Distribution", fontsize=10, fontweight="bold", color=PRIMARY_DARK)
                ax.set_ylabel("Yield (t/ha)", fontsize=9, color=MUTED)
                ax.set_xlabel("Water Management Practice", fontsize=9, color=MUTED)
                ax.set_xticklabels(["Rain-fed (0)", "Irrigated (1)"])
            else:
                ax.scatter(df[feature], df[TARGET], color=CHART_GREEN, s=16, alpha=0.55, edgecolors="none")
                ax.set_title(f"{label} vs Yield", fontsize=10, fontweight="bold", color=PRIMARY_DARK)
                ax.set_xlabel(label, fontsize=9, color=MUTED)
                ax.set_ylabel("Yield (t/ha)", fontsize=9, color=MUTED)
            ax.grid(alpha=0.15)

        canvas = FigureCanvasTkAgg(fig, master=self.insights_canvas_host)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.insights_canvas = canvas

    # =========================================================================
    # STATIC DATASET ANALYTICS: MODEL PERFORMANCE
    # =========================================================================
    def _build_model_performance_page(self):
        frame = tk.Frame(self.content, bg=APP_BG)
        frame.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text="Machine Learning Model Evaluation", bg=APP_BG, fg=TEXT, font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            text="Comprehensive benchmarking of regression algorithms using 5-fold cross-validation on training data and independent held-out test evaluation. Lower MAE indicates higher predictive accuracy.",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, sticky="w")

        table_card = self._card(frame, padx=10, pady=10)
        table_card.grid(row=2, column=0, sticky="ew", pady=(14, 8))
        columns = ("Model", "CV MAE (t/ha)", "CV R² Mean", "Test MAE (t/ha)", "RMSE (t/ha)", "Test R²")
        self.model_tree = ttk.Treeview(table_card, columns=columns, show="headings", height=5)
        for column in columns:
            self.model_tree.heading(column, text=column)
            self.model_tree.column(column, width=130, anchor="center")
        self.model_tree.pack(fill="x")

        self.model_chart_host = tk.Frame(frame, bg=APP_BG)
        self.model_chart_host.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        self.model_chart_host.grid_columnconfigure(0, weight=1)
        self.model_chart = None
        return frame

    def _render_model_performance(self):
        if self.bundle is None:
            return
        metrics = self.bundle["metrics_df"].round(3)
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        for model_name, row in metrics.iterrows():
            self.model_tree.insert(
                "",
                "end",
                values=(
                    model_name,
                    f"{row['CV_MAE']:.3f}",
                    f"{row['CV_R2_mean']:.3f}",
                    f"{row['MAE']:.3f}",
                    f"{row['RMSE']:.3f}",
                    f"{row['R2']:.3f}",
                ),
            )

        if self.model_chart is not None:
            self.model_chart.get_tk_widget().destroy()
            self.model_chart = None

        fig = Figure(figsize=(12, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(APP_BG)
        ax.set_facecolor(PANEL)
        mae_series = metrics["MAE"].sort_values(ascending=True)
        colors = [PRIMARY if idx == metrics.index.get_loc(self.bundle["best_name"]) else CHART_MUTED for idx in range(len(mae_series))]
        ax.barh(mae_series.index.astype(str), mae_series.values, color=colors, edgecolor="none", height=0.55)
        ax.set_title("Test Set MAE by Model (Lower is Better)", fontsize=11, fontweight="bold", color=PRIMARY_DARK)
        ax.set_xlabel("Mean Absolute Error (t/ha)", fontsize=9, color=MUTED)
        ax.set_ylabel("Algorithm", fontsize=9, color=MUTED)
        ax.grid(axis="x", alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=self.model_chart_host)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.model_chart = canvas

    # =========================================================================
    # STATIC DATASET ANALYTICS: FEATURE IMPORTANCE
    # =========================================================================
    def _build_feature_importance_page(self):
        frame = tk.Frame(self.content, bg=APP_BG)
        frame.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text="Feature Importance Analysis", bg=APP_BG, fg=TEXT, font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            text="Permutation feature importance measured by the increase in test-set MAE when a feature is randomly shuffled. Higher values indicate greater reliance by the Gradient Boosting model.",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, sticky="w")

        self.feature_chart_host = tk.Frame(frame, bg=APP_BG)
        self.feature_chart_host.grid(row=2, column=0, sticky="nsew", pady=(14, 6))
        self.feature_chart_host.grid_columnconfigure(0, weight=1)
        self.feature_chart = None

        feature_list = self._card(frame, padx=18, pady=14)
        feature_list.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        tk.Label(feature_list, text="Ranked Predictor Summary", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.feature_rank = tk.Label(feature_list, bg=PANEL, fg=TEXT, justify="left", anchor="w", font=("Segoe UI", 10), wraplength=1050)
        self.feature_rank.pack(fill="x", pady=(6, 0))
        return frame

    def _render_feature_importance(self):
        if self.bundle is None or self.bundle.get("importance") is None:
            return
        importance = self.bundle["importance"].sort_values(ascending=True)
        if self.feature_chart is not None:
            self.feature_chart.get_tk_widget().destroy()
            self.feature_chart = None

        fig = Figure(figsize=(12, 4.5), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(APP_BG)
        ax.set_facecolor(PANEL)
        ax.barh(importance.index.astype(str), importance.values, color=CHART_GREEN, edgecolor="none", height=0.55)
        ax.set_title("Permutation Importance (Increase in Test MAE when Shuffled)", fontsize=11, fontweight="bold", color=PRIMARY_DARK)
        ax.set_xlabel("MAE Increase (t/ha)", fontsize=9, color=MUTED)
        ax.set_ylabel("Agronomic Feature", fontsize=9, color=MUTED)
        ax.grid(axis="x", alpha=0.2)

        canvas = FigureCanvasTkAgg(fig, master=self.feature_chart_host)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.feature_chart = canvas

        ranked = [f"{index}. {FEATURE_LABELS.get(name, name)}: +{value:.3f} t/ha error penalty" for index, (name, value) in enumerate(importance.iloc[::-1].items(), start=1)]
        self.feature_rank.configure(text="\n".join(ranked))

    # =========================================================================
    # ABOUT PAGE
    # =========================================================================
    def _build_about_page(self):
        scroll = ScrollableFrame(self.content, bg=APP_BG)
        frame = scroll.scrollable_window

        banner = self._card(frame, padx=20, pady=18)
        banner.pack(fill="x", pady=(0, 16))
        tk.Label(banner, text="About AgriPredict", bg=PANEL, fg=TEXT, font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(
            banner,
            text="AgriPredict is an intelligent agricultural decision-support platform designed to predict crop productivity, assess farm condition profiles, and conduct sensitivity what-if simulations.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 11),
            wraplength=1050,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        sections = [
            ("Problem Statement", "Crop yield prediction is crucial for food security, precision agriculture, and risk management. Variabilities in climate, soil chemistry, and irrigation directly influence seasonal harvest outcomes."),
            ("Methodology & ML Pipeline", "The platform loads empirical data from data/crop_yield_data.csv, removes physical outliers and duplicates, performs median imputation inside isolated pipelines, and benchmarks 5 models using 5-fold cross-validation."),
            ("Selected Best Model", "Gradient Boosting (HistGradientBoostingRegressor) achieves the lowest CV MAE (0.640 t/ha) and test MAE (0.641 t/ha) with an R² score of 0.708."),
            ("Dynamic vs Static Analytics", "Dataset distributions and model metrics represent historical ground truth and remain static. In contrast, prediction gauges, farm profile benchmark comparisons, what-if scenarios, and session history charts update dynamically in real time."),
            ("Data Integrity", "All values are generated dynamically from the trained scikit-learn models and empirical dataset statistics without hardcoded estimates or fabricated records."),
        ]

        for heading, text in sections:
            card = self._card(frame, padx=16, pady=12)
            card.pack(fill="x", pady=(0, 10))
            tk.Label(card, text=heading, bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 12, "bold")).pack(anchor="w")
            tk.Label(card, text=text, bg=PANEL, fg=MUTED, font=("Segoe UI", 10), wraplength=1050, justify="left").pack(anchor="w", pady=(6, 0))

        tech = self._card(frame, padx=16, pady=12)
        tech.pack(fill="x", pady=(0, 10))
        tk.Label(tech, text="Technologies & Dependencies", bg=PANEL, fg=PRIMARY_DARK, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(tech, text="Python 3 • scikit-learn • pandas • NumPy • Matplotlib • Tkinter / ttk", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))

        return scroll

    # =========================================================================
    # DATASET PREVIEW MODAL
    # =========================================================================
    def _open_dataset_window(self):
        if self.bundle is None:
            return
        df = self.bundle["df"]
        top = tk.Toplevel(self)
        top.title("Cleaned Crop Yield Dataset Preview")
        top.geometry("1120x620")
        top.minsize(850, 480)

        tk.Label(top, text="Cleaned Dataset Records", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(top, text=f"Total Records: {len(df):,} | Features: {len(FEATURES)} | Target: {TARGET}", fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=16)

        search_frame = tk.Frame(top, bg=APP_BG)
        search_frame.pack(fill="x", padx=16, pady=(10, 6))
        search_var = tk.StringVar()
        tk.Label(search_frame, text="Filter records:", font=("Segoe UI", 10)).pack(side="left")
        ttk.Entry(search_frame, textvariable=search_var, width=30).pack(side="left", padx=(8, 10))

        table_frame = tk.Frame(top)
        table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        table = ttk.Treeview(table_frame, show="headings")
        columns = list(df.columns)
        for col in columns:
            table.heading(col, text=col)
            table.column(col, width=110, anchor="center")
        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        table.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")

        def update_rows():
            query = search_var.get().strip().lower()
            rows = df.head(300).copy()
            if query:
                mask = False
                for col in columns:
                    mask |= df[col].astype(str).str.lower().str.contains(query, na=False)
                rows = df.loc[mask].head(300)
            for item in table.get_children():
                table.delete(item)
            for _, row in rows.iterrows():
                table.insert("", "end", values=[f"{row[c]:.2f}" if isinstance(row[c], float) else row[c] for c in columns])

        update_rows()
        search_var.trace_add("write", lambda *_: update_rows())

    # =========================================================================
    # CONTROLLER & LIFECYCLE
    # =========================================================================
    def _compute_dataset_stats(self):
        if self.bundle is None:
            return
        df = self.bundle["df"]
        self.dataset_stats = {
            "medians": df[FEATURES].median().to_dict(),
            "means": df[FEATURES].mean().to_dict(),
            "mins": df[FEATURES].min().to_dict(),
            "maxs": df[FEATURES].max().to_dict(),
            "q25": df[FEATURES].quantile(0.25).to_dict(),
            "q75": df[FEATURES].quantile(0.75).to_dict(),
            "yield_mean": float(df[TARGET].mean()),
            "yield_median": float(df[TARGET].median()),
            "yield_min": float(df[TARGET].min()),
            "yield_max": float(df[TARGET].max()),
        }

    def _show_page(self, name):
        self.current_page.set(name)
        self.page_title.configure(text=name)
        for page_name, frame in self.pages.items():
            frame.grid_forget()
        self.pages[name].grid(row=0, column=0, sticky="nsew")
        for btn_name, btn in self.side_buttons.items():
            if btn_name == name:
                btn.configure(bg=PRIMARY, activebackground=PRIMARY, fg="white")
            else:
                btn.configure(bg=SIDEBAR, activebackground=PRIMARY, fg="white")

    def _bootstrap_model(self):
        try:
            saved = load_saved_bundle()
            if saved and saved.get("bounds") is not None and saved.get("metrics_df") is not None:
                self.bundle = saved
                self.status_var.set("Loaded saved Gradient Boosting model from crop_yield_results/")
            else:
                self.status_var.set("Training regression models…")
                self.update_idletasks()
                bundle = train_bundle(load_data())
                save_artifacts(bundle)
                self.bundle = bundle
                self.status_var.set("Models successfully trained and saved.")

            self._compute_dataset_stats()
            self._populate_overview()
            self._render_data_insights()
            self._render_model_performance()
            self._render_feature_importance()
            self._render_input_profile(DEFAULTS)
            self.sidebar_status_label.configure(text="● Ready", fg="#8ee0a9")
            if hasattr(self, "hero_model_status"):
                self.hero_model_status.configure(text="● Active", fg="#a8f6c6")
        except Exception as exc:
            self.status_var.set("Model initialization failed.")
            self.sidebar_status_label.configure(text="● Error", fg="#e08e8e")
            messagebox.showerror("Model Loading Error", str(exc))

    def _retrain(self):
        try:
            self.status_var.set("Retraining regression models…")
            self.sidebar_status_label.configure(text="● Training…", fg="#f7d986")
            self.update_idletasks()
            bundle = train_bundle(load_data())
            save_artifacts(bundle)
            self.bundle = bundle
            self._compute_dataset_stats()
            self._populate_overview()
            self._render_data_insights()
            self._render_model_performance()
            self._render_feature_importance()
            self._render_input_profile(self.current_prediction_inputs or DEFAULTS)
            self.sidebar_status_label.configure(text="● Ready", fg="#8ee0a9")
            self.status_var.set("Models retrained successfully.")
            messagebox.showinfo("AgriPredict", "Models retrained and artifacts updated successfully.")
        except Exception as exc:
            self.status_var.set("Retraining failed.")
            self.sidebar_status_label.configure(text="● Error", fg="#e08e8e")
            messagebox.showerror("Retraining Error", str(exc))


def main() -> None:
    app = FieldYieldApp()
    app.mainloop()


if __name__ == "__main__":
    main()
