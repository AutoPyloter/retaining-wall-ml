# app.py
# StabilityApp — main application window.
#
# Responsibilities:
#   - Build the two-tab UI (Input & Visualisation / Model Selection)
#   - Render the retaining wall cross-section on a Canvas
#   - Load models, run inference, and display results

import logging
import os
import re
import tkinter as tk
import warnings
from logging.handlers import RotatingFileHandler
from tkinter import messagebox, ttk
from typing import Any, List, Tuple

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from config import read_config, write_config


def resource_path(relative_path: str) -> str:
    try:
        import sys

        base = sys._MEIPASS
    except AttributeError:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


# joblib.load looks for select_top_k_features in __main__ because the pkl
# was saved while train_models.py was running as __main__.
# Injecting it here makes all models loadable without retraining.
import warnings

import __main__
from language import list_languages, load_translations
from model_info import MODEL_INFO
from pipeline_components import OptionalScaler, select_top_k_features  # noqa: F401
from preprocessing import preprocess_inputs

__main__.select_top_k_features = select_top_k_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = os.environ.get("LOG_FILE", "app.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
)
logger.addHandler(_handler)

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def log_exceptions(func):
    """Log unhandled exceptions raised inside *func* before re-raising."""

    def wrapper(*args, **kwargs):
        logger.debug("%s started.", func.__name__)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.exception("%s raised an error: %s", func.__name__, exc)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

# Suppress non-critical sklearn/LightGBM warnings in inference
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="This Pipeline instance is not fitted yet"
)

# ml/outputs/ klasörü — app/ bir üst dizin olan ml/ altındaki outputs/ klasörüne bakıyor
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_APP_DIR)
ML_OUTPUTS_DIR = os.path.join(_REPO_DIR, "ml", "outputs")
MODELS_DIR = os.path.join(ML_OUTPUTS_DIR, "saved_models")

metrics_df = pd.read_csv(
    os.path.join(ML_OUTPUTS_DIR, "all_models_random_search_results.csv"), sep=";", decimal=","
)
unseen_df = metrics_df[metrics_df["Dataset"] == "Unseen"][["Model", "MaxE"]].copy()
unseen_df["MaxE"] = unseen_df["MaxE"].astype(float)


def _is_loadable(prefix: str) -> bool:
    files = [
        f for f in os.listdir(MODELS_DIR) if f.startswith(f"{prefix}_k") and f.endswith(".pkl")
    ]
    if not files:
        return False
    try:
        joblib.load(os.path.join(MODELS_DIR, files[0]))
        return True
    except Exception as e:
        print(f"[WARN] Cannot load {prefix}: {e}")
        return False


_all_prefixes = unseen_df.sort_values("MaxE")["Model"].unique().tolist()
MODEL_PREFIXES: List[str] = [p for p in _all_prefixes if _is_loadable(p)]


@log_exceptions
def load_model_file(prefix: str) -> Tuple[Any, int]:
    """Load the saved model whose filename starts with *prefix*_k<n>.pkl.

    Returns the model object and the number of features *k* it was trained on.
    """
    files = [
        f for f in os.listdir(MODELS_DIR) if f.startswith(f"{prefix}_k") and f.endswith(".pkl")
    ]
    if not files:
        raise FileNotFoundError(f"No saved model found for prefix '{prefix}'.")

    match = re.search(r"_k(\d+)", files[0])
    k = int(match.group(1)) if match else 10
    model = joblib.load(os.path.join(MODELS_DIR, files[0]))
    return model, k


@log_exceptions
def run_prediction(X: np.ndarray, pipeline: Any) -> float:
    """Return the scalar F_ss prediction.

    The pipeline (feature selection + scaler + model) handles
    all preprocessing internally.
    """
    return float(pipeline.predict(X)[0])


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class StabilityApp(ctk.CTkFrame):
    """Two-tab desktop application for instant F_ss prediction."""

    def __init__(self, master: ctk.CTk) -> None:
        super().__init__(master, fg_color="white")
        self.master = master
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.vars: dict[str, tk.StringVar] = {}
        self.entry_labels: dict[str, ctk.CTkLabel] = {}

        current = read_config("language", "EN")
        self.translations = load_translations(current)
        self.lang_var = tk.StringVar(value=current)
        self.lang_var.trace_add("write", self._on_language_change)

        self._build_ui()
        self.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Language switching
    # ------------------------------------------------------------------

    def _on_language_change(self, *_args) -> None:
        new_lang = self.lang_var.get()
        write_config("language", new_lang)
        self.translations = load_translations(new_lang)
        # Mevcut girdi değerlerini kaydet
        saved_values = {key: var.get() for key, var in self.vars.items()}
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.vars.clear()
        self.entry_labels.clear()
        self._build_ui()
        # Kaydedilen değerleri geri yükle
        for key, value in saved_values.items():
            if key in self.vars and value:
                self.vars[key].set(value)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.master.title(self.translations["title"])

        # Language selector (top-right)
        lang_frame = ctk.CTkFrame(self)
        lang_frame.place(x=950, y=10)
        tk.Label(lang_frame, text=self.translations["language_label"]).pack(side="left", padx=5)
        ttk.Combobox(
            lang_frame,
            textvariable=self.lang_var,
            values=list_languages(),
            width=6,
            state="readonly",
        ).pack(side="left")

        # Title
        ctk.CTkLabel(
            self,
            text=self.translations["title"],
            font=("Helvetica", 24, "bold"),
        ).pack(pady=10)

        # Tabs
        self.tabview = ctk.CTkTabview(self, width=1050, height=900)
        self.tabview.pack(padx=10, pady=10)

        tab_input_label = self.translations["tabs"]["input"]
        tab_model_label = self.translations["tabs"]["model"]
        self.tabview.add(tab_input_label)
        self.tabview.add(tab_model_label)

        self._build_input_tab(self.tabview.tab(tab_input_label))
        self._build_model_tab(self.tabview.tab(tab_model_label), tab_model_label)

    def _build_input_tab(self, parent: tk.Widget) -> None:
        """Build the Input & Visualisation tab."""
        groups = [
            ["k", "h", "xx", "v1", "v2", "x1", "x2", "s1", "x3"],
            ["gama", "fi", "c"],
            ["sds", "hw", "q"],
        ]
        frm = ctk.CTkFrame(parent)
        frm.pack(pady=5, padx=10)

        labels = self.translations["labels"]
        for col_idx, group in enumerate(groups):
            col = ctk.CTkFrame(frm)
            col.grid(row=0, column=col_idx, padx=10, sticky="nw")
            for row_idx, key in enumerate(group):
                var = tk.StringVar(value="0")
                var.trace_add("write", self._redraw)
                self.vars[key] = var

                lbl = ctk.CTkLabel(
                    col,
                    text=f"{key}: {labels.get(key, key)}",
                    font=("Helvetica", 14),
                )
                lbl.grid(row=row_idx, column=0, sticky="w", pady=4)
                self.entry_labels[key] = lbl

                ent = ctk.CTkEntry(col, textvariable=var, width=100, font=("Helvetica", 12))
                ent.grid(row=row_idx, column=1, pady=4, padx=(5, 0))
                self.entries[key] = ent

        self.canvas = tk.Canvas(
            parent,
            bg="white",
            width=800,
            height=500,
            highlightthickness=1,
            highlightbackground="#ccc",
        )
        self.canvas.pack(padx=10, pady=10)

    def _build_model_tab(self, parent: tk.Widget, tab_label: str) -> None:
        """Build the Model Selection tab."""
        ctk.CTkLabel(parent, text=tab_label, font=("Helvetica", 16)).pack(
            anchor="nw", pady=(10, 0), padx=10
        )

        self.lb = tk.Listbox(parent, height=10)
        for prefix in MODEL_PREFIXES:
            maxe = unseen_df[unseen_df["Model"] == prefix]["MaxE"].values[0]
            self.lb.insert("end", f"{prefix} (MaxE={maxe:.4f})")
        self.lb.pack(fill="x", padx=10, pady=5)
        self.lb.bind("<<ListboxSelect>>", self._on_model_select)

        btn_frame = ctk.CTkFrame(parent)
        btn_frame.pack(pady=10)

        self.predict_btn = ctk.CTkButton(
            btn_frame,
            text=self.translations["buttons"]["predict"],
            state="disabled",
            command=self._run_model_predict,
        )
        self.predict_btn.grid(row=0, column=0, padx=5)

        self.info_btn = ctk.CTkButton(
            btn_frame,
            text=self.translations["buttons"]["info"],
            state="disabled",
            command=self._show_model_info,
        )
        self.info_btn.grid(row=0, column=1, padx=5)

        self.bulk_btn = ctk.CTkButton(
            btn_frame,
            text=self.translations["buttons"].get("bulk_predict", "Toplu Tahmin"),
            command=self._run_bulk_predict,
        )
        self.bulk_btn.grid(row=0, column=2, padx=5)

        self.result_label = ctk.CTkLabel(parent, text="", font=("Courier", 16, "bold"))
        self.result_label.pack(pady=5)

        self.detail_label = ctk.CTkLabel(parent, text="", font=("Helvetica", 12), justify="left")
        self.detail_label.pack(pady=(0, 10), padx=10, anchor="w")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_model_select(self, _event) -> None:
        selection = self.lb.curselection()
        if selection:
            self.model_prefix = self.lb.get(selection[0]).split()[0]
            self.predict_btn.configure(state="normal")
            self.info_btn.configure(state="normal")
        else:
            self.predict_btn.configure(state="disabled")
            self.info_btn.configure(state="disabled")

    @log_exceptions
    def _run_model_predict(self) -> None:
        pipeline, _ = load_model_file(self.model_prefix)

        try:
            vals = {k: float(v.get().replace(",", ".")) for k, v in self.vars.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please check all input fields.")
            return

        X = preprocess_inputs(vals)
        prediction = run_prediction(X, pipeline)

        maxe = unseen_df[unseen_df["Model"] == self.model_prefix]["MaxE"].values[0]
        self.result_label.configure(text=f"Predicted F_ss: {prediction:.4f} ± {maxe:.4f}")

        info = MODEL_INFO.get(self.model_prefix, {})
        name = info.get("name", self.model_prefix)
        equation = info.get("equation", "—")
        history = info.get("history", "—")
        parameters = info.get("parameters", {})

        param_lines = "\n".join(f"• {p}: {desc}" for p, desc in parameters.items())
        self.detail_label.configure(
            text=(
                f"{name}\n\n"
                f"Equation:\n  {equation}\n\n"
                f"History:\n  {history}\n\n"
                f"Parameters:\n{param_lines}"
            )
        )

    @log_exceptions
    def _run_bulk_predict(self) -> None:
        try:
            vals = {k: float(v.get().replace(",", ".")) for k, v in self.vars.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please check all input fields.")
            return

        X = preprocess_inputs(vals)
        results = {}
        for prefix in MODEL_PREFIXES:
            try:
                pipeline, _ = load_model_file(prefix)
                results[prefix] = run_prediction(X, pipeline)
            except Exception:
                pass

        if not results:
            messagebox.showinfo("Bulk Predict", "No predictions available.")
            return

        self._draw_number_line(results)

    def _draw_number_line(self, results: dict) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from scipy.stats import gaussian_kde

        maxe_map = {row["Model"]: float(row["MaxE"]) for _, row in unseen_df.iterrows()}

        vals = np.array(list(results.values()))
        prefixes = list(results.keys())
        maxes = np.array([maxe_map.get(p, 0) for p in prefixes])

        # Global axis range
        lo = float((vals - maxes).min())
        hi = float((vals + maxes).max())
        pad = (hi - lo) * 0.08
        x_min, x_max = lo - pad, hi + pad

        # --- Figure ---
        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.patch.set_facecolor("#f8f9fa")
        ax.set_facecolor("#f8f9fa")

        # KDE density curve (all predictions together → overall distribution)
        kde_pts = np.linspace(x_min, x_max, 800)
        if len(vals) > 1:
            kde = gaussian_kde(vals, bw_method=0.3)
            dens = kde(kde_pts)
            dens = dens / dens.max() * 0.38  # normalise to height fraction
            ax.fill_between(kde_pts, 0.55, 0.55 + dens, color="#3498db", alpha=0.18, zorder=1)
            ax.plot(kde_pts, 0.55 + dens, color="#3498db", linewidth=1.2, alpha=0.5, zorder=2)

        # Error boxes — all on y=0.5 band, semi-transparent
        BOX_H = 0.14
        BOX_Y = 0.5 - BOX_H / 2
        for prefix, val in results.items():
            maxe = maxe_map.get(prefix, 0)
            rect = mpatches.FancyBboxPatch(
                (val - maxe, BOX_Y),
                2 * maxe,
                BOX_H,
                boxstyle="round,pad=0.002",
                linewidth=0,
                facecolor="#3498db",
                alpha=0.13,
                zorder=3,
            )
            ax.add_patch(rect)

        # Centre lines (predictions)
        for prefix, val in results.items():
            ax.plot(
                [val, val],
                [BOX_Y - 0.03, BOX_Y + BOX_H + 0.03],
                color="#2c3e50",
                linewidth=1.0,
                alpha=0.55,
                zorder=4,
            )

        # Box plot (compact, on lower band)
        bp = ax.boxplot(
            vals,
            vert=False,
            positions=[0.28],
            widths=[0.10],
            patch_artist=True,
            manage_ticks=False,
            zorder=5,
            boxprops=dict(facecolor="#3498db", alpha=0.45, linewidth=1.2, edgecolor="#2980b9"),
            medianprops=dict(color="#e74c3c", linewidth=2.5),
            whiskerprops=dict(color="#2980b9", linewidth=1.5, linestyle="--"),
            capprops=dict(color="#2980b9", linewidth=2),
            flierprops=dict(marker="o", color="#e74c3c", markersize=5, alpha=0.7),
        )

        # Swarm-style jittered dots
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.03, 0.03, size=len(vals))
        ax.scatter(
            vals, np.full_like(vals, 0.28) + jitter, color="#2c3e50", s=18, alpha=0.6, zorder=6
        )

        # --- Axis (vernier-style ticks) ---
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.10, 1.05)
        ax.set_yticks([])

        # Major ticks every ~0.1 unit, minor every ~0.02
        span = x_max - x_min
        major_step = round(span / 8, 2) or 0.1
        minor_step = major_step / 5

        import matplotlib.ticker as ticker

        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        ax.tick_params(axis="x", which="major", length=8, width=1.2, labelsize=9, color="#333")
        ax.tick_params(axis="x", which="minor", length=4, width=0.8, labelsize=0, color="#555")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.2)

        # Stats annotation
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.set_title(
            f"Bulk Prediction  |  n={len(vals)}  "
            f"min={vals.min():.3f}  Q1={q1:.3f}  "
            f"median={med:.3f}  Q3={q3:.3f}  max={vals.max():.3f}",
            fontsize=10,
            color="#2c3e50",
            pad=8,
        )

        plt.tight_layout(rect=[0, 0, 1, 1])

        # --- Toplevel window with checkbox model selector ---
        win = tk.Toplevel(self.master)
        win.title("Bulk Prediction")
        win.geometry("1100x520")
        win.resizable(True, True)

        # Left panel — model checkboxes
        left = tk.Frame(win, width=160, bg="#f0f0f0")
        left.pack(side="left", fill="y", padx=(6, 0), pady=6)

        tk.Label(left, text="Models", font=("Helvetica", 10, "bold"), bg="#f0f0f0").pack(
            anchor="w", padx=4, pady=(4, 2)
        )

        # Select all / none buttons
        btn_row = tk.Frame(left, bg="#f0f0f0")
        btn_row.pack(fill="x", padx=2, pady=(0, 4))

        chk_vars = {}
        for prefix in sorted(results.keys()):
            var = tk.BooleanVar(value=True)
            chk_vars[prefix] = var

        def _redraw(*_):
            active = {p: v for p, v in results.items() if chk_vars[p].get()}
            if len(active) < 1:
                return
            _refresh_chart(active)

        def _select_all():
            for v in chk_vars.values():
                v.set(True)
            _redraw()

        def _select_none():
            for v in chk_vars.values():
                v.set(False)

        tk.Button(
            btn_row,
            text="All",
            command=_select_all,
            width=5,
            relief="flat",
            bg="#dde",
            font=("Helvetica", 8),
        ).pack(side="left", padx=2)
        tk.Button(
            btn_row,
            text="None",
            command=_select_none,
            width=5,
            relief="flat",
            bg="#dde",
            font=("Helvetica", 8),
        ).pack(side="left", padx=2)

        # Scrollable checkbox list
        list_frame = tk.Frame(left, bg="#f0f0f0")
        list_frame.pack(fill="both", expand=True)
        sb = tk.Scrollbar(list_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        lb_canvas = tk.Canvas(
            list_frame, bg="#f0f0f0", yscrollcommand=sb.set, highlightthickness=0, width=145
        )
        lb_canvas.pack(side="left", fill="both", expand=True)
        sb.config(command=lb_canvas.yview)
        inner = tk.Frame(lb_canvas, bg="#f0f0f0")
        lb_canvas.create_window((0, 0), window=inner, anchor="nw")

        for prefix in sorted(results.keys()):
            maxe = maxe_map.get(prefix, 0)
            tk.Checkbutton(
                inner,
                text=f"{prefix}  ({results[prefix]:.3f})",
                variable=chk_vars[prefix],
                command=_redraw,
                bg="#f0f0f0",
                anchor="w",
                font=("Helvetica", 8),
            ).pack(fill="x", padx=2, pady=1)

        inner.update_idletasks()
        lb_canvas.config(scrollregion=lb_canvas.bbox("all"))

        # Right panel — matplotlib chart
        right = tk.Frame(win, bg="white")
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        chart_holder = [None]  # mutable container for FigureCanvasTkAgg

        def _refresh_chart(active_results):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.patches as mpatches2
            import matplotlib.pyplot as plt2
            import matplotlib.ticker as ticker
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from scipy.stats import gaussian_kde

            if chart_holder[0] is not None:
                chart_holder[0].get_tk_widget().destroy()
                plt2.close("all")

            a_vals = np.array(list(active_results.values()))
            a_prefixes = list(active_results.keys())
            a_maxes = np.array([maxe_map.get(p, 0) for p in a_prefixes])

            a_lo = float((a_vals - a_maxes).min())
            a_hi = float((a_vals + a_maxes).max())
            a_pad = (a_hi - a_lo) * 0.08
            ax_min, ax_max = a_lo - a_pad, a_hi + a_pad

            fig2, ax2 = plt2.subplots(figsize=(9, 4.2))
            fig2.patch.set_facecolor("#f8f9fa")
            ax2.set_facecolor("#f8f9fa")

            kde_pts = np.linspace(ax_min, ax_max, 800)
            if len(a_vals) > 1:
                kde = gaussian_kde(a_vals, bw_method=0.3)
                dens = kde(kde_pts)
                dens = dens / dens.max() * 0.38
                ax2.fill_between(kde_pts, 0.55, 0.55 + dens, color="#3498db", alpha=0.18, zorder=1)
                ax2.plot(kde_pts, 0.55 + dens, color="#3498db", linewidth=1.2, alpha=0.5, zorder=2)

            BOX_H2 = 0.14
            BOX_Y2 = 0.5 - BOX_H2 / 2
            for p, v in active_results.items():
                maxe = maxe_map.get(p, 0)
                rect = mpatches2.FancyBboxPatch(
                    (v - maxe, BOX_Y2),
                    2 * maxe,
                    BOX_H2,
                    boxstyle="round,pad=0.002",
                    linewidth=0,
                    facecolor="#3498db",
                    alpha=0.13,
                    zorder=3,
                )
                ax2.add_patch(rect)

            for p, v in active_results.items():
                ax2.plot(
                    [v, v],
                    [BOX_Y2 - 0.03, BOX_Y2 + BOX_H2 + 0.03],
                    color="#2c3e50",
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=4,
                )

            bp = ax2.boxplot(
                a_vals,
                vert=False,
                positions=[0.28],
                widths=[0.10],
                patch_artist=True,
                manage_ticks=False,
                zorder=5,
                boxprops=dict(facecolor="#3498db", alpha=0.45, linewidth=1.2, edgecolor="#2980b9"),
                medianprops=dict(color="#e74c3c", linewidth=2.5),
                whiskerprops=dict(color="#2980b9", linewidth=1.5, linestyle="--"),
                capprops=dict(color="#2980b9", linewidth=2),
                flierprops=dict(marker="o", color="#e74c3c", markersize=5, alpha=0.7),
            )

            rng2 = np.random.default_rng(42)
            jitter2 = rng2.uniform(-0.03, 0.03, size=len(a_vals))
            ax2.scatter(
                a_vals,
                np.full_like(a_vals, 0.28) + jitter2,
                color="#2c3e50",
                s=18,
                alpha=0.6,
                zorder=6,
            )

            ax2.set_xlim(ax_min, ax_max)
            ax2.set_ylim(0.10, 1.05)
            ax2.set_yticks([])

            a_span = ax_max - ax_min
            maj_step = round(a_span / 8, 2) or 0.1
            min_step = maj_step / 5
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(maj_step))
            ax2.xaxis.set_minor_locator(ticker.MultipleLocator(min_step))
            ax2.tick_params(axis="x", which="major", length=8, width=1.2, labelsize=9, color="#333")
            ax2.tick_params(axis="x", which="minor", length=4, width=0.8, labelsize=0, color="#555")
            ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            for sp in ["top", "left", "right"]:
                ax2.spines[sp].set_visible(False)
            ax2.spines["bottom"].set_linewidth(1.2)

            q1, med, q3 = np.percentile(a_vals, [25, 50, 75])
            ax2.set_title(
                f"n={len(a_vals)}  min={a_vals.min():.3f}  Q1={q1:.3f}  "
                f"median={med:.3f}  Q3={q3:.3f}  max={a_vals.max():.3f}",
                fontsize=10,
                color="#2c3e50",
                pad=8,
            )

            plt2.tight_layout()
            cv = FigureCanvasTkAgg(fig2, master=right)
            cv.draw()
            cv.get_tk_widget().pack(fill="both", expand=True)
            chart_holder[0] = cv
            plt2.close(fig2)

        _refresh_chart(results)
        plt.close(fig)

    @log_exceptions
    def _show_model_info(self) -> None:
        rows = metrics_df[metrics_df["Model"] == self.model_prefix]
        if rows.empty:
            messagebox.showinfo(
                f"{self.model_prefix} Metrics",
                "No results found for this model.",
            )
            return

        cols = [c for c in metrics_df.columns if c != "Model"]
        lines: List[str] = []
        for _, row in rows.iterrows():
            lines.append(f"{row['Dataset']}:")
            for col in cols:
                val = row[col]
                lines.append(
                    f"  {col} = {val:.4f}" if isinstance(val, float) else f"  {col} = {val}"
                )
            lines.append("")

        messagebox.showinfo(f"{self.model_prefix} Metrics", "\n".join(lines))

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------

    def _redraw(self, *_args) -> None:
        self.canvas.delete("all")
        try:
            V = {k: float(v.get().replace(",", ".")) for k, v in self.vars.items()}
        except ValueError:
            return

        k, h, xx = V["k"], V["h"], V["xx"]
        v1, v2 = V["v1"], V["v2"]
        x1, x2 = V["x1"], V["x2"]
        s1, x3 = V["s1"], V["x3"]
        q_val = V.get("q", 0)
        hw_val = V.get("hw", 0)

        # --- Geometry ---
        bottom_body = (h / s1 + k) if s1 > 0 else k
        toe_thick = x1 - xx

        pts = [(0, 0)]
        pts.append((-v1, 0))
        pts.append((-v1, -xx))
        pts.append((-v1 + (v1 + v2 + bottom_body - x2 - x3), -xx))
        pts.append((pts[-1][0], -xx - toe_thick))
        pts.append((pts[-1][0] + x2, pts[-1][1]))
        pts.append((pts[-1][0], pts[-1][1] + toe_thick))
        pts.append((pts[-1][0] + x3, pts[-1][1]))
        pts.append((pts[-1][0], pts[-1][1] + xx))
        pts.append((pts[-1][0] - v2, pts[-1][1]))
        pts.append((pts[-1][0], pts[-1][1] + h))
        pts.append((pts[-1][0] - k, pts[-1][1]))
        pts.append((0, 0))

        H_pt = pts[7]
        L_pt = pts[10]
        B_pt = pts[1]
        yL = L_pt[1]
        dz = yL - H_pt[1]
        Z_pt = (H_pt[0] + dz, yL)
        Y_pt = (L_pt[0], H_pt[1])

        soil = [pts[2], pts[1], (v1 + v2 + bottom_body - v1, 0), (v1 + v2 + bottom_body - v1, -xx)]
        backfill = [H_pt, Z_pt, L_pt, Y_pt]

        # --- Bounding box ---
        xs = [x for x, _ in pts] + [p[0] for p in soil] + [p[0] for p in backfill]
        ys = [y for _, y in pts] + [p[1] for p in soil] + [p[1] for p in backfill]

        arrow_model_h = q_val / 10
        if arrow_model_h > 0:
            ys.append(max(ys) + arrow_model_h)

        if hw_val > 0:
            K_pt = pts[10]
            GW1 = (K_pt[0], K_pt[1] - hw_val)
            GW2 = (Z_pt[0], Z_pt[1] - hw_val)
            xs += [GW1[0], GW2[0]]
            ys += [GW1[1], GW2[1]]
            shape_depth = h + x1
            if hw_val > shape_depth:
                extra = hw_val - shape_depth
                shape_bottom = min(
                    [p[1] for p in pts] + [p[1] for p in soil] + [p[1] for p in backfill]
                )
                ys.append(shape_bottom - extra)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        model_w = max_x - min_x
        model_h = max_y - min_y
        M = 50
        scale = min(
            (800 - 2 * M) / model_w if model_w > 0 else float("inf"),
            (250 - 2 * M) / model_h if model_h > 0 else float("inf"),
        )
        if scale == float("inf"):
            scale = 1
        tx = -min_x * scale + M
        ty = max_y * scale + M

        def to_px(pt):
            return pt[0] * scale + tx, -pt[1] * scale + ty

        # --- Draw polygons ---
        self.canvas.create_polygon(*[to_px(p) for p in soil], fill="#FAFAB3", outline="#7f8c8d")
        self.canvas.create_polygon(
            *[c for p in backfill for c in to_px(p)], fill="#ecf0f1", outline="#7f8c8d"
        )
        self.canvas.create_polygon(
            *[c for p in pts for c in to_px(p)], fill="#bdc3c7", outline="#333", width=2
        )

        # --- Reference lines ---
        H_px, Z_px, L_px, Y_px, B_px = map(to_px, [H_pt, Z_pt, L_pt, Y_pt, B_pt])
        self.canvas.create_line(L_px[0], L_px[1], 800, L_px[1], fill="#333", dash=(4, 2))
        self.canvas.create_line(B_px[0], B_px[1], 0, B_px[1], fill="#333", dash=(4, 2))
        for a, b in [(H_px, Z_px), (Z_px, L_px), (L_px, Y_px), (Y_px, H_px)]:
            self.canvas.create_line(*a, *b, fill="#34495e", dash=(4, 2))

        # --- Surcharge arrows ---
        if q_val > 0:
            K_pt = pts[10]
            K_px = to_px(K_pt)
            Z_px = to_px(Z_pt)
            self.canvas.create_line(*K_px, *Z_px, fill="#e74c3c", width=2)

            n_arrows = max(2, int(q_val / 10))
            arrow_len = arrow_model_h * scale
            tail_pts = []
            for i in range(n_arrows):
                t = i / (n_arrows - 1) if n_arrows > 1 else 0.5
                x_px = K_px[0] + (Z_px[0] - K_px[0]) * t
                y_px = K_px[1] + (Z_px[1] - K_px[1]) * t
                y_top = y_px - arrow_len
                tail_pts.append((x_px, y_top))
                self.canvas.create_line(
                    x_px, y_top, x_px, y_px, arrow=tk.LAST, arrowshape=(8, 10, 4), fill="#e74c3c"
                )
            if tail_pts:
                xs_t = [p[0] for p in tail_pts]
                ytop = tail_pts[0][1]
                # Surcharge top line extends to canvas right edge (like ground surface)
                self.canvas.create_line(min(xs_t), ytop, 800, ytop, fill="#e74c3c")

        # --- Groundwater line ---
        if hw_val > 0:
            K_pt = pts[10]
            GW1_px = to_px((K_pt[0], K_pt[1] - hw_val))
            GW2_px = to_px((Z_pt[0], Z_pt[1] - hw_val))
            # If hw > wall height, GW is below wall base → extend left to canvas edge
            wall_height = h + x1
            gw_x_start = 0 if hw_val > wall_height else GW1_px[0]
            self.canvas.create_line(
                gw_x_start, GW1_px[1], 800, GW2_px[1], fill="blue", width=2, dash=(4, 2)
            )
