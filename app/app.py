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
import warnings
from logging.handlers import RotatingFileHandler
from typing import Any, List, Tuple

import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd

from config import read_config, write_config, resource_path
from language import list_languages, load_translations
from model_info import MODEL_INFO
from preprocessing import preprocess_inputs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = os.environ.get("LOG_FILE", "app.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
))
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

MODELS_DIR = os.path.abspath("saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

metrics_df = pd.read_csv(resource_path("all_models_random_search_results.csv"), sep=";", decimal=",")
unseen_df  = metrics_df[metrics_df["Dataset"] == "Unseen"][["Model", "MaxE"]].copy()
unseen_df["MaxE"] = unseen_df["MaxE"].astype(float)
MODEL_PREFIXES: List[str] = unseen_df.sort_values("MaxE")["Model"].unique().tolist()


@log_exceptions
def load_model_file(prefix: str) -> Tuple[Any, int]:
    """Load the saved model whose filename starts with *prefix*_k<n>.pkl.

    Returns the model object and the number of features *k* it was trained on.
    """
    files = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(f"{prefix}_k") and f.endswith(".pkl")
    ]
    if not files:
        raise FileNotFoundError(f"No saved model found for prefix '{prefix}'.")

    match = re.search(r"_k(\d+)", files[0])
    k = int(match.group(1)) if match else 10
    model = joblib.load(resource_path(os.path.join("saved_models", files[0])))
    return model, k


@log_exceptions
def run_prediction(inputs: List[float], model: Any, k: int) -> float:
    """Return the scalar F_ss prediction for *inputs*."""
    return float(model.predict(np.array([inputs[:k]]))[0])

# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class StabilityApp(ctk.CTkFrame):
    """Two-tab desktop application for instant F_ss prediction."""

    def __init__(self, master: ctk.CTk) -> None:
        super().__init__(master, fg_color="white")
        self.master = master
        self.entries:      dict[str, ctk.CTkEntry]  = {}
        self.vars:         dict[str, tk.StringVar]   = {}
        self.entry_labels: dict[str, ctk.CTkLabel]  = {}

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
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.vars.clear()
        self.entry_labels.clear()
        self._build_ui()

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
            parent, bg="white", width=800, height=500,
            highlightthickness=1, highlightbackground="#ccc",
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

        self.result_label = ctk.CTkLabel(parent, text="", font=("Courier", 16, "bold"))
        self.result_label.pack(pady=5)

        self.detail_label = ctk.CTkLabel(
            parent, text="", font=("Helvetica", 12), justify="left"
        )
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
        model_obj, model_k = load_model_file(self.model_prefix)

        try:
            vals = {k: float(v.get().replace(",", ".")) for k, v in self.vars.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please check all input fields.")
            return

        inputs = preprocess_inputs(vals, self.model_prefix, model_k)
        prediction = run_prediction(inputs, model_obj, model_k)

        maxe = unseen_df[unseen_df["Model"] == self.model_prefix]["MaxE"].values[0]
        self.result_label.configure(text=f"Predicted F_ss: {prediction:.4f} ± {maxe:.4f}")

        info = MODEL_INFO.get(self.model_prefix, {})
        name       = info.get("name", self.model_prefix)
        equation   = info.get("equation", "—")
        history    = info.get("history", "—")
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
                lines.append(f"  {col} = {val:.4f}" if isinstance(val, float) else f"  {col} = {val}")
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
        v1, v2   = V["v1"], V["v2"]
        x1, x2   = V["x1"], V["x2"]
        s1, x3   = V["s1"], V["x3"]
        q_val    = V.get("q", 0)
        hw_val   = V.get("hw", 0)

        # --- Geometry ---
        bottom_body = (h / s1 + k) if s1 > 0 else k
        toe_thick   = x1 - xx

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
        yL   = L_pt[1]
        dz   = yL - H_pt[1]
        Z_pt = (H_pt[0] + dz, yL)
        Y_pt = (L_pt[0], H_pt[1])

        soil     = [pts[2], pts[1], (v1 + v2 + bottom_body - v1, 0), (v1 + v2 + bottom_body - v1, -xx)]
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
            shape_depth  = h + x1
            if hw_val > shape_depth:
                extra        = hw_val - shape_depth
                shape_bottom = min([p[1] for p in pts] + [p[1] for p in soil] + [p[1] for p in backfill])
                ys.append(shape_bottom - extra)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        model_w = max_x - min_x
        model_h = max_y - min_y
        M = 50
        scale = min(
            (800 - 2 * M) / model_w if model_w > 0 else float("inf"),
            (500 - 2 * M) / model_h if model_h > 0 else float("inf"),
        )
        if scale == float("inf"):
            scale = 1
        tx = -min_x * scale + M
        ty =  max_y * scale + M

        def to_px(pt):
            return pt[0] * scale + tx, -pt[1] * scale + ty

        # --- Draw polygons ---
        self.canvas.create_polygon(*[to_px(p) for p in soil],     fill="#FAFAB3", outline="#7f8c8d")
        self.canvas.create_polygon(*[c for p in backfill for c in to_px(p)], fill="#ecf0f1", outline="#7f8c8d")
        self.canvas.create_polygon(*[c for p in pts     for c in to_px(p)], fill="#bdc3c7", outline="#333", width=2)

        # --- Reference lines ---
        H_px, Z_px, L_px, Y_px, B_px = map(to_px, [H_pt, Z_pt, L_pt, Y_pt, B_pt])
        self.canvas.create_line(L_px[0], L_px[1], 800, L_px[1], fill="#333", dash=(4, 2))
        self.canvas.create_line(B_px[0], B_px[1],   0, B_px[1], fill="#333", dash=(4, 2))
        for a, b in [(H_px, Z_px), (Z_px, L_px), (L_px, Y_px), (Y_px, H_px)]:
            self.canvas.create_line(*a, *b, fill="#34495e", dash=(4, 2))

        # --- Surcharge arrows ---
        if q_val > 0:
            K_pt = pts[10]
            K_px = to_px(K_pt)
            Z_px = to_px(Z_pt)
            self.canvas.create_line(*K_px, *Z_px, fill="#e74c3c", width=2)

            n_arrows  = max(2, int(q_val / 10))
            arrow_len = arrow_model_h * scale
            tail_pts  = []
            for i in range(n_arrows):
                t     = i / (n_arrows - 1) if n_arrows > 1 else 0.5
                x_px  = K_px[0] + (Z_px[0] - K_px[0]) * t
                y_px  = K_px[1] + (Z_px[1] - K_px[1]) * t
                y_top = y_px - arrow_len
                tail_pts.append((x_px, y_top))
                self.canvas.create_line(x_px, y_top, x_px, y_px,
                                        arrow=tk.LAST, arrowshape=(8, 10, 4), fill="#e74c3c")
            if tail_pts:
                xs_t = [p[0] for p in tail_pts]
                ytop = tail_pts[0][1]
                self.canvas.create_line(min(xs_t), ytop, max(xs_t), ytop, fill="#e74c3c")

        # --- Groundwater line ---
        if hw_val > 0:
            K_pt = pts[10]
            GW1_px = to_px((K_pt[0], K_pt[1] - hw_val))
            GW2_px = to_px((Z_pt[0], Z_pt[1] - hw_val))
            self.canvas.create_line(*GW1_px, *GW2_px, fill="blue", width=2, dash=(4, 2))
