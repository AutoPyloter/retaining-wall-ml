"""
generate_figures.py
-------------------
Retaining Wall ML — SoftwareX makale görselleri ve tablo LaTeX kodu

Çalıştırma:
    cd figs/
    python generate_figures.py [--csv PATH] [--shap_values PATH] [--outdir PATH]

Argümanlar:
    --csv          all_models_random_search_results.csv yolu
                   (varsayılan: ../ml/outputs/all_models_random_search_results.csv)
    --shap_values  SHAP mean absolute values CSV yolu — önerilen yöntem
                   format: feature,shap_value (train_models.py çıktısı)
                   (varsayılan: ../ml/outputs/shap_mean_abs_values.csv)
    --shap         shap_bar.png yolu — shap_values yoksa fallback olarak kullanılır
                   (varsayılan: ../ml/outputs/plots/shap_bar.png)
    --shap_summary shap_summary.png yolu
                   (varsayılan: ../ml/outputs/plots/shap_summary.png)
    --outdir       Çıktı klasörü (varsayılan: ./output/)

Üretilen dosyalar:
    fig1_architecture.pdf     — Yazılım mimarisi flowchart
    fig4a_shap_bar.pdf        — SHAP bar plot (okunabilir feature isimleriyle)
    fig4b_shap_summary.pdf    — SHAP beeswarm (yeniden formatlanmış)
    fig5_model_comparison.pdf — 35 model MaxE karşılaştırması
    table2_design_space.tex   — Dataset tasarım uzayı LaTeX tablosu
    table3_top_models.tex     — En iyi 5 model metrikleri LaTeX tablosu
"""

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SoftwareX / Elsevier stil sabitleri
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 9
LABEL_SIZE = 8
TICK_SIZE = 7.5
LEGEND_SIZE = 7.5
FIG_DPI = 300
SINGLE_COL_W = 3.46  # inch  (88 mm — Elsevier single column)
DOUBLE_COL_W = 7.08  # inch  (180 mm — Elsevier double column)

# Renk paleti (renk körü dostu)
C_TEAL = "#1D9E75"
C_PURPLE = "#7F77DD"
C_AMBER = "#BA7517"
C_GRAY = "#888780"
C_CORAL = "#D85A30"
C_BLUE = "#185FA5"
C_GREEN = "#3B6D11"
C_RED = "#A32D2D"

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#dddddd",
        "lines.linewidth": 1.2,
    }
)

# ---------------------------------------------------------------------------
# Argüman ayrıştırma
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Generate SoftwareX paper figures")
    p.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__), "..", "ml", "outputs", "all_models_random_search_results.csv"
        ),
    )
    p.add_argument(
        "--shap",
        default=os.path.join(
            os.path.dirname(__file__), "..", "ml", "outputs", "plots", "shap_bar.png"
        ),
    )
    p.add_argument(
        "--shap_summary",
        default=os.path.join(
            os.path.dirname(__file__), "..", "ml", "outputs", "plots", "shap_summary.png"
        ),
    )
    p.add_argument(
        "--shap_values",
        default=os.path.join(
            os.path.dirname(__file__), "..", "ml", "outputs", "shap_mean_abs_values.csv"
        ),
        help="CSV with columns [feature, shap_value] — produced by train_models.py",
    )
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "output"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Yardımcı: kaydet
# ---------------------------------------------------------------------------


def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ===========================================================================
# FIG 1 — Yazılım mimarisi flowchart
# ===========================================================================


def fig1_architecture(outdir):
    """
    Üç katman:
      Katman 1: Dataset  (>2,000 scenarios)
      Katman 2: ML module  (split → SHAP → train → evaluate → serialise)
      Katman 3: App  (preprocess → pipeline → visualise)
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ------ kutu çizici ------
    def box(cx, cy, w, h, label, sublabel=None, fc="#E1F5EE", ec=C_TEAL, lc=C_TEAL, fontsize=7.5):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.06",
            fc=fc,
            ec=ec,
            lw=0.8,
            zorder=3,
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(
                cx,
                cy + 0.13,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=lc,
                zorder=4,
            )
            ax.text(
                cx,
                cy - 0.15,
                sublabel,
                ha="center",
                va="center",
                fontsize=6.5,
                color=C_GRAY,
                zorder=4,
            )
        else:
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=lc,
                zorder=4,
            )

    def arrow(x1, y1, x2, y2, color=C_GRAY, lw=0.9):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7),
            zorder=2,
        )

    def hline(y, color=C_GRAY, ls="--"):
        ax.axhline(y, color=color, lw=0.5, ls=ls, zorder=1, alpha=0.5)

    # ---- Arka plan şeritleri ----
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.15, 3.25), 9.7, 0.55, boxstyle="round,pad=0.0", fc="#F0FDF4", ec="none", zorder=0
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.15, 1.55), 9.7, 1.55, boxstyle="round,pad=0.0", fc="#F5F4FE", ec="none", zorder=0
        )
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.15, 0.15), 9.7, 1.25, boxstyle="round,pad=0.0", fc="#FFF8EE", ec="none", zorder=0
        )
    )

    # Şerit etiketleri (sol kenar)
    ax.text(
        0.28, 3.52, "Data", fontsize=7, color=C_TEAL, fontweight="bold", va="center", rotation=90
    )
    ax.text(
        0.28,
        2.32,
        "ML module",
        fontsize=7,
        color=C_PURPLE,
        fontweight="bold",
        va="center",
        rotation=90,
    )
    ax.text(
        0.28,
        0.77,
        "Application",
        fontsize=7,
        color=C_AMBER,
        fontweight="bold",
        va="center",
        rotation=90,
    )

    # ---- Katman 1: Dataset ----
    box(
        5.0,
        3.52,
        2.6,
        0.42,
        "Dataset  (>2\u202f000 scenarios)",
        "18 inputs (13 sampled\u202f+\u202f5 derived) \u00b7 F\u209bS per row",
        fc="#D6F5E9",
        ec=C_TEAL,
        lc=C_TEAL,
    )

    # ---- Katman 2: ML module ----
    ml_boxes = [
        (1.3, 2.32, "split_dataset.py", "70/20/10 split"),
        (3.1, 2.32, "XGBoost\ngrid search", "256 combos"),
        (5.0, 2.32, "SHAP ranking", "18 features"),
        (6.9, 2.32, "Randomized\nSearchCV", "35 models"),
        (8.8, 2.32, "saved_models/", "*.pkl pipelines"),
    ]
    fc_ml = "#EEEDFE"
    ec_ml = C_PURPLE
    lc_ml = C_PURPLE
    for cx, cy, lbl, sub in ml_boxes:
        box(cx, cy, 1.5, 0.60, lbl, sub, fc=fc_ml, ec=ec_ml, lc=lc_ml)

    # ML okları
    for i in range(len(ml_boxes) - 1):
        x1 = ml_boxes[i][0] + 0.75
        x2 = ml_boxes[i + 1][0] - 0.75
        arrow(x1, 2.32, x2, 2.32, color=C_PURPLE)

    # Dataset → split_dataset.py
    arrow(5.0, 3.30, 1.3, 2.62, color=C_TEAL)

    # ---- Katman 3: App ----
    app_boxes = [
        (2.2, 0.77, "preprocessing.py", "15 inputs → feature vector"),
        (5.0, 0.77, "Pipeline.predict()", "select → scale → model"),
        (7.8, 0.77, "StabilityApp", "prediction + uncertainty"),
    ]
    fc_ap = "#FFF0D6"
    ec_ap = C_AMBER
    lc_ap = C_AMBER
    for cx, cy, lbl, sub in app_boxes:
        box(cx, cy, 2.2, 0.58, lbl, sub, fc=fc_ap, ec=ec_ap, lc=lc_ap)

    arrow(3.3, 0.77, 3.9, 0.77, color=C_AMBER)
    arrow(6.1, 0.77, 6.7, 0.77, color=C_AMBER)

    # saved_models → pipeline
    arrow(8.8, 2.02, 5.0, 1.06, color=C_GRAY, lw=0.8)

    # Stil
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0.3)
    savefig(fig, outdir, "fig1_architecture.pdf")


# ===========================================================================
# FIG 4 — SHAP görselleri (mevcut PNG'yi yeniden çerçevele)
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature name map: CSV column → human-readable label
# ---------------------------------------------------------------------------
FEATURE_NAMES = {
    "H": "Wall height",
    "X1": "Foundation total width",
    "X2": "Front overhang",
    "X3": "Stem bottom width",
    "X4": "Stem top width",
    "X5": "Foundation thickness",
    "X6": "Key thickness",
    "X7": "Key width",
    "X8": "Key offset from heel",
    "q": "Surcharge load",
    "sds": "Spectral acceleration",
    "v2": "Rear overhang",
    "x1": "Foundation + key thickness",
    "s1": "Wall batter slope",
    "gama": "Soil unit weight  γ",
    "c": "Cohesion  c",
    "fi": "Friction angle  φ",
    "hw": "Groundwater level index",
}


# ===========================================================================
# FIG 4 — SHAP bar plot (sıfırdan üretim veya PNG fallback)
# ===========================================================================


def fig4_shap(shap_bar_path, shap_sum_path, outdir, shap_values_csv=None):
    """
    Öncelik: shap_values_csv varsa matplotlib ile sıfırdan çizer
    (feature isimleri FEATURE_NAMES ile Türkçe/İngilizce okunabilir hale gelir).
    Yoksa eski shap_bar.png'yi yeniden çerçeveler.
    shap_summary.png her iki durumda da aynı şekilde çerçevelenir.
    """

    # ── 4a: Bar plot ────────────────────────────────────────────────────────
    if shap_values_csv and os.path.isfile(shap_values_csv):
        _fig4a_from_csv(shap_values_csv, outdir)
    else:
        _fig4a_from_png(shap_bar_path, outdir)

    # ── 4b: Beeswarm / summary (her zaman PNG çerçeveleme) ─────────────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.2))
    if os.path.isfile(shap_sum_path):
        from PIL import Image

        img = Image.open(shap_sum_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("SHAP summary (beeswarm)", fontsize=TITLE_SIZE, pad=4)
    else:
        ax.text(
            0.5,
            0.5,
            f"[Placeholder]\n{os.path.basename(shap_sum_path)}\nnot found",
            ha="center",
            va="center",
            fontsize=8,
            color=C_GRAY,
            transform=ax.transAxes,
        )
        ax.set_title("SHAP summary (beeswarm)", fontsize=TITLE_SIZE, pad=4)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    savefig(fig, outdir, "fig4b_shap_summary.pdf")


def _fig4a_from_csv(csv_path, outdir):
    """
    shap_mean_abs_values.csv formatı (train_models.py çıktısı):
        feature,shap_value
        H,0.312
        sds,0.278
        ...
    Sütun adları farklıysa ilk iki sütun alınır.
    """
    df = pd.read_csv(csv_path)
    # Sütun adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]
    feat_col = df.columns[0]
    val_col = df.columns[1]

    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col]).sort_values(val_col, ascending=True)

    # Feature isimlerini okunabilir hale getir
    df["label"] = df[feat_col].map(lambda x: FEATURE_NAMES.get(str(x), str(x)))

    n = len(df)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, max(2.8, n * 0.28 + 0.6)))

    colors = [C_TEAL if v >= df[val_col].quantile(0.5) else C_GRAY for v in df[val_col].values]

    bars = ax.barh(range(n), df[val_col].values, color=colors, height=0.72, edgecolor="none")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["label"].values, fontsize=7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=LABEL_SIZE)
    ax.set_title(
        "(a) Feature importance — XGBoost baseline", fontsize=TITLE_SIZE, pad=4, loc="left"
    )
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    # En yüksek 3'ü etiketle
    top3 = df[val_col].nlargest(3).index.tolist()
    idx_map = {idx: i for i, idx in enumerate(df.index)}
    for idx in top3:
        i = idx_map[idx]
        val = df.loc[idx, val_col]
        ax.text(
            val + df[val_col].max() * 0.01, i, f"{val:.3f}", va="center", fontsize=6, color="#333"
        )

    fig.tight_layout(pad=0.4)
    savefig(fig, outdir, "fig4a_shap_bar.pdf")
    print(f"    → {n} features plotted with readable names")


def _fig4a_from_png(shap_bar_path, outdir):
    """Eski yöntem: mevcut PNG'yi çerçevele."""
    from PIL import Image

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 3.2))
    if os.path.isfile(shap_bar_path):
        img = Image.open(shap_bar_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Mean |SHAP value|", fontsize=TITLE_SIZE, pad=4)
    else:
        ax.text(
            0.5,
            0.5,
            f"[Placeholder]\n{os.path.basename(shap_bar_path)}\nnot found\n\n"
            f"Pass --shap_values PATH to generate from CSV instead.",
            ha="center",
            va="center",
            fontsize=7.5,
            color=C_GRAY,
            transform=ax.transAxes,
        )
        ax.set_title("Mean |SHAP value|", fontsize=TITLE_SIZE, pad=4)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    savefig(fig, outdir, "fig4a_shap_bar.pdf")


# ===========================================================================
# FIG 5 — 35 model MaxE karşılaştırması
# ===========================================================================

# Model aile renk haritası
FAMILY_COLORS = {
    "Linear": C_BLUE,
    "GLM": C_GREEN,
    "Kernel": C_PURPLE,
    "Neural": C_CORAL,
    "Tree": C_TEAL,
    "Boosting": C_AMBER,
    "Ensemble": C_RED,
    "Other": C_GRAY,
}

MODEL_FAMILY = {
    "OLS": "Linear",
    "Ridge": "Linear",
    "Lasso": "Linear",
    "Elastic": "Linear",
    "Bayesian": "Linear",
    "ARD": "Linear",
    "Huber": "Linear",
    "RANSAC": "Linear",
    "TheilSen": "Linear",
    "OMP": "Linear",
    "PA": "Linear",
    "PLS": "Other",
    "Quantile": "GLM",
    "Poisson": "GLM",
    "Tweedie": "GLM",
    "Gamma": "GLM",
    "SVM": "Kernel",
    "kNN": "Kernel",
    "KR": "Kernel",
    "GPR": "Kernel",
    "MLP": "Neural",
    "PolyR": "Other",
    "DT": "Tree",
    "AdaBoost": "Tree",
    "RF": "Tree",
    "ExtraTrees": "Tree",
    "ET": "Tree",
    "GBDT": "Boosting",
    "HGB": "Boosting",
    "XGBoost": "Boosting",
    "XGBoost_RF": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
    "CAT": "Boosting",
    "NGBoost": "Boosting",
    "Stack": "Ensemble",
    "Voting": "Ensemble",
}


def fig5_model_comparison(csv_path, outdir):
    if not os.path.isfile(csv_path):
        print(f"  [WARN] CSV not found: {csv_path} — fig5 placeholder üretiliyor")
        _fig5_placeholder(outdir)
        return

    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # Her model için Unseen MaxE al
    unseen = df[df["Dataset"] == "Unseen"][["Model", "MaxE", "R2", "RMSE"]].copy()
    unseen["MaxE"] = pd.to_numeric(unseen["MaxE"], errors="coerce")
    unseen["R2"] = pd.to_numeric(unseen["R2"], errors="coerce")
    unseen["RMSE"] = pd.to_numeric(unseen["RMSE"], errors="coerce")
    unseen = unseen.dropna(subset=["MaxE"]).sort_values("MaxE")

    # Aile ataması
    unseen["Family"] = unseen["Model"].map(lambda m: MODEL_FAMILY.get(m, "Other"))
    unseen["Color"] = unseen["Family"].map(lambda f: FAMILY_COLORS.get(f, C_GRAY))

    n = len(unseen)
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_W, max(3.0, n * 0.22 + 1.0)), gridspec_kw={"width_ratios": [3, 1]}
    )

    # ── Sol panel: MaxE yatay bar ──────────────────────────────────
    ax = axes[0]
    y = np.arange(n)
    bars = ax.barh(
        y, unseen["MaxE"].values, color=unseen["Color"].values, height=0.72, edgecolor="none"
    )

    # En iyi modeli (en küçük MaxE) vurgula
    best_idx = 0  # zaten sıralı
    bars[best_idx].set_edgecolor("#222")
    bars[best_idx].set_linewidth(0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(unseen["Model"].values, fontsize=6.8)
    ax.set_xlabel("MaxE on unseen set", fontsize=LABEL_SIZE)
    ax.set_title("(a) Max absolute error (unseen)", fontsize=TITLE_SIZE, pad=4, loc="left")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    # Değer etiketleri — alt %30'a giren modeller (veri odaklı eşik)
    threshold = unseen["MaxE"].quantile(0.30)
    for i, (val, bar) in enumerate(zip(unseen["MaxE"].values, bars)):
        if val <= threshold:
            ax.text(
                val + unseen["MaxE"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=5.5,
                color="#333",
            )

    # ── Sağ panel: R² nokta grafiği ───────────────────────────────
    ax2 = axes[1]
    ax2.scatter(unseen["R2"].values, y, c=unseen["Color"].values, s=14, zorder=3, linewidths=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("R² (unseen)", fontsize=LABEL_SIZE)
    ax2.set_title("(b) R²", fontsize=TITLE_SIZE, pad=4, loc="left")
    ax2.set_xlim(max(0, unseen["R2"].min() - 0.05), 1.02)
    ax2.axvline(1.0, color=C_GRAY, lw=0.5, ls="--")
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=col, label=fam)
        for fam, col in FAMILY_COLORS.items()
        if fam in unseen["Family"].values
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        fontsize=6.5,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    fig.tight_layout(pad=0.4, rect=[0, 0.06, 1, 1])
    savefig(fig, outdir, "fig5_model_comparison.pdf")


def _fig5_placeholder(outdir):
    """CSV yokken yer tutucu grafik."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.0))
    ax.text(
        0.5,
        0.5,
        "[Placeholder]\nall_models_random_search_results.csv\nnot found",
        ha="center",
        va="center",
        fontsize=9,
        color=C_GRAY,
        transform=ax.transAxes,
    )
    ax.axis("off")
    savefig(fig, outdir, "fig5_model_comparison.pdf")


# ===========================================================================
# TABLO 2 — Dataset tasarım uzayı
# ===========================================================================

TABLE2_ROWS = [
    # (Sembol, Açıklama, Alt sınır, Üst sınır, Adım/Notlar)
    (r"$H$", "Wall height (m)", "4", "10", "1"),
    (r"$X_1$", "Foundation total width (m)", r"$0.3H$", "10.0", r"$0.05H$"),
    (r"$X_2$", "Front overhang (m)", r"$0.15X_1$", r"$0.45X_1$", r"$0.05X_1$"),
    (r"$X_3$", "Stem bottom width (m)", "0.3", "0.6", "0.05"),
    (r"$X_4$", "Stem top width (m)", "0.3", r"$X_3$", "0.05"),
    (r"$X_5$", "Foundation thickness (m)", r"$0.06H$", r"$0.18H$", r"$0.01H$"),
    (r"$X_6$", "Key thickness (m)", "0", r"$1.2X_5$", r"$0.05X_5$"),
    (r"$X_7$", "Key width (m)", "0", r"$0.30X_1$", r"$0.05X_1$"),
    (r"$X_8$", "Key offset from heel (m)", "0", r"$0.70X_1$", r"$0.05X_1$"),
    (r"$q$", "Surcharge load (kN/m$^2$)", "0", "20", "5"),
    (r"$s_{DS}$", "Spectral acceleration (g)", "0.6", "1.8", "0.1"),
    (r"$v_2$", "Rear overhang (m)", "---", "---", "geometric"),
    (r"$x_1$", "$X_5 + X_6$ (m)", "---", "---", "derived"),
    (r"$s_1$", "Wall batter slope", "---", "---", "geometric"),
    (r"$\gamma$", "Soil unit weight (kN/m$^3$)", "17", "20", "numerical"),
    (r"$c$", "Cohesion (kPa)", "0", "40", "numerical"),
    (r"$\varphi$", "Friction angle ($^\circ$)", "20", "40", "numerical"),
    (r"$h_w$", "Water level index (0--4)", "0", "4", "1"),
]


def table2_design_space(outdir):
    lines = []
    lines.append(r"% TABLE 2 — Design space (auto-generated by generate_figures.py)")
    lines.append(r"\begin{table}[!ht]")
    lines.append(
        r"\caption{Design space of the 2{,}048-scenario dataset. "
        r"All geometric parameters are in metres unless stated otherwise. "
        r"Dependent bounds are evaluated after sampling the parent "
        r"parameter, ensuring geometric consistency.}"
    )
    lines.append(r"\label{tab:design_space}")
    lines.append(r"\begin{tabular*}{\tblwidth}{@{} L L L L L @{}}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Description & Min & Max & Step/Notes \\")
    lines.append(r"\midrule")
    for sym, desc, lo, hi, step in TABLE2_ROWS:
        lines.append(f"{sym} & {desc} & {lo} & {hi} & {step} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "table2_design_space.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {path}")


# ===========================================================================
# TABLO 3 — En iyi 5 model metrikleri
# ===========================================================================

METRICS_SHOW = ["MAE", "RMSE", "R2", "MaxE", "NSE", "KGE"]


def table3_top_models(csv_path, outdir, top_n=5):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "table3_top_models.tex")

    if not os.path.isfile(csv_path):
        # Yer tutucu
        with open(path, "w") as f:
            f.write("% CSV not found — placeholder\n")
            f.write(r"\begin{table}[!ht]" + "\n")
            f.write(r"\caption{Top-5 model performance (placeholder).}" + "\n")
            f.write(r"\label{tab:top_models}" + "\n")
            f.write(r"\begin{tabular*}{\tblwidth}{@{} l l @{}}" + "\n")
            f.write(r"\toprule Model & MaxE \\ \midrule" + "\n")
            f.write(r"— & — \\" + "\n")
            f.write(r"\bottomrule\end{tabular*}\end{table}" + "\n")
        print(f"  [WARN] {path} (placeholder — CSV not found)")
        return

    df = pd.read_csv(csv_path, sep=";", decimal=",")
    for col in METRICS_SHOW + ["MaxE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    unseen = df[df["Dataset"] == "Unseen"].copy()
    unseen = unseen.sort_values("MaxE").head(top_n)
    top_models = unseen["Model"].tolist()

    # Tüm split'ler için bu modellerin metriklerini al
    subset = df[df["Model"].isin(top_models)].copy()

    lines = []
    lines.append(r"% TABLE 3 — Top-5 model performance (auto-generated)")
    lines.append(r"\begin{table*}[!ht]")
    lines.append(
        r"\caption{Performance of the five best-ranking models "
        r"(ranked by MaxE on the unseen hold-out set) across all "
        r"three evaluation splits. "
        r"The best value in each column is shown in bold.}"
    )
    lines.append(r"\label{tab:top_models}")

    # Sütun sayısı: Model + Dataset + len(METRICS_SHOW)
    n_cols = 2 + len(METRICS_SHOW)
    col_fmt = "@{} L L " + " ".join(["R"] * len(METRICS_SHOW)) + " @{}"
    lines.append(r"\begin{tabular*}{\tblwidth}{" + col_fmt + "}")
    lines.append(r"\toprule")

    header_metrics = " & ".join(
        m.replace("R2", r"$R^2$").replace("CV(RMSE)%", r"CV(RMSE)\%") for m in METRICS_SHOW
    )
    lines.append(r"Model & Split & " + header_metrics + r" \\")
    lines.append(r"\midrule")

    # Her model sırası için en iyi değerleri bul (bold için)
    best = {}
    for m in METRICS_SHOW:
        if m not in subset.columns:
            continue
        if m in {"R2", "NSE", "KGE", "EVS", "CCC"}:
            best[m] = subset[m].max()  # büyük iyidir
        else:
            best[m] = subset[m].min()  # küçük iyidir

    splits = ["Train", "Test", "Unseen"]

    for model in top_models:
        mrows = subset[subset["Model"] == model]
        first = True
        for split in splits:
            row = mrows[mrows["Dataset"] == split]
            if row.empty:
                continue
            row = row.iloc[0]
            cells = []
            for m in METRICS_SHOW:
                if m not in row.index:
                    cells.append("—")
                    continue
                val = row[m]
                if pd.isna(val):
                    cells.append("—")
                    continue
                fmt = f"{val:.4f}"
                if abs(val - best.get(m, np.nan)) < 1e-9:
                    fmt = r"\textbf{" + fmt + "}"
                cells.append(fmt)

            model_cell = model if first else ""
            lines.append(f"{model_cell} & {split} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")

    # Son \midrule'ü kaldır
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {path}")


# ===========================================================================
# Metadata tablosu — SoftwareX zorunlu formatı
# ===========================================================================


def table_metadata(outdir):
    """
    SoftwareX her makale için bir metadata tablosu zorunlu kılar.
    Bu fonksiyon doğrudan LaTeX'e gömülebilecek bir .tex dosyası üretir.
    Değerleri kendi bilgilerinize göre güncelleyin.
    """
    content = r"""% TABLE 1 — SoftwareX Metadata Table (auto-generated)
% Kaynak: https://www.elsevier.com/journals/softwarex/2352-7110/guide-for-authors
\begin{table}[!ht]
\caption{Metadata summary of \textit{Retaining Wall ML}.}
\label{tab:metadata}
\begin{tabular*}{\tblwidth}{@{} L L @{}}
\toprule
\textbf{Field} & \textbf{Value} \\
\midrule
Current code version        & v1.0.0 \\
Permanent link to code/repo & \url{https://github.com/<username>/retaining-wall-ml} \\
Legal software license      & MIT \\
Computing platform/OS       & Windows 10/11, Python 3.11+ \\
Installation requirements   & See \texttt{requirements.txt} \\
If available, link to user manual & \textit{Provided in} \texttt{README.md} \\
Support email               & 221300247@ogrenci.karatay.edu.tr \\
\bottomrule
\end{tabular*}
\end{table}
"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "table1_metadata.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] {path}")


# ===========================================================================
# Ana akış
# ===========================================================================


def main():
    args = parse_args()
    out = args.outdir

    print("\n=== Retaining Wall ML — Figure Generator ===")
    print(f"Output directory : {os.path.abspath(out)}")
    print(f"CSV path         : {args.csv}")
    print(f"SHAP bar path    : {args.shap}")
    print(f"SHAP summary     : {args.shap_summary}")
    print()

    print("[1/6] Fig 1 — Architecture flowchart...")
    fig1_architecture(out)

    print("[2/6] Fig 4a/4b — SHAP plots...")
    fig4_shap(args.shap, args.shap_summary, out, shap_values_csv=args.shap_values)

    print("[3/6] Fig 5 — Model comparison...")
    fig5_model_comparison(args.csv, out)

    print("[4/6] Table 1 — SoftwareX metadata...")
    table_metadata(out)

    print("[5/6] Table 2 — Design space...")
    table2_design_space(out)

    print("[6/6] Table 3 — Top-5 model metrics...")
    table3_top_models(args.csv, out)

    print("\n=== Done ===")
    print(f"All outputs saved to: {os.path.abspath(out)}")
    print()
    print("Sonraki adımlar:")
    print("  1. output/fig1_architecture.pdf    — makaleye ekle")
    print("  2. output/fig4a_shap_bar.pdf        — makaleye ekle")
    print("  3. output/fig4b_shap_summary.pdf    — makaleye ekle")
    print("  4. output/fig5_model_comparison.pdf — makaleye ekle")
    print("  5. output/table1_metadata.tex       — \\input{} ile makaleye ekle")
    print("  6. output/table2_design_space.tex   — \\input{} ile makaleye ekle")
    print("  7. output/table3_top_models.tex     — \\input{} ile makaleye ekle")
    print("  8. Fig 2 ve Fig 3 için uygulamayı açıp ekran görüntüsü al")


if __name__ == "__main__":
    main()
