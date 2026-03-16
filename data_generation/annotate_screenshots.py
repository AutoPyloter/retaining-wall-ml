# annotate_screenshots.py
#
# GEO5 2018 - Cantilever Wall | Screenshot Annotation Script
# ----------------------------------------------------------
# Reads output.txt (comma-separated, no header) and matches each row
# to its screenshot by the counter value in column 0.
# Overlays a metadata box in the top-left corner of each image.
#
# Column order in output.txt:
#   [0]  counter   — scenario index (matches screenshot filename)
#   [1]  H         — wall height (m)
#   [2]  X1        — base width (m)
#   [3]  X2        — toe slab thickness (m)
#   [4]  X3        — toe projection (m)
#   [5]  X4        — heel projection (m)
#   [6]  X5        — stem bottom width (m)
#   [7]  X6        — stem top width (m)
#   [8]  X7        — key depth (m)  [X8 in some versions]
#   [9]  q         — surcharge (kN/m²)
#   [10] sds       — seismic coefficient SDS
#   [11] soil_class — soil class index
#   [12] hw        — groundwater depth (m)
#   [13] Fa        — active force (kN/m)
#   [14] Fp        — passive force (kN/m)
#   [15] Ma        — overturning moment (kNm/m)
#   [16] Mp        — stabilizing moment (kNm/m)
#   [17] x         — resultant position x (m)
#   [18] z         — resultant position z (m)
#   [19] R         — resultant force R (kN/m)
#   Fss = Mp / Ma  (computed)
#
# Usage:
#   python annotate_screenshots.py
#
# Output: annotated images saved to screenshots/annotated/

import os
import glob
import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

DATA_FILE      = "output.txt"
SCREENSHOT_DIR = "screenshots"
OUTPUT_DIR     = os.path.join(SCREENSHOT_DIR, "annotated")

# Column indices in output.txt
# No counter column — first column is H
COL_H          = 0
COL_X1         = 1
COL_X2         = 2
COL_X3         = 3
COL_X4         = 4
COL_X5         = 5
COL_X6         = 6
COL_X7         = 7
COL_X8         = 8
COL_Q          = 9
COL_SDS        = 10
COL_SOIL       = 11
COL_HW         = 12
COL_FA         = 13
COL_FP         = 14
COL_MA         = 15
COL_MP         = 16
COL_X          = 17
COL_Z          = 18
COL_R          = 19

# Box appearance
BOX_ALPHA   = 0.82
BOX_COLOR   = (255, 255, 255)
BOX_BORDER  = (60, 60, 60)
TEXT_COLOR  = (20, 20, 20)
TITLE_COLOR = (0, 80, 160)
VAL_COLOR   = (0, 0, 0)
FSS_COLOR   = (0, 140, 0)    # green for safety factor
FSS_WARN    = (0, 0, 200)    # red for Fss < 1.5
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_TITLE  = 0.52
FONT_TEXT   = 0.44
THICKNESS   = 1
LINE_H      = 22
PADDING     = 10
MARGIN      = 14


# =============================================================================
# Data reader
# =============================================================================

def read_data(filepath: str) -> list[list[str]]:
    """Read comma-separated output.txt, return list of value lists."""
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([v.strip() for v in line.split(",")])
    return rows


def parse_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Screenshot finder
# =============================================================================

def find_screenshot(counter: int) -> str | None:
    pattern = os.path.join(SCREENSHOT_DIR, f"{counter}_stability_*.png")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# =============================================================================
# Annotation
# =============================================================================

def build_lines(cols: list[str]) -> list[tuple[str, str, tuple]]:
    """Return list of (label, value_string, text_color) tuples."""

    def fmt(idx, decimals=3, unit=""):
        v = parse_float(cols[idx]) if idx < len(cols) else None
        s = f"{v:.{decimals}f}" if v is not None else "—"
        return f"{s}  {unit}".rstrip() if unit else s

    def _fmt_hw(cols):
        """Compute actual groundwater depth from hw index and geometry."""
        H_v  = parse_float(cols[COL_H])  if COL_H  < len(cols) else None
        x1_v = parse_float(cols[COL_X1]) if COL_X1 < len(cols) else None
        hw_v = parse_float(cols[COL_HW]) if COL_HW < len(cols) else None
        if H_v is None or x1_v is None or hw_v is None:
            return "—"
        options = [0, H_v / 2, H_v, x1_v / 2 + H_v, H_v + x1_v]
        idx = int(hw_v)
        if 0 <= idx < len(options):
            return f"{options[idx]:.3f}  m"
        return "—"

    # Compute Fss
    ma = parse_float(cols[COL_MA]) if COL_MA < len(cols) else None
    mp = parse_float(cols[COL_MP]) if COL_MP < len(cols) else None
    if ma and mp and ma != 0:
        fss = mp / ma
        fss_str = f"{fss:.4f}"
        fss_col = FSS_WARN if fss < 1.5 else FSS_COLOR
    else:
        fss_str = "—"
        fss_col = TEXT_COLOR

    lines = [
        # Geometry
        ("Wall height H",         fmt(COL_H,   3, "m"),    TEXT_COLOR),
        ("Base width X1",         fmt(COL_X1,  3, "m"),    TEXT_COLOR),
        ("Toe slab X2",           fmt(COL_X2,  3, "m"),    TEXT_COLOR),
        ("Toe proj. X3",          fmt(COL_X3,  3, "m"),    TEXT_COLOR),
        ("Heel proj. X4",         fmt(COL_X4,  3, "m"),    TEXT_COLOR),
        ("Stem bottom X5",        fmt(COL_X5,  3, "m"),    TEXT_COLOR),
        ("Stem top X6",           fmt(COL_X6,  3, "m"),    TEXT_COLOR),
        ("Key depth X7",          fmt(COL_X7,  3, "m"),    TEXT_COLOR),
        ("Key width X8",          fmt(COL_X8,  3, "m"),    TEXT_COLOR),
        # Loads
        ("Surcharge q",           fmt(COL_Q,   1, "kN/m2"),TEXT_COLOR),
        ("Seismic SDS",           fmt(COL_SDS, 2),         TEXT_COLOR),
        ("Soil class",            cols[COL_SOIL] if COL_SOIL < len(cols) else "—",
                                                           TEXT_COLOR),
        ("Groundwater idx",       fmt(COL_HW,  0),         TEXT_COLOR),
        ("Groundwater depth",     _fmt_hw(cols),           TEXT_COLOR),
        # Forces & moments
        ("Active force Fa",       fmt(COL_FA,  2, "kN/m"), TEXT_COLOR),
        ("Passive force Fp",      fmt(COL_FP,  2, "kN/m"), TEXT_COLOR),
        ("Overturning Ma",        fmt(COL_MA,  2, "kNm/m"),TEXT_COLOR),
        ("Stabilizing Mp",        fmt(COL_MP,  2, "kNm/m"),TEXT_COLOR),
        ("Resultant x",           fmt(COL_X,   3, "m"),    TEXT_COLOR),
        ("Resultant z",           fmt(COL_Z,   3, "m"),    TEXT_COLOR),
        ("Resultant R",           fmt(COL_R,   2, "kN/m"), TEXT_COLOR),
        # Safety factor
        ("Safety factor Fss",     fss_str,                 fss_col),
    ]
    return lines


def annotate(img: np.ndarray, cols: list[str], counter: int) -> np.ndarray:
    img   = img.copy()
    title = f"Scenario  #{counter}"
    lines = build_lines(cols)

    # Measure
    (tw, th), _ = cv2.getTextSize(title, FONT, FONT_TITLE, THICKNESS + 1)
    label_w = max(
        cv2.getTextSize(f"{lbl}:", FONT, FONT_TEXT, THICKNESS)[0][0]
        for lbl, _, _ in lines
    ) + 10
    val_w = max(
        cv2.getTextSize(v, FONT, FONT_TEXT, THICKNESS)[0][0]
        for _, v, _ in lines
    ) + 8

    box_w = max(tw, label_w + val_w) + PADDING * 2
    box_h = PADDING * 2 + LINE_H + 6 + len(lines) * LINE_H

    x0, y0 = MARGIN, MARGIN
    x1, y1 = x0 + box_w, y0 + box_h

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), BOX_COLOR, -1)
    cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), BOX_BORDER, 1)

    # Title
    ty = y0 + PADDING + th
    cv2.putText(img, title, (x0 + PADDING, ty),
                FONT, FONT_TITLE, TITLE_COLOR, THICKNESS + 1, cv2.LINE_AA)

    # Separator
    sep_y = ty + 6
    cv2.line(img, (x0 + PADDING, sep_y), (x1 - PADDING, sep_y), BOX_BORDER, 1)

    # Parameter lines
    for i, (label, val, col) in enumerate(lines):
        ly = sep_y + 5 + (i + 1) * LINE_H
        cv2.putText(img, f"{label}:", (x0 + PADDING, ly),
                    FONT, FONT_TEXT, (80, 80, 80), THICKNESS, cv2.LINE_AA)
        cv2.putText(img, val, (x0 + PADDING + label_w, ly),
                    FONT, FONT_TEXT, col, THICKNESS, cv2.LINE_AA)

    return img


# =============================================================================
# Main
# =============================================================================

def run() -> None:
    print("=" * 55)
    print("Screenshot Annotation")
    print("=" * 55)

    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: '{DATA_FILE}'")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = read_data(DATA_FILE)
    print(f"[OK] {len(rows)} rows loaded from '{DATA_FILE}'")

    annotated = 0
    skipped   = 0

    for row_idx, cols in enumerate(rows, start=1):
        if not cols:
            continue
        counter = row_idx

        png = find_screenshot(counter)
        if png is None:
            print(f"  [SKIP] No screenshot for scenario #{counter}")
            skipped += 1
            continue

        img = cv2.imread(png)
        if img is None:
            print(f"  [SKIP] Cannot read: {png}")
            skipped += 1
            continue

        out  = annotate(img, cols, counter)
        dest = os.path.join(OUTPUT_DIR, os.path.basename(png))
        cv2.imwrite(dest, out)
        print(f"  [OK] #{counter:4d}  {os.path.basename(png)}")
        annotated += 1

    print("=" * 55)
    print(f"Done.  Annotated: {annotated}   Skipped: {skipped}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run()