# geo5_setup.py
#
# GEO5 2018 - Cantilever Wall | First-Run Setup Script
# -----------------------------------------------------
# This script automates the one-time GEO5 configuration required before
# running the data generation pipeline (generate_dataset.py).
#
# It will:
#   1. Locate and launch GEO5 automatically
#   2. Configure all required settings, frames and parameters
#
# Usage:
#   python geo5_setup.py
#
# Run this script ONCE on each machine before starting data generation.
# After setup is complete, use generate_dataset.py for data collection.

from pywinauto import Application, keyboard
import time
import os
import subprocess


# =============================================================================
# Timing constants — increase if GEO5 is slow
# =============================================================================
T_KEY    = 0.08   # delay between keystrokes (s)
T_SHORT  = 0.15   # short pause (s)
T_MEDIUM = 0.40   # window opening (s)
T_LONG   = 0.80   # dialog full load (s)


def pause(t: float = T_KEY) -> None:
    time.sleep(t)


# =============================================================================
# Keyboard helpers
# =============================================================================

def tabs(n: int) -> None:
    for _ in range(n):
        keyboard.send_keys("{TAB}")
        pause(T_KEY)

def shift_tabs(n: int) -> None:
    for _ in range(n):
        keyboard.send_keys("+{TAB}")
        pause(T_KEY)

def space() -> None:
    keyboard.send_keys("{SPACE}")
    pause(T_KEY)

def down(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{DOWN}")
        pause(T_KEY)

def up(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{UP}")
        pause(T_KEY)

def right(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{RIGHT}")
        pause(T_KEY)

def enter() -> None:
    keyboard.send_keys("{ENTER}")
    pause(T_SHORT)

def type_text(text: str) -> None:
    """Paste text via clipboard to avoid locale-specific keyboard issues."""
    import pyperclip
    pyperclip.copy(str(text))
    keyboard.send_keys("^v")
    pause(T_SHORT)

def nav(letter: str, shift: bool = True) -> None:
    """Navigate to a frame via F10 (or Shift+F10) -> i -> letter."""
    if shift:
        keyboard.send_keys("+{F10}")
    else:
        keyboard.send_keys("{F10}")
    pause(T_SHORT)
    keyboard.send_keys("i")
    pause(T_SHORT)
    keyboard.send_keys(letter)
    pause(T_MEDIUM)


# =============================================================================
# Element-based click helper
# =============================================================================

def get_dpi_scale() -> float:
    """Return Windows DPI scale factor (e.g. 1.25 for 125% scaling)."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        hdc   = ctypes.windll.user32.GetDC(0)
        dpi   = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        scale = dpi / 96.0
        print(f"    DPI={dpi}  scale={scale:.2f}")
        return scale
    except Exception:
        return 1.0


def click_element(window, class_name: str, found_index: int = 0,
                  rel_x: float = 0.5, rel_y: float = 0.5) -> None:
    """Find a child element and click at a relative position within it.

    Coordinates are fractions of element width/height, so clicks scale
    correctly regardless of screen resolution or DPI.

    Args:
        window      : pywinauto window object
        class_name  : Win32 class name (e.g. 'TEnvToolScroller')
        found_index : pywinauto found_index (0-based)
        rel_x       : horizontal fraction (0.0=left, 1.0=right)
        rel_y       : vertical fraction   (0.0=top,  1.0=bottom)
    """
    import pyautogui
    el   = window.child_window(class_name=class_name, found_index=found_index)
    rect = el.rectangle()
    w    = rect.width()
    h    = rect.height()
    cx   = rect.left + int(w * rel_x)
    cy   = rect.top  + int(h * rel_y)
    print(f"    [{class_name}][{found_index}] rect={rect} (w={w} h={h}) "
          f"rel=({rel_x:.2f},{rel_y:.2f}) -> click ({cx}, {cy})")
    pyautogui.click(cx, cy)
    pause(T_MEDIUM)


# =============================================================================
# GEO5 executable search
# =============================================================================

GEO5_EXE_NAMES = [
    "CantileverWall_5_EN.exe",
    "CantileverWall.exe",
]


def find_geo5_exe() -> str:
    """Locate GEO5 exe via registry -> where command -> os.walk scan."""
    import winreg

    def _check(path: str):
        if path and os.path.isfile(path):
            print(f"    [OK] GEO5 found: {path}")
            return path
        return None

    # 1. Registry
    reg_roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, subkey in reg_roots:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        with winreg.OpenKey(key, winreg.EnumKey(key, i)) as sub:
                            try:
                                loc, _ = winreg.QueryValueEx(sub, "InstallLocation")
                                for name in GEO5_EXE_NAMES:
                                    r = _check(os.path.join(loc, name))
                                    if r: return r
                            except FileNotFoundError:
                                pass
                    except OSError:
                        pass
        except OSError:
            pass

    # 2. where command
    for name in GEO5_EXE_NAMES:
        try:
            out = subprocess.check_output(
                ["where", name], stderr=subprocess.DEVNULL, text=True
            ).strip()
            for line in out.splitlines():
                r = _check(line.strip())
                if r: return r
        except subprocess.CalledProcessError:
            pass

    # 3. os.walk scan
    search_roots = [
        os.environ.get("ProgramFiles",      r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramW6432",      r"C:\Program Files"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
    ]
    exe_lower = [n.lower() for n in GEO5_EXE_NAMES]
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        print(f"    [..] Scanning: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower() in exe_lower:
                    r = _check(os.path.join(dirpath, fname))
                    if r: return r

    raise FileNotFoundError(
        f"GEO5 executable not found.\n"
        f"Searched names: {GEO5_EXE_NAMES}\n"
        f"Registry, PATH and Program Files were scanned."
    )


# =============================================================================
# GEO5 launch and connection
# =============================================================================

def launch_geo5() -> None:
    """Launch GEO5 and dismiss the splash/startup dialog with Enter."""
    exe = find_geo5_exe()
    print("[LAUNCH] Starting GEO5...")
    subprocess.Popen([exe])
    pause(T_LONG * 4)
    print("[LAUNCH] Dismissing startup dialog (Enter)...")
    keyboard.send_keys("{ENTER}")
    pause(T_LONG * 3)
    print("[LAUNCH] GEO5 ready.")


def connect():
    """Connect to the GEO5 Cantilever Wall window. Returns (app, window)."""
    regex = r"(?=.*GEO5)(?=.*Cantilever Wall)(?=.*guz)"
    app    = Application(backend="win32").connect(title_re=regex)
    window = app.window(title_re=regex)
    window.set_focus()
    pause(T_MEDIUM)
    print("[OK] Connected to GEO5 window.")
    return app, window


# =============================================================================
# STEP 1 — Settings: configure analysis methods
#
# Shift+F10 -> i -> s             : open Settings window
# 4x Tab + Space                  : open Edit dialog
# 3x Tab + Space + Down + Enter   : Active earth pressure -> Coulomb
# 2x Tab + Space + Down + Space   : Shape of earth wedge -> Consider always vertical
# 11x Tab + Space                 : OK
# =============================================================================

def setup_settings() -> None:
    print("\n[SETTINGS] Starting...")

    nav("s")
    print("[SETTINGS] Window opened.")

    tabs(4); space(); pause(T_LONG)
    print("[SETTINGS] Edit dialog opened.")

    tabs(3); space(); pause(T_SHORT); down(1); enter()
    print("[SETTINGS] Active earth pressure -> Coulomb.")

    tabs(2); space(); pause(T_SHORT); down(1); space(); pause(T_SHORT)
    print("[SETTINGS] Shape of earth wedge -> Consider always vertical.")

    tabs(11); space(); pause(T_MEDIUM)
    print("[SETTINGS] Done.")


# =============================================================================
# STEP 2 — Profile: add second depth layer
#
# Shift+F10 -> i -> r             : Profile frame
# 2x Tab + "0"                    : Terrain elevation = 0
# click TEnvToolScroller[1]       : Add button (+15% right, +4% down)
# "10" + Tab + Enter + Esc        : enter depth value and close
# =============================================================================

def setup_profile(window) -> None:
    print("\n[PROFILE] Starting...")

    nav("r")
    print("[PROFILE] Frame opened.")

    tabs(2); keyboard.send_keys("0"); pause(T_SHORT)
    print("[PROFILE] Terrain elevation = 0.")

    print("[PROFILE] Clicking Add button...")
    click_element(window, "TEnvToolScroller", found_index=1, rel_x=0.15, rel_y=0.04)

    keyboard.send_keys("10"); pause(T_SHORT)
    tabs(1); enter(); pause(T_SHORT)
    keyboard.send_keys("{ESC}"); pause(T_SHORT)
    print("[PROFILE] Done.")


# =============================================================================
# STEP 3 — Soils: define soil1 (cohesive) and backfill (cohesionless)
#
# F10 -> i -> o                   : Soils frame
# click TEnvToolScroller[1]       : Add button
#
# For each soil:
#   Name -> Tab
#   Unit weight -> Tab
#   Stress-state (list) -> Tab    (skip)
#   Angle phi -> Tab
#   Cohesion c -> Tab
#   Delta -> Tab
#   Soil type (list)              : Right = cohesive, Left = cohesionless
#   -> Tab -> Poisson -> Tab
#   Calc. mode (list) -> Tab      (skip)
#   gamma_sat -> Shift+Tab x11 -> Space (Add)
#
# After second soil: Shift+Tab -> Enter (Cancel/close)
# =============================================================================

SOIL1 = {
    "name":      "soil1",
    "gamma":     "20",
    "phi":       "10",
    "c":         "0",
    "delta":     "0",
    "cohesive":  True,
    "poisson":   "0.33",
    "gamma_sat": "20",
}

BACKFILL = {
    "name":      "backfill",
    "gamma":     "20",
    "phi":       "40",
    "c":         "0",
    "delta":     "26.67",
    "cohesive":  False,
    "poisson":   "0.33",
    "gamma_sat": "20",
}


def _enter_soil(params: dict) -> None:
    """Enter soil parameters into the open soil dialog and click Add."""

    type_text(params["name"]); tabs(1)
    print(f"  Name = {params['name']}")

    keyboard.send_keys("^a"); type_text(params["gamma"]); tabs(1)
    print(f"  gamma = {params['gamma']}")

    tabs(1)  # skip Stress-state list

    keyboard.send_keys("^a"); type_text(params["phi"]); tabs(1)
    print(f"  phi = {params['phi']}")

    keyboard.send_keys("^a"); type_text(params["c"]); tabs(1)
    print(f"  c = {params['c']}")

    keyboard.send_keys("^a"); type_text(params["delta"]); tabs(1)
    print(f"  delta = {params['delta']}")

    if params["cohesive"]:
        right(1)
        print("  Soil type -> cohesive")
    else:
        keyboard.send_keys("{LEFT}")
        pause(T_KEY)
        print("  Soil type -> cohesionless")

    tabs(1)
    keyboard.send_keys("^a"); type_text(params["poisson"]); tabs(1)
    print(f"  poisson = {params['poisson']}")

    tabs(1)  # skip Calc. mode list

    keyboard.send_keys("^a"); type_text(params["gamma_sat"])
    print(f"  gamma_sat = {params['gamma_sat']}")

    shift_tabs(11); space(); pause(T_MEDIUM)
    print(f"  [{params['name']}] added.")


def setup_soils(window) -> None:
    print("\n[SOILS] Starting...")

    nav("o", shift=False)
    print("[SOILS] Frame opened.")

    print("[SOILS] Clicking Add button...")
    click_element(window, "TEnvToolScroller", found_index=1, rel_x=0.01, rel_y=0.5)

    print("[SOILS] Entering soil1...")
    _enter_soil(SOIL1)

    print("[SOILS] Entering backfill...")
    _enter_soil(BACKFILL)

    shift_tabs(1); enter(); pause(T_MEDIUM)
    print("[SOILS] Dialog closed.")
    print("[SOILS] Done.")


# =============================================================================
# STEP 4 — Assign: navigate to Assign frame (auto-assigns on open)
#
# F10 -> i -> a
# =============================================================================

def setup_assign() -> None:
    print("\n[ASSIGN] Starting...")
    nav("a", shift=False)
    print("[ASSIGN] Done.")


# =============================================================================
# STEP 5 — Backfill: select backfill type and soil
#
# click TEnvToolScroller[0]       : Frames panel, horizontal center,
#                                   offset_y = 200% of element width
# 3x Tab + Space                  : open backfill type dropdown
# Tab + Space + 2x Down + Space   : select backfill soil
# =============================================================================

def setup_backfill(window) -> None:
    import pyautogui
    print("\n[BACKFILL] Starting...")

    el   = window.child_window(class_name="TEnvToolScroller", found_index=0)
    rect = el.rectangle()
    cx   = rect.left + rect.width() // 2
    cy   = rect.top  + int(rect.width() * 2)
    print(f"    [TEnvToolScroller][0] rect={rect} -> click ({cx}, {cy})")
    pyautogui.click(cx, cy)
    pause(T_LONG)
    print("[BACKFILL] Frame opened.")

    tabs(3); space(); pause(T_SHORT)
    print("[BACKFILL] Backfill type dropdown opened.")

    tabs(1); space(); pause(T_SHORT)
    down(2); space(); pause(T_SHORT)
    print("[BACKFILL] Soil -> backfill selected.")

    print("[BACKFILL] Done.")


# =============================================================================
# STEP 6 — Water: indicate water behind wall
#
# F10 -> i -> w                   : Water frame
# 2x Tab + Space                  : enable water behind wall
# =============================================================================

def setup_water() -> None:
    print("\n[WATER] Starting...")
    nav("w", shift=False)
    print("[WATER] Frame opened.")
    tabs(2); space()
    print("[WATER] Water behind wall enabled.")
    print("[WATER] Done.")


# =============================================================================
# STEP 7 — FF Resistance: set passive resistance type and soil
#
# F10 -> i -> e                   : FF Resistance frame
# 2x Tab + Space                  : enable resistance
# 5x Tab + Space + Up + Space     : resistance type -> passive
# Tab + Space + Down + Space      : soil -> soil1
# =============================================================================

def setup_ff_resistance() -> None:
    print("\n[FF RESISTANCE] Starting...")
    nav("e", shift=False)
    print("[FF RESISTANCE] Frame opened.")

    tabs(2); space()
    print("[FF RESISTANCE] Resistance enabled.")

    tabs(5); space(); pause(T_SHORT); up(1); space(); pause(T_SHORT)
    print("[FF RESISTANCE] Resistance type -> passive.")

    tabs(1); space(); pause(T_SHORT); down(1); space(); pause(T_SHORT)
    print("[FF RESISTANCE] Soil -> soil1.")

    print("[FF RESISTANCE] Done.")


# =============================================================================
# STEP 8 — Earthquake: enable earthquake analysis
#
# F10 -> i -> h                   : Earthquake frame
# Tab + Space                     : enable earthquake analysis
# =============================================================================

def setup_earthquake() -> None:
    print("\n[EARTHQUAKE] Starting...")
    nav("h", shift=False)
    print("[EARTHQUAKE] Frame opened.")
    tabs(1); space()
    print("[EARTHQUAKE] Earthquake analysis enabled.")
    print("[EARTHQUAKE] Done.")


# =============================================================================
# Main flow
# =============================================================================

def run_setup() -> None:
    print("=" * 50)
    print("GEO5 Setup Automation — Experimental")
    print("=" * 50)

    launch_geo5()
    _, window = connect()

    window.set_focus(); pause(T_LONG)
    setup_settings()

    window.set_focus(); pause(T_LONG)
    setup_profile(window)

    window.set_focus(); pause(T_LONG)
    setup_soils(window)

    window.set_focus(); pause(T_LONG)
    setup_assign()

    window.set_focus(); pause(T_LONG)
    setup_backfill(window)

    window.set_focus(); pause(T_LONG)
    setup_water()

    window.set_focus(); pause(T_LONG)
    setup_ff_resistance()

    window.set_focus(); pause(T_LONG)
    setup_earthquake()

    print("\n" + "=" * 50)
    print("Setup complete.")
    print("=" * 50)


if __name__ == "__main__":
    run_setup()
