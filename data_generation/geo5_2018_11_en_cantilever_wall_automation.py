# geo5_setup_experiment.py
#
# GEO5 2018 - Cantilever Wall
# Deneysel ilk kurulum otomasyonu.
#
# Kullanım:
#   python geo5_setup_experiment.py
#
# NOT: Ana otomasyon koduna entegre edilmeden önce adım adım test edilmelidir.

from pywinauto import Application, keyboard
import time
import os
import subprocess


# =============================================================================
# Zamanlama sabitleri — gerekirse artırın
# =============================================================================
T_KEY    = 0.08   # tek tuş basımı arası (s)
T_SHORT  = 0.15   # kısa bekleme (s)
T_MEDIUM = 0.40   # pencere açılması (s)
T_LONG   = 0.80   # dialog tam yüklenme (s)


def pause(t: float = T_KEY) -> None:
    time.sleep(t)


# =============================================================================
# Yardımcı tuş fonksiyonları
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

def right(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{RIGHT}")
        pause(T_KEY)

def left(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{LEFT}")
        pause(T_KEY)

def enter() -> None:
    keyboard.send_keys("{ENTER}")
    pause(T_SHORT)

def type_text(text: str) -> None:
    """Metni karakter karakter yazar (Türkçe karakter sorununu önler)."""
    import pyperclip
    pyperclip.copy(str(text))
    keyboard.send_keys("^v")
    pause(T_SHORT)

def nav(letter: str, shift: bool = True) -> None:
    """F10 (veya Shift+F10) → i → letter ile frame değiştir."""
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
# GEO5 exe bulma
# =============================================================================

GEO5_EXE_NAMES = [
    "CantileverWall_5_EN.exe",
    "CantileverWall.exe",
]


def find_geo5_exe() -> str:
    """GEO5 exe dosyasını registry → where → os.walk sırasıyla arar."""
    import winreg

    def _check(path: str):
        if path and os.path.isfile(path):
            print(f"[OK] GEO5 bulundu: {path}")
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

    # 2. where komutu
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

    # 3. os.walk taraması
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
        print(f"[..] Taranıyor: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower() in exe_lower:
                    r = _check(os.path.join(dirpath, fname))
                    if r: return r

    raise FileNotFoundError(
        f"GEO5 exe bulunamadı.\n"
        f"Aranan isimler: {GEO5_EXE_NAMES}\n"
        f"Registry, PATH ve Program Files tarandı."
    )


# =============================================================================
# GEO5 başlatma ve bağlantı
# =============================================================================

def launch_geo5() -> None:
    """GEO5'i başlatır, açılış penceresini Enter ile geçer."""
    exe = find_geo5_exe()
    print("[LAUNCH] GEO5 başlatılıyor...")
    subprocess.Popen([exe])
    pause(T_LONG * 4)
    print("[LAUNCH] Açılış penceresi geçiliyor (Enter)...")
    keyboard.send_keys("{ENTER}")
    pause(T_LONG * 3)
    print("[LAUNCH] GEO5 hazır.")


def connect():
    """GEO5 Cantilever Wall penceresine bağlan. (app, window) döndürür."""
    regex = r"(?=.*GEO5)(?=.*Cantilever Wall)(?=.*guz)"
    app    = Application(backend="win32").connect(title_re=regex)
    window = app.window(title_re=regex)
    window.set_focus()
    pause(T_MEDIUM)
    print("[OK] GEO5 penceresine bağlandı.")
    return app, window


# =============================================================================
# ADIM 1 — Settings: analiz metodlarını ayarla
#
# Shift+F10 → i → s            : Settings penceresi
# 4x Tab + Space               : Edit dialogu aç
# 3x Tab + Space + Down + Enter: Active earth pressure → Coulomb
# 2x Tab + Space + Down + Space: Shape of earth wedge → Consider always vertical
# 11x Tab + Space              : OK
# =============================================================================

def setup_settings() -> None:
    print("\n[SETTINGS] Başlıyor...")

    nav("s")
    print("[SETTINGS] Pencere açıldı.")

    tabs(4); space(); pause(T_LONG)
    print("[SETTINGS] Edit dialogu açıldı.")

    tabs(3); space(); pause(T_SHORT); down(1); enter()
    print("[SETTINGS] Active earth pressure → Coulomb.")

    tabs(2); space(); pause(T_SHORT); down(1); space(); pause(T_SHORT)
    print("[SETTINGS] Shape of earth wedge → Consider always vertical.")

    tabs(11); space(); pause(T_MEDIUM)
    print("[SETTINGS] Tamamlandı.")


# =============================================================================
# ADIM 2 — Profile: ikinci derinlik katmanını ekle
#
# Shift+F10 → i → r            : Profile frame
# 2x Tab + "0"                 : Terrain elevation = 0
# Mouse click (Add butonu)     : TEnvToolScroller {l:35, t:583, h:401} → +260px sağ, +15px aşağı
# "10" + Tab + Enter + Esc     : Derinlik değeri gir, kapat
# =============================================================================

def setup_profile(window) -> None:
    import pyautogui
    print("\n[PROFILE] Başlıyor...")

    nav("r")
    print("[PROFILE] Frame açıldı.")

    tabs(2); keyboard.send_keys("0"); pause(T_SHORT)
    print("[PROFILE] Terrain elevation = 0.")

    rect    = window.rectangle()
    click_x = rect.left + 35 + 260
    click_y = rect.top  + 583 + 15
    print(f"[PROFILE] Add butonuna tıklanıyor: ({click_x}, {click_y})")
    pyautogui.click(click_x, click_y)
    pause(T_LONG)

    keyboard.send_keys("10"); pause(T_SHORT)
    tabs(1); enter(); pause(T_SHORT)
    keyboard.send_keys("{ESC}"); pause(T_SHORT)
    print("[PROFILE] Tamamlandı.")


# =============================================================================
# ADIM 3 — Soils: soil1 ve backfill tanımla
#
# F10 → i → o                 : Soils frame
# Mouse click (Add butonu)    : TEnvToolScroller {l:35, t:583, h:28} → +15px sağ, dikey orta
#
# Her zemin için diyalog akışı:
#   Name        → Tab
#   Unit weight → Tab
#   Stress-state (liste) → Tab  (geç)
#   Angle φ     → Tab
#   Cohesion c  → Tab
#   Delta δ     → Tab
#   Soil type (liste, cohesionless default) → Right (cohesive için) → Tab
#   Poisson ν   → Tab
#   Calc. mode  (liste) → Tab  (geç)
#   γsat        → Shift+Tab×11 → Add (Space)
#
# İkinci zemin aynı sıra, sonra Shift+Tab → Cancel (Enter)
# =============================================================================

# Zemin parametreleri
# cohesive=True  → liste 1 sağ ok ile cohesive'e geçer
# cohesive=False → default (cohesionless), ok yok
SOIL1 = {
    "name":     "soil1",
    "gamma":    "20",
    "phi":      "10",
    "c":        "0",
    "delta":    "0",
    "cohesive": True,
    "poisson":  "0.33",
    "gamma_sat":"20",
}

BACKFILL = {
    "name":     "backfill",
    "gamma":    "20",
    "phi":      "40",
    "c":        "0",
    "delta":    "26.67",
    "cohesive": False,
    "poisson":  "0.33",
    "gamma_sat":"20",
}


def _enter_soil(params: dict) -> None:
    """Açık zemin diyaloguna parametreleri girer ve Add'e basar."""

    # Name
    type_text(params["name"]); tabs(1)
    print(f"  Name = {params['name']}")

    # Unit weight
    keyboard.send_keys("^a")          # mevcut değeri seç
    type_text(params["gamma"]); tabs(1)
    print(f"  γ = {params['gamma']}")

    # Stress-state (liste) — geç
    tabs(1)

    # Angle of internal friction φ
    keyboard.send_keys("^a")
    type_text(params["phi"]); tabs(1)
    print(f"  φ = {params['phi']}")

    # Cohesion c
    keyboard.send_keys("^a")
    type_text(params["c"]); tabs(1)
    print(f"  c = {params['c']}")

    # Angle of friction struct-soil δ
    keyboard.send_keys("^a")
    type_text(params["delta"]); tabs(1)
    print(f"  δ = {params['delta']}")

    # Soil type
    # cohesive   → Right (cohesionless'tan cohesive'e geç)
    # cohesionless → Left (bir önceki zemin cohesive bıraktıysa geri al)
    if params["cohesive"]:
        right(1)
        print("  Soil type → cohesive")
    else:
        keyboard.send_keys("{LEFT}")
        pause(T_KEY)
        print("  Soil type → cohesionless")

    # Poisson ν — bir sonraki kutu
    tabs(1)
    keyboard.send_keys("^a")
    type_text(params["poisson"]); tabs(1)
    print(f"  ν = {params['poisson']}")

    # Calc. mode of uplift (liste) — geç
    tabs(1)

    # γsat
    keyboard.send_keys("^a")
    type_text(params["gamma_sat"])
    print(f"  γsat = {params['gamma_sat']}")

    # Shift+Tab × 11 → Add butonu
    shift_tabs(11)
    space()
    pause(T_MEDIUM)
    print(f"  [{params['name']}] eklendi.")


def setup_soils(window) -> None:
    import pyautogui
    print("\n[SOILS] Başlıyor...")

    nav("o", shift=False)
    print("[SOILS] Frame açıldı.")

    # Add butonuna tıkla — diyalog açılır
    rect    = window.rectangle()
    click_x = rect.left + 35 + 15
    click_y = rect.top  + 583 + 28 // 2
    print(f"[SOILS] Add butonuna tıklanıyor: ({click_x}, {click_y})")
    pyautogui.click(click_x, click_y)
    pause(T_LONG)

    # soil1 gir
    print("[SOILS] soil1 giriliyor...")
    _enter_soil(SOIL1)

    # Diyalog kapanmadı, backfill gir
    # (önceki veriler kalır, name kutusu aktif)
    print("[SOILS] backfill giriliyor...")
    _enter_soil(BACKFILL)

    # Diyalogu kapat: Shift+Tab → Cancel → Enter
    shift_tabs(1)
    enter()
    pause(T_MEDIUM)
    print("[SOILS] Diyalog kapatıldı.")
    print("[SOILS] Tamamlandı.")


# =============================================================================
# Ana akış
# =============================================================================

def run_setup() -> None:
    print("=" * 50)
    print("GEO5 Setup Otomasyonu — Deneysel")
    print("=" * 50)

    launch_geo5()
    _, window = connect()

    window.set_focus(); pause(T_SHORT)
    setup_settings()

    window.set_focus(); pause(T_SHORT)
    setup_profile(window)

    window.set_focus(); pause(T_SHORT)
    setup_soils(window)

    print("\n" + "=" * 50)
    print("Setup tamamlandı.")
    print("=" * 50)


if __name__ == "__main__":
    run_setup()