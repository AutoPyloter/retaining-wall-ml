# geo5_setup_experiment.py
#
# GEO5 2018 - Cantilever Wall
# Deneysel ilk kurulum otomasyonu.
#
# Amaç: GEO5 açıkken bu betiği çalıştırın.
# Betik Settings penceresini açıp analiz metodlarını ayarlar.
#
# Kullanım:
#   python geo5_setup_experiment.py
#
# NOT: Ana otomasyon koduna entegre edilmeden önce adım adım test edilmelidir.

from pywinauto import Application, keyboard
import time

# ---------------------------------------------------------------------------
# Timing constants — gerekirse artırın
# ---------------------------------------------------------------------------
T_KEY    = 0.08   # tek tuş basımı arası bekleme (s)
T_SHORT  = 0.15   # kısa bekleme (s)
T_MEDIUM = 0.40   # orta bekleme — pencere açılması (s)
T_LONG   = 0.80   # uzun bekleme — dialog tam yüklenme (s)


def pause(t: float = T_KEY) -> None:
    time.sleep(t)


# ---------------------------------------------------------------------------
# GEO5 exe yolunu bul
# ---------------------------------------------------------------------------
GEO5_EXE_NAMES = [
    "CantileverWall_5_EN.exe",
    "CantileverWall.exe",
]


def find_geo5_exe() -> str:
    """GEO5 Cantilever Wall exe dosyasını birden fazla yöntemle arar.

    Arama sırası:
      1. Windows registry  (HKLM / HKCU uninstall kayıtları)
      2. where komutu      (PATH üzerinde)
      3. Yaygın kurulum kökleri altında os.walk taraması
    """
    import os, subprocess, winreg

    def _check(path: str) -> str | None:
        """Dosya gerçekten varsa yolu döndür."""
        if path and os.path.isfile(path):
            print(f"[OK] GEO5 bulundu: {path}")
            return path
        return None

    # ------------------------------------------------------------------
    # 1. Registry taraması
    # ------------------------------------------------------------------
    reg_roots = [
        (winreg.HKEY_LOCAL_MACHINE,  r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,  r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER,   r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, subkey in reg_roots:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        with winreg.OpenKey(key, winreg.EnumKey(key, i)) as sub:
                            try:
                                install_loc, _ = winreg.QueryValueEx(sub, "InstallLocation")
                                for exe_name in GEO5_EXE_NAMES:
                                    candidate = os.path.join(install_loc, exe_name)
                                    result = _check(candidate)
                                    if result:
                                        return result
                            except FileNotFoundError:
                                pass
                    except OSError:
                        pass
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 2. where komutu (PATH üzerinde kayıtlıysa)
    # ------------------------------------------------------------------
    for exe_name in GEO5_EXE_NAMES:
        try:
            out = subprocess.check_output(
                ["where", exe_name], stderr=subprocess.DEVNULL, text=True
            ).strip()
            for line in out.splitlines():
                result = _check(line.strip())
                if result:
                    return result
        except subprocess.CalledProcessError:
            pass

    # ------------------------------------------------------------------
    # 3. os.walk taraması — Program Files kökleri + kullanıcı dizini
    # ------------------------------------------------------------------
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
                    result = _check(os.path.join(dirpath, fname))
                    if result:
                        return result

    raise FileNotFoundError(
        f"GEO5 exe bulunamadı.\n"
        f"Aranan isimler: {GEO5_EXE_NAMES}\n"
        f"Registry, PATH ve Program Files tarandı.\n"
        f"GEO5 kurulu değilse lütfen önce kurun."
    )


# ---------------------------------------------------------------------------
# GEO5'i başlat ve açılış penceresini geç
# ---------------------------------------------------------------------------
def launch_geo5() -> None:
    """GEO5'i başlatır, açılış splash/dialog penceresini Enter ile geçer."""
    import subprocess, os
    exe = find_geo5_exe()
    print("[1] GEO5 başlatılıyor...")
    subprocess.Popen([exe])
    pause(T_LONG * 4)   # uygulama yüklensin

    # Açılış penceresi — başlığı bilinmiyor, Enter ile geçiyoruz
    print("[1] Açılış penceresi geçiliyor (Enter)...")
    keyboard.send_keys("{ENTER}")
    pause(T_LONG * 3)   # ana pencere yüklensin
    print("[1] GEO5 hazır.")


# ---------------------------------------------------------------------------
# GEO5 penceresine bağlan
# ---------------------------------------------------------------------------
def connect() -> tuple:
    """GEO5 Cantilever Wall penceresine bağlan. (app, window) döndürür."""
    regex = r"(?=.*GEO5)(?=.*Cantilever Wall)(?=.*guz)"
    app    = Application(backend="win32").connect(title_re=regex)
    window = app.window(title_re=regex)
    window.set_focus()
    pause(T_MEDIUM)
    print("[OK] GEO5 penceresine bağlandı.")
    return app, window


# ---------------------------------------------------------------------------
# Yardımcı: tekrarlı Tab ve Space
# ---------------------------------------------------------------------------
def tabs(n: int) -> None:
    for _ in range(n):
        keyboard.send_keys("{TAB}")
        pause(T_KEY)

def space() -> None:
    keyboard.send_keys("{SPACE}")
    pause(T_KEY)

def down(n: int = 1) -> None:
    for _ in range(n):
        keyboard.send_keys("{DOWN}")
        pause(T_KEY)

def enter() -> None:
    keyboard.send_keys("{ENTER}")
    pause(T_SHORT)


# ---------------------------------------------------------------------------
# ADIM 1 — Settings penceresini aç
#
# Shift+F10  →  context menu açılır
# i          →  (Input menüsü kısayolu — GEO5 ana menüsünde i)
# s          →  Settings seçeneği
# ---------------------------------------------------------------------------
def open_settings_window() -> None:
    print("[1] Settings penceresi açılıyor...")
    keyboard.send_keys("+{F10}")   # Shift + F10
    pause(T_MEDIUM)
    keyboard.send_keys("i")
    pause(T_SHORT)
    keyboard.send_keys("s")
    pause(T_LONG)
    print("[1] Settings penceresi açıldı.")


# ---------------------------------------------------------------------------
# ADIM 2 — Edit butonuna git ve aç
#
# 4x Tab → Edit butonuna odaklan
# Space  → Edit penceresi açılır
# ---------------------------------------------------------------------------
def open_edit_dialog() -> None:
    print("[2] Edit dialogu açılıyor...")
    tabs(4)
    space()
    pause(T_LONG)
    print("[2] Edit dialogu açıldı.")


# ---------------------------------------------------------------------------
# ADIM 3 — Active earth pressure → Coulomb seç
#
# 3x Tab → Active earth pressure combo'ya git
# Space  → Dropdown aç
# Down   → Bir aşağı (Coulomb)
# Enter  → Onayla
# ---------------------------------------------------------------------------
def set_active_pressure_coulomb() -> None:
    print("[3] Active earth pressure → Coulomb...")
    tabs(3)
    space()
    pause(T_SHORT)
    down(1)
    enter()
    print("[3] Coulomb seçildi.")


# ---------------------------------------------------------------------------
# ADIM 4 — Shape of earth wedge → Consider always vertical
#
# 2x Tab → Shape of earth wedge combo'ya git
# Space  → Dropdown aç
# Down   → Bir aşağı (Consider always vertical)
# Space  → Onayla
# ---------------------------------------------------------------------------
def set_earth_wedge_vertical() -> None:
    print("[4] Shape of earth wedge → Consider always vertical...")
    tabs(2)
    space()
    pause(T_SHORT)
    down(1)
    space()
    pause(T_SHORT)
    print("[4] Consider always vertical seçildi.")


# ---------------------------------------------------------------------------
# ADIM 5 — Ayarları uygula / kapat
#
# 11x Tab → OK/Apply butonuna git
# Space   → Onayla
# ---------------------------------------------------------------------------
def confirm_settings() -> None:
    print("[5] Ayarlar onaylanıyor...")
    tabs(11)
    space()
    pause(T_MEDIUM)
    print("[5] Ayarlar uygulandı.")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def run_setup() -> None:
    print("=" * 50)
    print("GEO5 Setup Otomasyonu — Deneysel")
    print("=" * 50)

    launch_geo5()

    _, window = connect()
    window.set_focus()
    pause(T_SHORT)

    open_settings_window()
    open_edit_dialog()
    set_active_pressure_coulomb()
    set_earth_wedge_vertical()
    confirm_settings()

    print("=" * 50)
    print("Setup tamamlandı.")
    print("=" * 50)


if __name__ == "__main__":
    run_setup()