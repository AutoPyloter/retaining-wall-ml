# config.py
# Utilities for resolving file paths (PyInstaller-compatible) and
# persisting user preferences to a plain-text config file.

import os
import sys


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resource_path(relative_path: str) -> str:
    """Return absolute path to *relative_path*, compatible with PyInstaller bundles.

    When the application is frozen by PyInstaller, temporary files are
    extracted to ``sys._MEIPASS``.  During normal Python execution the
    working directory is used instead.
    """
    try:
        base = sys._MEIPASS  # PyInstaller runtime
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, relative_path)


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

CONFIG_FILE = resource_path("config.cfg")


def read_config(key: str, default: str | None = None) -> str | None:
    """Read a single *key* from the config file.

    Returns *default* if the file does not exist or the key is absent.
    """
    if not os.path.exists(CONFIG_FILE):
        return default
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{key}="):
                return line.strip().split("=", 1)[1]
    return default


def write_config(key: str, value: str) -> None:
    """Write or update *key = value* in the config file."""
    lines: list[str] = []
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}\n")

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)