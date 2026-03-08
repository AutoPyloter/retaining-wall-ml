# main.py
# Application entry point.
#
# Run from the app/ directory:
#
#   python main.py

import customtkinter as ctk

from config import read_config
from language import load_translations
from app import StabilityApp


def main() -> None:
    ctk.set_appearance_mode("Light")
    root = ctk.CTk()
    root.geometry("1100x1000")
    root.resizable(False, False)

    lang  = read_config("language", "EN")
    trans = load_translations(lang)
    root.title(trans["title"])

    StabilityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
