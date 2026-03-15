# main.py
# Application entry point.
#
# Run from the app/ directory:
#
#   python main.py

import customtkinter as ctk

# ---------------------------------------------------------------------------
# joblib compatibility — pkl files were saved while train_models.py ran as
# __main__, so joblib looks for these classes/functions in __main__.
# Injecting them here ensures they are found regardless of entry point.
# ---------------------------------------------------------------------------
import __main__
from pipeline_components import OptionalScaler, select_top_k_features
__main__.OptionalScaler         = OptionalScaler
__main__.select_top_k_features  = select_top_k_features

from config import read_config
from language import load_translations
from app import StabilityApp


def main() -> None:
    ctk.set_appearance_mode("Light")
    root = ctk.CTk()
    root.geometry("1100x700")
    root.resizable(False, False)

    lang  = read_config("language", "EN")
    trans = load_translations(lang)
    root.title(trans["title"])

    StabilityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()