# language.py
# Helpers for listing available UI languages and loading translation dictionaries
# from JSON files stored in the Language/ directory.

import json
import os
from typing import List

from config import resource_path


def list_languages() -> List[str]:
    """Return language codes available in the Language/ directory (e.g. ['EN', 'TR'])."""
    lang_dir = resource_path("Language")
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(lang_dir)
        if f.endswith(".json")
    ]


def load_translations(lang_code: str) -> dict:
    """Load and return the translation dictionary for *lang_code*.

    Raises ``FileNotFoundError`` if the corresponding JSON file does not exist.
    """
    path = resource_path(os.path.join("Language", f"{lang_code}.json"))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
