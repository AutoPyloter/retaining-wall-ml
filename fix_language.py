# Bu scripti app/ klasöründen çalıştır:
# python fix_language.py

import re

path = r"C:\Users\ASUS\Documents\GitHub\retaining-wall-ml\app\app.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """    def _on_language_change(self, *_args) -> None:
        new_lang = self.lang_var.get()
        write_config("language", new_lang)
        self.translations = load_translations(new_lang)
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.vars.clear()
        self.entry_labels.clear()
        self._build_ui()"""

new = """    def _on_language_change(self, *_args) -> None:
        new_lang = self.lang_var.get()
        write_config("language", new_lang)
        self.translations = load_translations(new_lang)
        # Mevcut girdi değerlerini kaydet
        saved_values = {key: var.get() for key, var in self.vars.items()}
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        self.vars.clear()
        self.entry_labels.clear()
        self._build_ui()
        # Kaydedilen değerleri geri yükle
        for key, value in saved_values.items():
            if key in self.vars and value:
                self.vars[key].set(value)"""

if old in content:
    content = content.replace(old, new)
    print("OK")
else:
    print("MISS - exact match bulunamadı")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
