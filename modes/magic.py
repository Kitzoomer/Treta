from __future__ import annotations

import os
import subprocess
import sys
from tkinter import messagebox


def open_magic_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_magic\n")

    base_dir = app.base_dir
    magic_path = os.path.join(base_dir, "magic_hub.py")
    if not os.path.exists(magic_path):
        messagebox.showerror("Modo Magic", f"No encuentro magic_hub.py en:\n{magic_path}")
        return

    try:
        if app.magic_proc is not None and app.magic_proc.poll() is None:
            messagebox.showinfo("Modo Magic", "Modo Magic ya está abierto.")
            return
    except Exception:
        pass

    try:
        app._set_status("abriendo Modo Magic…")
        app.magic_proc = subprocess.Popen(
            [sys.executable, magic_path],
            cwd=os.path.dirname(magic_path),
        )
        app._log("🪄 Modo Magic abierto.\n")
    except Exception as exc:
        messagebox.showerror("Modo Magic", f"No pude abrir Modo Magic.\n\nDetalle: {exc}")
    finally:
        app._set_status("en espera")
