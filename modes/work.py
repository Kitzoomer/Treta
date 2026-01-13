from __future__ import annotations

from tkinter import messagebox


def open_work_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_work\n")
    messagebox.showinfo(
        "Modo Trabajo",
        "Modo Trabajo está en preparación.\n\n"
        "Aquí irá el Pomodoro, foco y reproductor lofi.",
    )
