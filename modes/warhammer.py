from __future__ import annotations

from tkinter import messagebox


def open_warhammer_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_warhammer\n")
    messagebox.showinfo(
        "Modo Warhammer",
        "Coming soon.\n\nEste modo está reservado para Warhammer.",
    )
