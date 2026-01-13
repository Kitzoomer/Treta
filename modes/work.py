from __future__ import annotations

from modes.ui_helpers import open_mode_window


def open_work_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_work\n")
    open_mode_window(
        app,
        "Modo Trabajo",
        "Modo Trabajo está en preparación.\n\nAquí irá el Pomodoro, foco y reproductor lofi.",
    )
