from __future__ import annotations

from modes.ui_helpers import open_mode_window


def open_warhammer_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_warhammer\n")
    open_mode_window(
        app,
        "Modo Warhammer",
        "Coming soon.\n\nEste modo está reservado para Warhammer.",
    )
