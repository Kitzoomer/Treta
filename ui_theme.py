from __future__ import annotations

import customtkinter as ctk

# ===========================
# THEME: Adeptus Mechanicus
# ===========================
MECH_BG = "#0b0f12"  # fondo ventana
MECH_PANEL = "#0f151a"  # paneles laterales y base
MECH_CARD = "#121b21"  # tarjetas
MECH_CARD_2 = "#10181e"  # segunda capa
MECH_BORDER = "#2b3842"  # borde suave
MECH_TEXT = "#d6dde3"  # texto principal
MECH_MUTED = "#93a4b1"  # texto secundario
MECH_RED = "#c1121f"  # acento mechanicus
MECH_RED_DARK = "#7a0c12"  # hover/active rojo
MECH_BTN_HOVER = "#1b2831"  # hover neutro
MECH_DANGER = "#2a3137"  # botón desactivado
MECH_OK = "#2fd27d"


def apply_treta_theme(widget_scaling: float = 1.15) -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    try:
        ctk.set_widget_scaling(widget_scaling)
    except Exception:
        pass
