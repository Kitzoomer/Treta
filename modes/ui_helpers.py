from __future__ import annotations

import customtkinter as ctk

from ui_theme import (
    MECH_BG,
    MECH_BTN_HOVER,
    MECH_CARD,
    MECH_PANEL,
    MECH_TEXT,
    apply_treta_theme,
)


def open_mode_window(app: "TretaApp", title: str, message: str) -> None:
    apply_treta_theme()

    window = ctk.CTkToplevel(app)
    window.title(title)
    window.geometry("520x260")
    window.minsize(480, 220)
    window.configure(fg_color=MECH_BG)

    frame = ctk.CTkFrame(window, fg_color=MECH_PANEL, corner_radius=18)
    frame.pack(fill="both", expand=True, padx=18, pady=18)
    frame.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        frame,
        text=title,
        text_color=MECH_TEXT,
        font=("Segoe UI", 16, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 6))

    ctk.CTkLabel(
        frame,
        text=message,
        text_color=MECH_TEXT,
        font=("Segoe UI", 12),
        wraplength=460,
        justify="left",
    ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 16))

    ctk.CTkButton(
        frame,
        text="Cerrar",
        command=window.destroy,
        fg_color=MECH_CARD,
        hover_color=MECH_BTN_HOVER,
        text_color=MECH_TEXT,
        corner_radius=12,
    ).grid(row=2, column=0, sticky="e", padx=16, pady=(0, 16))

    window.lift()
    window.focus()
