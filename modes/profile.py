from __future__ import annotations

import customtkinter as ctk

from ui_theme import (
    MECH_BG,
    MECH_BTN_HOVER,
    MECH_CARD,
    MECH_CARD_2,
    MECH_MUTED,
    MECH_PANEL,
    MECH_TEXT,
    apply_treta_theme,
)


def _make_section(parent: ctk.CTkFrame, title: str) -> ctk.CTkFrame:
    section = ctk.CTkFrame(parent, fg_color=MECH_CARD, corner_radius=16)
    section.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        section,
        text=title,
        text_color=MECH_TEXT,
        font=("Segoe UI", 14, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
    return section


def _make_progress(section: ctk.CTkFrame, value: float) -> ctk.CTkProgressBar:
    bar = ctk.CTkProgressBar(section, progress_color="#2fd27d")
    bar.set(value)
    return bar


def open_profile_mode(app: "TretaApp") -> None:
    app._log("▶ Ejecutado: mode_profile\n")
    apply_treta_theme()

    window = ctk.CTkToplevel(app)
    window.title("Modo Perfil")
    window.geometry("980x720")
    window.minsize(920, 680)
    window.configure(fg_color=MECH_BG)

    header = ctk.CTkFrame(window, fg_color=MECH_PANEL, corner_radius=18)
    header.pack(fill="x", padx=20, pady=(20, 12))
    header.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        header,
        text="Modo Perfil · Marian",
        text_color=MECH_TEXT,
        font=("Segoe UI", 18, "bold"),
    ).grid(row=0, column=0, sticky="w", padx=18, pady=16)
    ctk.CTkLabel(
        header,
        text="My Isekai Life – Level Up Dashboard",
        text_color=MECH_MUTED,
        font=("Segoe UI", 12),
    ).grid(row=0, column=1, sticky="e", padx=18, pady=16)

    body = ctk.CTkFrame(window, fg_color=MECH_BG)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    body.grid_columnconfigure(0, weight=1)
    body.grid_columnconfigure(1, weight=1)
    body.grid_rowconfigure(0, weight=1)

    left = ctk.CTkFrame(body, fg_color=MECH_BG)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right = ctk.CTkFrame(body, fg_color=MECH_BG)
    right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

    left.grid_columnconfigure(0, weight=1)
    right.grid_columnconfigure(0, weight=1)

    dashboard = _make_section(left, "Customizable Character Dashboard")
    dashboard.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

    dashboard_info = [
        ("Nombre", "Marian"),
        ("Clase", "The Silent Strategist"),
        ("Nivel", "7"),
        ("Score total", "340 XP"),
        ("Título actual", "Warrior of Focus"),
    ]
    for index, (label, value) in enumerate(dashboard_info, start=1):
        row = ctk.CTkFrame(dashboard, fg_color=MECH_CARD_2, corner_radius=12)
        row.grid(row=index, column=0, sticky="ew", padx=14, pady=6)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=label, text_color=MECH_MUTED, font=("Segoe UI", 11)).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=8,
        )
        ctk.CTkLabel(row, text=value, text_color=MECH_TEXT, font=("Segoe UI", 12, "bold")).grid(
            row=0,
            column=1,
            sticky="e",
            padx=10,
            pady=8,
        )

    level_section = _make_section(left, "Level Up Progress Bar")
    level_section.grid(row=1, column=0, sticky="nsew", pady=(0, 12))

    ctk.CTkLabel(
        level_section,
        text="Nivel 7 → Nivel 8 (500 XP)",
        text_color=MECH_TEXT,
        font=("Segoe UI", 12),
    ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 6))
    level_bar = _make_progress(level_section, 0.68)
    level_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))

    radar_section = _make_section(left, "Character Radar Chart")
    radar_section.grid(row=2, column=0, sticky="nsew")

    radar_stats = [
        ("Strength", 0.62),
        ("Intelligence", 0.78),
        ("Charisma", 0.55),
        ("Discipline", 0.81),
        ("Resilience", 0.69),
    ]
    for idx, (stat, value) in enumerate(radar_stats, start=1):
        row = ctk.CTkFrame(radar_section, fg_color=MECH_CARD_2, corner_radius=12)
        row.grid(row=idx, column=0, sticky="ew", padx=14, pady=6)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=stat, text_color=MECH_TEXT, font=("Segoe UI", 11)).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=8,
        )
        bar = _make_progress(row, value)
        bar.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)

    stats_section = _make_section(right, "Character Stats Database [Manual & Automated]")
    stats_section.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
    stats_section.grid_columnconfigure(0, weight=1)

    headers = ctk.CTkFrame(stats_section, fg_color=MECH_CARD_2, corner_radius=12)
    headers.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
    headers.grid_columnconfigure(1, weight=1)
    headers.grid_columnconfigure(2, weight=1)
    ctk.CTkLabel(headers, text="Stat", text_color=MECH_MUTED, font=("Segoe UI", 11, "bold")).grid(
        row=0,
        column=0,
        sticky="w",
        padx=10,
        pady=6,
    )
    ctk.CTkLabel(
        headers,
        text="Manual",
        text_color=MECH_MUTED,
        font=("Segoe UI", 11, "bold"),
    ).grid(row=0, column=1, sticky="w", padx=10, pady=6)
    ctk.CTkLabel(
        headers,
        text="Automated",
        text_color=MECH_MUTED,
        font=("Segoe UI", 11, "bold"),
    ).grid(row=0, column=2, sticky="w", padx=10, pady=6)

    stat_rows = [
        ("💪 Strength", "+2", "+5"),
        ("🧠 Intelligence", "+3", "+4"),
        ("🔥 Confidence", "+1", "+2"),
        ("🌿 Spirituality", "+2", "+1"),
        ("🌟 Charisma", "+1", "+3"),
    ]
    for idx, (stat, manual, auto) in enumerate(stat_rows, start=2):
        row = ctk.CTkFrame(stats_section, fg_color=MECH_CARD_2, corner_radius=12)
        row.grid(row=idx, column=0, sticky="ew", padx=14, pady=4)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(row, text=stat, text_color=MECH_TEXT, font=("Segoe UI", 11)).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=6,
        )
        ctk.CTkLabel(row, text=manual, text_color=MECH_TEXT, font=("Segoe UI", 11)).grid(
            row=0,
            column=1,
            sticky="w",
            padx=10,
            pady=6,
        )
        ctk.CTkLabel(row, text=auto, text_color=MECH_TEXT, font=("Segoe UI", 11)).grid(
            row=0,
            column=2,
            sticky="w",
            padx=10,
            pady=6,
        )

    quests_section = _make_section(right, "Automated Quest Database")
    quests_section.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
    quests_section.grid_columnconfigure(0, weight=1)

    quest_rows = [
        ("🏋️ Workout", "+15 XP", "Completada"),
        ("📖 Reading", "+10 XP", "En progreso"),
        ("🧘 Meditation", "+12 XP", "Completada"),
        ("🍟 Junk Food", "-8 XP", "Evitar"),
        ("📱 Doom Scrolling", "-5 XP", "Evitar"),
    ]
    for idx, (quest, reward, status) in enumerate(quest_rows, start=1):
        row = ctk.CTkFrame(quests_section, fg_color=MECH_CARD_2, corner_radius=12)
        row.grid(row=idx, column=0, sticky="ew", padx=14, pady=4)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text=quest, text_color=MECH_TEXT, font=("Segoe UI", 11)).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=6,
        )
        ctk.CTkLabel(row, text=reward, text_color=MECH_MUTED, font=("Segoe UI", 11)).grid(
            row=0,
            column=1,
            sticky="e",
            padx=10,
            pady=6,
        )
        ctk.CTkLabel(row, text=status, text_color=MECH_TEXT, font=("Segoe UI", 11, "bold")).grid(
            row=0,
            column=2,
            sticky="e",
            padx=10,
            pady=6,
        )

    progress_section = _make_section(right, "Quest Progression Tracking")
    progress_section.grid(row=2, column=0, sticky="nsew")

    ctk.CTkLabel(
        progress_section,
        text="Diario · 3/5 hábitos completos",
        text_color=MECH_TEXT,
        font=("Segoe UI", 11),
    ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 6))
    daily_bar = _make_progress(progress_section, 0.6)
    daily_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))

    ctk.CTkLabel(
        progress_section,
        text="Semanal · 12/20 hábitos completos",
        text_color=MECH_TEXT,
        font=("Segoe UI", 11),
    ).grid(row=3, column=0, sticky="w", padx=14, pady=(0, 6))
    weekly_bar = _make_progress(progress_section, 0.6)
    weekly_bar.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))

    close_button = ctk.CTkButton(
        window,
        text="Cerrar",
        command=window.destroy,
        fg_color=MECH_CARD,
        hover_color=MECH_BTN_HOVER,
        text_color=MECH_TEXT,
        corner_radius=12,
    )
    close_button.pack(anchor="e", padx=24, pady=(0, 18))

    window.lift()
    window.focus()
