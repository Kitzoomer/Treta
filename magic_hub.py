import os
import json
import threading
import random
import tkinter as tk
from tkinter import messagebox, filedialog

import customtkinter as ctk

from ui_theme import (
    MECH_BG,
    MECH_BORDER,
    MECH_BTN_HOVER,
    MECH_CARD,
    MECH_CARD_2,
    MECH_MUTED,
    MECH_PANEL,
    MECH_RED,
    MECH_RED_DARK,
    MECH_TEXT,
    apply_treta_theme,
)
import re

from judge_ai import RulesCache, answer_question

from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import quote


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "magic")
SAVES_DIR = os.path.join(DATA_DIR, "saves")
MUSIC_DIR = os.path.join(BASE_DIR, "music", "magic")

SCRYFALL_NAMED = "https://api.scryfall.com/cards/named?fuzzy="


def ensure_dirs():
    os.makedirs(SAVES_DIR, exist_ok=True)
    os.makedirs(MUSIC_DIR, exist_ok=True)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_history(path, event):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def fetch_text(url: str, timeout=12) -> str:
    req = Request(url, headers={"User-Agent": "TretaMagicHub/2.2"})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw.decode("utf-8", errors="ignore")


def scryfall_lookup(name: str) -> dict | None:
    try:
        url = SCRYFALL_NAMED + quote(name)
        txt = fetch_text(url, timeout=12)
        data = json.loads(txt)
        if data.get("object") == "error":
            return None
        return data
    except Exception:
        return None


def gatherer_link(name: str) -> str:
    return "https://gatherer.wizards.com/Pages/Search/Default.aspx?name=+%5b" + quote(name) + "%5d"


def guess_card_name_from_text(text: str) -> str | None:
    """
    Fix clave: en frases tipo "qué carta es Animar?"
    NO debe capturar "es Animar", debe capturar "Animar".
    """
    t = (text or "").strip()
    if not t:
        return None

    # 1) comillas
    m = re.search(r'["“”](.{2,80}?)["“”]', t)
    if m:
        cand = m.group(1).strip()
        if 2 <= len(cand) <= 80:
            return cand

    m = re.search(r"[‘’'](.{2,80}?)[‘’']", t)
    if m:
        cand = m.group(1).strip()
        if 2 <= len(cand) <= 80:
            return cand

    # 2) "carta (es)? X"  ✅ aquí va el arreglo
    m = re.search(r"\b(carta|card)\s+(?:es\s+)?([A-Za-z0-9][A-Za-z0-9 ,'\-]{2,80})", t, flags=re.I)
    if m:
        cand = m.group(2).strip()
        cand = re.split(r"[?.!;:]", cand)[0].strip()
        cand = cand[:60].strip()
        if len(cand) >= 2:
            return cand

    # 3) Título capitalizado (2-5 palabras)
    m = re.search(r"\b([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){1,4})\b", t)
    if m:
        cand = m.group(1).strip()
        bad = {"Magic", "Commander", "Fase", "Turno", "Jugador", "Reglas"}
        if cand.split()[0] not in bad and len(cand) <= 60:
            return cand

    # 4) Una sola palabra con mayúscula (Animar / Atraxa)
    m = re.search(r"\b([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ'’\-]{2,25})\b", t)
    if m:
        cand = m.group(1).strip()
        bad2 = {"Que", "Qué", "Como", "Cómo", "Cuando", "Cuándo", "Reglas", "Carta", "Cartas"}
        if cand not in bad2:
            return cand

    return None


class MagicHub(ctk.CTk):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        apply_treta_theme()

        self.title("TRETA · Magic Hub (Commander)")
        self.geometry("1200x720")
        self.minsize(1050, 650)
        self.configure(fg_color=MECH_BG)

        self.state_path = os.path.join(DATA_DIR, "current_game.json")
        self.history_path = os.path.join(DATA_DIR, "history.jsonl")

        self.game = self._default_game(players=2)
        self.undo_stack = []

        # Música
        self.music_backend = None
        self.music_playlist = []
        self.music_current_idx = 0
        self.music_playing = False
        self.music_volume = 0.6

        # cache reglas
        self.rules_cache = RulesCache()

        self._init_styles()
        self._build_ui()
        self._load_on_start()

    # --------------------- State ---------------------
    def _default_game(self, players=2):
        players = 2 if players not in (2, 4) else players
        return {
            "meta": {"format": "commander", "players": players, "starting_life": 40, "created_at": now_iso()},
            "turn": {"number": 1, "active_player": 0, "phase_index": 0},
            "players": [{"name": f"Jugador {i+1}", "life": 40, "poison": 0, "energy": 0, "cmd_tax": 0} for i in range(players)],
            "cmd_damage": [[0 for _ in range(players)] for _ in range(players)],
        }

    def _snapshot(self):
        self.undo_stack.append(json.loads(json.dumps(self.game)))
        if len(self.undo_stack) > 200:
            self.undo_stack = self.undo_stack[-200:]

    def _save_game(self):
        write_json(self.state_path, self.game)

    def _load_on_start(self):
        g = read_json(self.state_path, None)
        if g and isinstance(g, dict) and "meta" in g and "players" in g:
            self.game = g
        self._sync_ui_from_game()
        self._reload_history_ui()

    def _reset_game(self):
        if not messagebox.askyesno("Reset", "¿Resetear partida?"):
            return
        players = int(self.players_var.get())
        self.game = self._default_game(players=players)
        self.undo_stack.clear()
        self._log_event("reset_game", {"players": players})
        self._sync_ui_from_game()
        self._save_game()

    def _undo(self):
        if not self.undo_stack:
            messagebox.showinfo("Undo", "No hay nada que deshacer.")
            return
        self.game = self.undo_stack.pop()
        self._log_event("undo", {})
        self._sync_ui_from_game()
        self._save_game()

    # --------------------- History ---------------------
    def _log_event(self, typ: str, payload: dict):
        event = {"ts": now_iso(), "type": typ, "payload": payload, "meta": {"format": "commander"}}
        append_history(self.history_path, event)
        self._append_history_ui(event)

    def _append_history_ui(self, event: dict):
        if not hasattr(self, "history_text"):
            return
        s = f"[{event['ts']}] {event['type']}: {event.get('payload', {})}\n"
        self.history_text.configure(state="normal")
        self.history_text.insert("end", s)
        self.history_text.see("end")
        self.history_text.configure(state="disabled")

    def _reload_history_ui(self):
        if not hasattr(self, "history_text"):
            return
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.configure(state="disabled")
        if not os.path.exists(self.history_path):
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._append_history_ui(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass

    # --------------------- UI ---------------------
    def _init_styles(self):
        # Reservado para estilos de widgets Tk clásicos si se necesitan.
        pass

    def _build_ui(self):
        top = ctk.CTkFrame(self, fg_color=MECH_PANEL, corner_radius=18)
        top.pack(fill="x", padx=18, pady=(18, 10))
        top.grid_columnconfigure(10, weight=1)

        ctk.CTkLabel(
            top,
            text="TRETA · Magic Hub (Commander)",
            text_color=MECH_TEXT,
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=12)

        ctk.CTkLabel(
            top,
            text="Jugadores:",
            text_color=MECH_MUTED,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=1, sticky="w", padx=(20, 6))
        self.players_var = tk.StringVar(value=str(self.game["meta"]["players"]))
        ctk.CTkComboBox(
            top,
            values=["2", "4"],
            variable=self.players_var,
            width=80,
            state="readonly",
            fg_color=MECH_CARD,
            border_color=MECH_BORDER,
            button_color=MECH_CARD_2,
            text_color=MECH_TEXT,
        ).grid(row=0, column=2, sticky="w")
        ctk.CTkButton(
            top,
            text="Aplicar",
            command=self._apply_players,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=3, sticky="w", padx=8)

        ctk.CTkLabel(
            top,
            text="Vida inicial:",
            text_color=MECH_MUTED,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=4, sticky="w", padx=(20, 6))
        self.startlife_var = tk.StringVar(value=str(self.game["meta"]["starting_life"]))
        ctk.CTkEntry(
            top,
            textvariable=self.startlife_var,
            width=90,
            fg_color=MECH_CARD,
            border_color=MECH_BORDER,
            text_color=MECH_TEXT,
        ).grid(row=0, column=5, sticky="w")
        ctk.CTkButton(
            top,
            text="Set",
            command=self._apply_startlife,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=6, sticky="w", padx=8)

        ctk.CTkButton(
            top,
            text="Undo",
            command=self._undo,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=8, sticky="e", padx=(8, 6))
        ctk.CTkButton(
            top,
            text="Reset partida",
            command=self._reset_game,
            fg_color=MECH_RED,
            hover_color=MECH_RED_DARK,
            text_color="white",
            corner_radius=12,
        ).grid(row=0, column=9, sticky="e", padx=(0, 12))

        self.tabs = ctk.CTkTabview(
            self,
            fg_color=MECH_PANEL,
            segmented_button_fg_color=MECH_CARD,
            segmented_button_selected_color=MECH_RED,
            segmented_button_unselected_color=MECH_CARD_2,
            segmented_button_selected_hover_color=MECH_RED_DARK,
            segmented_button_unselected_hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
        )
        self.tabs.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.tab_game = self.tabs.add("Juego")
        self.tab_judge = self.tabs.add("Juez")
        self.tab_music = self.tabs.add("Música")
        self.tab_history = self.tabs.add("Historial")

        self._build_tab_game()
        self._build_tab_judge()
        self._build_tab_music()
        self._build_tab_history()

    # --------------------- Game tab ---------------------
    def _build_tab_game(self):
        root = ctk.CTkFrame(self.tab_game, fg_color=MECH_PANEL, corner_radius=18)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        turnbar = ctk.CTkFrame(root, fg_color=MECH_CARD_2, corner_radius=16)
        turnbar.pack(fill="x", pady=(0, 10), padx=8)
        turnbar.grid_columnconfigure(4, weight=1)

        self.turn_label = ctk.CTkLabel(
            turnbar,
            text="",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        )
        self.turn_label.grid(row=0, column=0, sticky="w", padx=12, pady=10)

        self.phase_label = ctk.CTkLabel(
            turnbar,
            text="",
            text_color=MECH_MUTED,
            font=("Segoe UI", 11),
        )
        self.phase_label.grid(row=0, column=1, sticky="w", padx=(8, 12))

        ctk.CTkButton(
            turnbar,
            text="Siguiente turno",
            command=self._next_turn,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=2, sticky="e", padx=6, pady=8)
        ctk.CTkButton(
            turnbar,
            text="Siguiente fase",
            command=self._next_phase,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=3, sticky="e", padx=6, pady=8)
        ctk.CTkButton(
            turnbar,
            text="🎲 Quién empieza",
            command=self._roll_who_starts,
            fg_color=MECH_RED,
            hover_color=MECH_RED_DARK,
            text_color="white",
            corner_radius=12,
        ).grid(row=0, column=5, sticky="e", padx=12, pady=8)

        self.players_frame = ctk.CTkFrame(root, fg_color=MECH_PANEL, corner_radius=18)
        self.players_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _apply_players(self):
        players = int(self.players_var.get())
        if players not in (2, 4):
            return
        if not messagebox.askyesno("Jugadores", f"¿Cambiar a {players} jugadores? (resetea partida)"):
            return
        self.game = self._default_game(players=players)
        self.undo_stack.clear()
        self._log_event("set_players", {"players": players})
        self._sync_ui_from_game()
        self._save_game()

    def _apply_startlife(self):
        try:
            v = int(self.startlife_var.get())
            if v < 1 or v > 999:
                raise ValueError()
        except Exception:
            messagebox.showerror("Vida inicial", "Pon un número válido (1..999).")
            return

        self._snapshot()
        self.game["meta"]["starting_life"] = v
        for p in self.game["players"]:
            p["life"] = v

        self._log_event("set_starting_life", {"life": v})
        self._sync_ui_from_game()
        self._save_game()

    def _roll_who_starts(self):
        players = self.game["meta"]["players"]
        rolls = [random.randint(1, 20) for _ in range(players)]
        winner = max(range(players), key=lambda i: rolls[i])
        while rolls.count(rolls[winner]) > 1:
            tied = [i for i in range(players) if rolls[i] == rolls[winner]]
            for i in tied:
                rolls[i] = random.randint(1, 20)
            winner = max(range(players), key=lambda i: rolls[i])

        msg = " · ".join([f"J{i+1}:{rolls[i]}" for i in range(players)])
        self._log_event("roll_start", {"rolls": rolls, "winner": winner})
        messagebox.showinfo("🎲 Quién empieza", f"{msg}\n\nEmpieza Jugador {winner+1} ✅")

        self._snapshot()
        self.game["turn"]["active_player"] = winner
        self.game["turn"]["number"] = 1
        self.game["turn"]["phase_index"] = 0
        self._sync_ui_from_game()
        self._save_game()

    def _next_turn(self):
        self._snapshot()
        players = self.game["meta"]["players"]
        self.game["turn"]["number"] += 1
        self.game["turn"]["active_player"] = (self.game["turn"]["active_player"] + 1) % players
        self.game["turn"]["phase_index"] = 0
        self._log_event("next_turn", dict(self.game["turn"]))
        self._sync_ui_from_game()
        self._save_game()

    def _next_phase(self):
        self._snapshot()
        self.game["turn"]["phase_index"] = (self.game["turn"]["phase_index"] + 1) % 4
        self._log_event("next_phase", dict(self.game["turn"]))
        self._sync_ui_from_game()
        self._save_game()

    def _phase_name(self, idx: int) -> str:
        return ["Beginning", "Main", "Combat", "End"][idx % 4]

    def _sync_ui_from_game(self):
        t = self.game["turn"]
        self.turn_label.configure(text=f"Turno {t['number']} · Activo: Jugador {t['active_player']+1}")
        self.phase_label.configure(text=f"Fase: {self._phase_name(t['phase_index'])}")

        for w in self.players_frame.winfo_children():
            w.destroy()

        players = self.game["meta"]["players"]
        if players == 2:
            self._build_players_2()
        else:
            self._build_players_4()

    def _build_player_card(self, parent, idx: int):
        p = self.game["players"][idx]
        card = ctk.CTkFrame(parent, fg_color=MECH_CARD, corner_radius=16)
        card.grid_columnconfigure(0, weight=1)

        name_var = tk.StringVar(value=p["name"])
        ctk.CTkEntry(
            card,
            textvariable=name_var,
            fg_color=MECH_CARD_2,
            border_color=MECH_BORDER,
            text_color=MECH_TEXT,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        def save_name():
            self._snapshot()
            self.game["players"][idx]["name"] = name_var.get().strip() or f"Jugador {idx+1}"
            self._log_event("set_name", {"player": idx, "name": self.game["players"][idx]["name"]})
            self._save_game()

        ctk.CTkButton(
            card,
            text="Guardar nombre",
            command=save_name,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=0, column=1, padx=10, pady=(10, 4))

        ctk.CTkLabel(
            card,
            text=f"Vida: {p['life']}",
            text_color=MECH_TEXT,
            font=("Segoe UI", 16, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=10, pady=(8, 0))

        def set_life(delta):
            self._snapshot()
            self.game["players"][idx]["life"] += delta
            self._log_event("life_change", {"player": idx, "delta": delta, "life": self.game["players"][idx]["life"]})
            self._sync_ui_from_game()
            self._save_game()

        btnrow = ctk.CTkFrame(card, fg_color="transparent")
        btnrow.grid(row=2, column=0, sticky="w", padx=10, pady=(6, 0))
        for txt, d in [("-5", -5), ("-1", -1), ("+1", 1), ("+5", 5)]:
            ctk.CTkButton(
                btnrow,
                text=txt,
                command=lambda dd=d: set_life(dd),
                width=48,
                fg_color=MECH_CARD_2,
                hover_color=MECH_BTN_HOVER,
                text_color=MECH_TEXT,
                corner_radius=12,
            ).pack(side="left", padx=3)

        pe = ctk.CTkFrame(card, fg_color="transparent")
        pe.grid(row=3, column=0, sticky="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(
            pe,
            text=f"Poison: {p['poison']}",
            text_color=MECH_TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        def poison(delta):
            self._snapshot()
            self.game["players"][idx]["poison"] = max(0, self.game["players"][idx]["poison"] + delta)
            self._log_event("poison_change", {"player": idx, "delta": delta, "poison": self.game["players"][idx]["poison"]})
            self._sync_ui_from_game()
            self._save_game()

        ctk.CTkButton(
            pe,
            text="-",
            command=lambda: poison(-1),
            width=28,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=10,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            pe,
            text="+",
            command=lambda: poison(1),
            width=28,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=10,
        ).pack(side="left", padx=4)

        ctk.CTkLabel(
            pe,
            text=f"   Energy: {p['energy']}",
            text_color=MECH_TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        def energy(delta):
            self._snapshot()
            self.game["players"][idx]["energy"] = max(0, self.game["players"][idx]["energy"] + delta)
            self._log_event("energy_change", {"player": idx, "delta": delta, "energy": self.game["players"][idx]["energy"]})
            self._sync_ui_from_game()
            self._save_game()

        ctk.CTkButton(
            pe,
            text="-",
            command=lambda: energy(-1),
            width=28,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=10,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            pe,
            text="+",
            command=lambda: energy(1),
            width=28,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=10,
        ).pack(side="left", padx=4)

        tax = ctk.CTkFrame(card, fg_color="transparent")
        tax.grid(row=4, column=0, sticky="w", padx=10, pady=(10, 10))
        ctk.CTkLabel(
            tax,
            text=f"Impuesto comandante: +{p['cmd_tax']}",
            text_color=MECH_TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        def cmd_died():
            self._snapshot()
            self.game["players"][idx]["cmd_tax"] += 2
            self._log_event("cmd_tax_add", {"player": idx, "cmd_tax": self.game["players"][idx]["cmd_tax"]})
            self._sync_ui_from_game()
            self._save_game()

        def cmd_tax_reset():
            self._snapshot()
            self.game["players"][idx]["cmd_tax"] = 0
            self._log_event("cmd_tax_reset", {"player": idx})
            self._sync_ui_from_game()
            self._save_game()

        ctk.CTkButton(
            tax,
            text="Commander murió (+2)",
            command=cmd_died,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            tax,
            text="Reset impuesto",
            command=cmd_tax_reset,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=6)

        return card

    def _build_cmd_damage_panel(self, parent):
        players = self.game["meta"]["players"]
        frame = ctk.CTkFrame(parent, fg_color=MECH_CARD, corner_radius=16)
        ctk.CTkLabel(
            frame,
            text="Commander Damage",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 6))

        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="x", padx=6, pady=(0, 6))

        ctk.CTkLabel(
            grid,
            text="Ataque\\Defensa",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, padx=4, pady=4)
        for d in range(players):
            ctk.CTkLabel(
                grid,
                text=f"J{d+1}",
                text_color=MECH_MUTED,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=0, column=d + 1, padx=4, pady=4)

        for a in range(players):
            ctk.CTkLabel(
                grid,
                text=f"J{a+1}",
                text_color=MECH_MUTED,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=a + 1, column=0, padx=4, pady=4, sticky="e")
            for d in range(players):
                if a == d:
                    ctk.CTkLabel(
                        grid,
                        text="—",
                        text_color=MECH_MUTED,
                        font=("Segoe UI", 10),
                    ).grid(row=a + 1, column=d + 1, padx=4, pady=4)
                    continue

                val = self.game["cmd_damage"][a][d]
                cell = ctk.CTkFrame(grid, fg_color=MECH_CARD_2, corner_radius=12)
                cell.grid(row=a + 1, column=d + 1, padx=3, pady=3, sticky="ew")

                ctk.CTkLabel(
                    cell,
                    text=str(val),
                    text_color=MECH_TEXT,
                    font=("Segoe UI", 11, "bold"),
                ).pack(side="left", padx=6, pady=6)

                def add(att=a, de=d, delta=1):
                    self._snapshot()
                    self.game["cmd_damage"][att][de] = max(0, self.game["cmd_damage"][att][de] + delta)
                    self._log_event("cmd_damage_change", {"attacker": att, "defender": de, "delta": delta, "value": self.game["cmd_damage"][att][de]})
                    self._sync_ui_from_game()
                    self._save_game()

                ctk.CTkButton(
                    cell,
                    text="-",
                    width=28,
                    command=lambda att=a, de=d: add(att, de, -1),
                    fg_color=MECH_CARD,
                    hover_color=MECH_BTN_HOVER,
                    text_color=MECH_TEXT,
                    corner_radius=10,
                ).pack(side="left", padx=4, pady=4)
                ctk.CTkButton(
                    cell,
                    text="+",
                    width=28,
                    command=lambda att=a, de=d: add(att, de, 1),
                    fg_color=MECH_CARD,
                    hover_color=MECH_BTN_HOVER,
                    text_color=MECH_TEXT,
                    corner_radius=10,
                ).pack(side="left", padx=4, pady=4)

        ctk.CTkButton(
            frame,
            text="Reset commander damage",
            command=self._reset_cmd_damage,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(anchor="w", padx=10, pady=(6, 10))
        return frame

    def _reset_cmd_damage(self):
        if not messagebox.askyesno("Reset", "¿Resetear commander damage?"):
            return
        self._snapshot()
        players = self.game["meta"]["players"]
        self.game["cmd_damage"] = [[0 for _ in range(players)] for _ in range(players)]
        self._log_event("cmd_damage_reset", {})
        self._sync_ui_from_game()
        self._save_game()

    def _build_players_2(self):
        container = ctk.CTkFrame(self.players_frame, fg_color="transparent")
        container.pack(fill="both", expand=True)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        self._build_player_card(container, 0).grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._build_player_card(container, 1).grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self._build_cmd_damage_panel(container).grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

    def _build_players_4(self):
        container = ctk.CTkFrame(self.players_frame, fg_color="transparent")
        container.pack(fill="both", expand=True)
        for c in range(2):
            container.grid_columnconfigure(c, weight=1)
        for r in range(3):
            container.grid_rowconfigure(r, weight=1)

        for i in range(4):
            self._build_player_card(container, i).grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)

        self._build_cmd_damage_panel(container).grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

    # --------------------- Judge tab ---------------------
    def _build_tab_judge(self):
        root = ctk.CTkFrame(self.tab_judge, fg_color=MECH_PANEL, corner_radius=18)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)

        left = ctk.CTkFrame(root, fg_color=MECH_CARD, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        left.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(
            left,
            text="Juez (Reglas)",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.rules_entry = ctk.CTkEntry(
            left,
            fg_color=MECH_CARD_2,
            border_color=MECH_BORDER,
            text_color=MECH_TEXT,
        )
        self.rules_entry.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))

        ctk.CTkButton(
            left,
            text="Responder (IA)",
            command=self._judge_answer,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))

        out_wrap = ctk.CTkFrame(left, fg_color=MECH_CARD_2, corner_radius=12)
        out_wrap.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        out_wrap.grid_rowconfigure(0, weight=1)
        out_wrap.grid_columnconfigure(0, weight=1)
        self.rules_out = tk.Text(
            out_wrap,
            height=18,
            wrap="word",
            font=("Segoe UI", 10),
            bg=MECH_BG,
            fg=MECH_TEXT,
            insertbackground=MECH_TEXT,
            relief="flat",
        )
        ys = ctk.CTkScrollbar(out_wrap, orientation="vertical", command=self.rules_out.yview)
        self.rules_out.configure(yscrollcommand=ys.set)
        self.rules_out.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        ys.grid(row=0, column=1, sticky="ns", pady=6, padx=(0, 6))

        right = ctk.CTkFrame(root, fg_color=MECH_CARD, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        right.grid_rowconfigure(3, weight=1)
        ctk.CTkLabel(
            right,
            text="Cartas (Scryfall + Gatherer)",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.card_entry = ctk.CTkEntry(
            right,
            fg_color=MECH_CARD_2,
            border_color=MECH_BORDER,
            text_color=MECH_TEXT,
        )
        self.card_entry.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))
        ctk.CTkButton(
            btns,
            text="Buscar carta",
            command=self._card_search,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left")
        ctk.CTkButton(
            btns,
            text="Abrir Gatherer",
            command=self._open_gatherer,
            fg_color=MECH_CARD_2,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=8)

        card_wrap = ctk.CTkFrame(right, fg_color=MECH_CARD_2, corner_radius=12)
        card_wrap.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        card_wrap.grid_rowconfigure(0, weight=1)
        card_wrap.grid_columnconfigure(0, weight=1)
        self.card_out = tk.Text(
            card_wrap,
            height=18,
            wrap="word",
            font=("Segoe UI", 10),
            bg=MECH_BG,
            fg=MECH_TEXT,
            insertbackground=MECH_TEXT,
            relief="flat",
        )
        ys2 = ctk.CTkScrollbar(card_wrap, orientation="vertical", command=self.card_out.yview)
        self.card_out.configure(yscrollcommand=ys2.set)
        self.card_out.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        ys2.grid(row=0, column=1, sticky="ns", pady=6, padx=(0, 6))

        hint = ctk.CTkLabel(
            root,
            text="Modo juez: Commander por defecto (si no indicas otro formato). No muestra fuentes, solo respuesta.",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10),
        )
        hint.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))

    def _judge_answer(self):
        q = (self.rules_entry.get() or "").strip()
        if not q:
            return

        card_name = (self.card_entry.get() or "").strip() or None

        # ✅ autodetecta carta si el campo está vacío (y con fix “carta es X”)
        if not card_name:
            guessed = guess_card_name_from_text(q)
            if guessed and len(guessed) >= 3:
                card_name = guessed
                self.card_entry.delete(0, "end")
                self.card_entry.insert(0, card_name)

        self.rules_out.delete("1.0", "end")
        self.rules_out.insert("end", "Pensando…\n")

        def worker():
            ans = answer_question(q, self.rules_cache, card_name=card_name)
            self.after(0, lambda: (self.rules_out.delete("1.0", "end"), self.rules_out.insert("end", ans)))
            self._log_event("judge_ai", {"q": q, "card": card_name or ""})

        threading.Thread(target=worker, daemon=True).start()

    def _card_search(self):
        name = (self.card_entry.get() or "").strip()
        if not name:
            return
        self.card_out.delete("1.0", "end")
        self.card_out.insert("end", "Buscando carta…\n")

        def worker():
            data = scryfall_lookup(name)
            if not data:
                out = "No encontré la carta.\nPrueba nombre en inglés o más aproximado."
            else:
                oracle = data.get("oracle_text", "")
                mana = data.get("mana_cost", "")
                typ = data.get("type_line", "")
                cname = data.get("name", name)
                out = f"{cname}\n{mana}\n{typ}\n\n{oracle}"

            self.after(0, lambda: (self.card_out.delete("1.0", "end"), self.card_out.insert("end", out)))
            self._log_event("card_search", {"name": name})

        threading.Thread(target=worker, daemon=True).start()

    def _open_gatherer(self):
        name = (self.card_entry.get() or "").strip()
        if not name:
            return
        import webbrowser
        webbrowser.open(gatherer_link(name))
        self._log_event("open_gatherer", {"name": name})

    # --------------------- Music tab ---------------------
    def _build_tab_music(self):
        root = ctk.CTkFrame(self.tab_music, fg_color=MECH_PANEL, corner_radius=18)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(
            root,
            text="Música (local)",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 4))
        ctk.CTkLabel(
            root,
            text=f"Carpeta: {MUSIC_DIR}",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=12, pady=(0, 10))

        controls = ctk.CTkFrame(root, fg_color="transparent")
        controls.pack(fill="x", padx=12)

        ctk.CTkButton(
            controls,
            text="Cargar carpeta",
            command=self._music_load_folder,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left")
        ctk.CTkButton(
            controls,
            text="⏯ Play/Pause",
            command=self._music_toggle,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            controls,
            text="⏭ Next",
            command=self._music_next,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left")
        ctk.CTkButton(
            controls,
            text="🔀 Shuffle",
            command=self._music_shuffle,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=8)

        ctk.CTkLabel(
            controls,
            text="Volumen",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(20, 6))
        self.vol_var = tk.DoubleVar(value=self.music_volume)
        vol = ctk.CTkSlider(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.vol_var,
            command=self._music_set_volume,
            fg_color=MECH_CARD,
            progress_color=MECH_RED,
            button_color=MECH_RED,
        )
        vol.pack(side="left", fill="x", expand=True)

        self.music_status = ctk.CTkLabel(
            root,
            text="Backend: (no cargado)",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10),
        )
        self.music_status.pack(anchor="w", padx=12, pady=(10, 0))

        self.music_list = tk.Listbox(
            root,
            height=18,
            bg=MECH_BG,
            fg=MECH_TEXT,
            highlightthickness=0,
            relief="flat",
        )
        self.music_list.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        hint = ctk.CTkLabel(
            root,
            text="Nota: para mp3/ogg recomiendo instalar pygame:  pip install pygame\nSi no, Treta lo indicará aquí.",
            text_color=MECH_MUTED,
            font=("Segoe UI", 10),
        )
        hint.pack(anchor="w", padx=12, pady=(10, 12))

    def _music_try_init(self):
        if self.music_backend:
            return True
        try:
            import pygame
            pygame.mixer.init()
            self.music_backend = "pygame"
            self._music_apply_volume()
            return True
        except Exception:
            self.music_backend = None
            return False

    def _music_load_folder(self):
        folder = MUSIC_DIR
        os.makedirs(folder, exist_ok=True)

        if messagebox.askyesno("Carpeta", "¿Elegir otra carpeta distinta a la de Treta?"):
            chosen = filedialog.askdirectory()
            if chosen:
                folder = chosen

        ok = self._music_try_init()
        if not ok:
            self.music_status.configure(text="Backend: NO disponible. Instala pygame (pip install pygame).")
            return

        exts = (".mp3", ".ogg", ".wav")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        self.music_playlist = files
        self.music_current_idx = 0
        self.music_list.delete(0, "end")
        for f in files:
            self.music_list.insert("end", os.path.basename(f))

        self.music_status.configure(text=f"Backend: pygame · {len(files)} pistas cargadas")
        self._log_event("music_load", {"folder": folder, "count": len(files)})

    def _music_play_index(self, idx: int):
        if not self.music_backend or not self.music_playlist:
            return
        idx = idx % len(self.music_playlist)
        self.music_current_idx = idx
        path = self.music_playlist[idx]
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.music_playing = True
            self._music_apply_volume()
            self.music_status.configure(text=f"Reproduciendo: {os.path.basename(path)}")
            self.music_list.selection_clear(0, "end")
            self.music_list.selection_set(idx)
            self.music_list.see(idx)
            self._log_event("music_play", {"track": path})
        except Exception as e:
            self.music_status.configure(text=f"Error música: {e}")

    def _music_toggle(self):
        if not self._music_try_init():
            self.music_status.configure(text="Backend: NO disponible. Instala pygame (pip install pygame).")
            return
        if not self.music_playlist:
            self.music_status.configure(text="Carga una carpeta con música primero.")
            return
        try:
            import pygame
            if self.music_playing:
                pygame.mixer.music.pause()
                self.music_playing = False
                self.music_status.configure(text="Pausado")
                self._log_event("music_pause", {})
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                    self.music_playing = True
                    self.music_status.configure(text="Reanudado")
                    self._log_event("music_unpause", {})
                else:
                    self._music_play_index(self.music_current_idx)
        except Exception:
            pass

    def _music_next(self):
        if self.music_playlist:
            self._music_play_index(self.music_current_idx + 1)

    def _music_shuffle(self):
        if not self.music_playlist:
            return
        random.shuffle(self.music_playlist)
        self.music_current_idx = 0
        self.music_list.delete(0, "end")
        for f in self.music_playlist:
            self.music_list.insert("end", os.path.basename(f))
        self._log_event("music_shuffle", {})
        self._music_play_index(0)

    def _music_set_volume(self, *_):
        self.music_volume = float(self.vol_var.get())
        self._music_apply_volume()

    def _music_apply_volume(self):
        if not self.music_backend:
            return
        try:
            import pygame
            pygame.mixer.music.set_volume(max(0.0, min(1.0, self.music_volume)))
        except Exception:
            pass

    # --------------------- History tab ---------------------
    def _build_tab_history(self):
        root = ctk.CTkFrame(self.tab_history, fg_color=MECH_PANEL, corner_radius=18)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            root,
            text="Historial de eventos",
            text_color=MECH_TEXT,
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        btns = ctk.CTkFrame(root, fg_color="transparent")
        btns.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))
        ctk.CTkButton(
            btns,
            text="Recargar",
            command=self._reload_history_ui,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left")
        ctk.CTkButton(
            btns,
            text="Guardar partida…",
            command=self._save_as,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            btns,
            text="Cargar partida…",
            command=self._load_from,
            fg_color=MECH_CARD,
            hover_color=MECH_BTN_HOVER,
            text_color=MECH_TEXT,
            corner_radius=12,
        ).pack(side="left")

        self.history_text = tk.Text(
            root,
            wrap="word",
            bg=MECH_BG,
            fg=MECH_TEXT,
            insertbackground=MECH_TEXT,
            relief="flat",
        )
        self.history_text.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.history_text.configure(state="disabled")

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            initialdir=SAVES_DIR,
            defaultextension=".json",
            filetypes=[("Magic Save", "*.json")],
            title="Guardar partida",
        )
        if not path:
            return
        write_json(path, self.game)
        self._log_event("save_game", {"path": path})
        messagebox.showinfo("Guardar", "Partida guardada.")

    def _load_from(self):
        path = filedialog.askopenfilename(
            initialdir=SAVES_DIR,
            filetypes=[("Magic Save", "*.json")],
            title="Cargar partida",
        )
        if not path:
            return
        g = read_json(path, None)
        if not g or "meta" not in g or "players" not in g:
            messagebox.showerror("Cargar", "Archivo no válido.")
            return
        self._snapshot()
        self.game = g
        self.players_var.set(str(self.game["meta"].get("players", 2)))
        self.startlife_var.set(str(self.game["meta"].get("starting_life", 40)))
        self._log_event("load_game", {"path": path})
        self._sync_ui_from_game()
        self._save_game()

    def on_close(self):
        try:
            self._save_game()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MagicHub()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
