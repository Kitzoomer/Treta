import os
import json
import threading
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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


class MagicHub(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        self.title("TRETA · Magic Hub (Commander)")
        self.geometry("1200x720")
        self.minsize(1050, 650)

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
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            style.theme_use("clam")
        style.configure("TButton", padding=(10, 6))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Big.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 10))

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="TRETA · Magic Hub (Commander)", style="Header.TLabel").pack(side="left")

        ttk.Label(top, text="Jugadores:", padding=(20, 0)).pack(side="left")
        self.players_var = tk.StringVar(value=str(self.game["meta"]["players"]))
        ttk.Combobox(top, textvariable=self.players_var, values=["2", "4"], width=3, state="readonly").pack(side="left")
        ttk.Button(top, text="Aplicar", command=self._apply_players).pack(side="left", padx=6)

        ttk.Label(top, text="Vida inicial:", padding=(20, 0)).pack(side="left")
        self.startlife_var = tk.StringVar(value=str(self.game["meta"]["starting_life"]))
        ttk.Entry(top, textvariable=self.startlife_var, width=5).pack(side="left")
        ttk.Button(top, text="Set", command=self._apply_startlife).pack(side="left", padx=6)

        ttk.Button(top, text="Reset partida", command=self._reset_game).pack(side="right")
        ttk.Button(top, text="Undo", command=self._undo).pack(side="right", padx=8)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_game = ttk.Frame(nb)
        self.tab_judge = ttk.Frame(nb)
        self.tab_music = ttk.Frame(nb)
        self.tab_history = ttk.Frame(nb)

        nb.add(self.tab_game, text="Juego")
        nb.add(self.tab_judge, text="Juez")
        nb.add(self.tab_music, text="Música")
        nb.add(self.tab_history, text="Historial")

        self._build_tab_game()
        self._build_tab_judge()
        self._build_tab_music()
        self._build_tab_history()

    # --------------------- Game tab ---------------------
    def _build_tab_game(self):
        root = ttk.Frame(self.tab_game, padding=10)
        root.pack(fill="both", expand=True)

        turnbar = ttk.Frame(root)
        turnbar.pack(fill="x", pady=(0, 10))
        self.turn_label = ttk.Label(turnbar, text="", style="Header.TLabel")
        self.turn_label.pack(side="left")

        ttk.Button(turnbar, text="🎲 Quién empieza", command=self._roll_who_starts).pack(side="right")
        ttk.Button(turnbar, text="Siguiente fase", command=self._next_phase).pack(side="right", padx=6)
        ttk.Button(turnbar, text="Siguiente turno", command=self._next_turn).pack(side="right", padx=6)

        self.phase_label = ttk.Label(turnbar, text="", style="Small.TLabel", padding=(20, 0))
        self.phase_label.pack(side="left")

        self.players_frame = ttk.Frame(root)
        self.players_frame.pack(fill="both", expand=True)

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
        self.turn_label.config(text=f"Turno {t['number']} · Activo: Jugador {t['active_player']+1}")
        self.phase_label.config(text=f"Fase: {self._phase_name(t['phase_index'])}")

        for w in self.players_frame.winfo_children():
            w.destroy()

        players = self.game["meta"]["players"]
        if players == 2:
            self._build_players_2()
        else:
            self._build_players_4()

    def _build_player_card(self, parent, idx: int):
        p = self.game["players"][idx]
        card = ttk.Frame(parent, padding=10, relief="ridge")
        card.columnconfigure(0, weight=1)

        name_var = tk.StringVar(value=p["name"])
        ttk.Entry(card, textvariable=name_var).grid(row=0, column=0, sticky="ew")

        def save_name():
            self._snapshot()
            self.game["players"][idx]["name"] = name_var.get().strip() or f"Jugador {idx+1}"
            self._log_event("set_name", {"player": idx, "name": self.game["players"][idx]["name"]})
            self._save_game()

        ttk.Button(card, text="Guardar nombre", command=save_name).grid(row=0, column=1, padx=6)

        ttk.Label(card, text=f"Vida: {p['life']}", style="Big.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))

        def set_life(delta):
            self._snapshot()
            self.game["players"][idx]["life"] += delta
            self._log_event("life_change", {"player": idx, "delta": delta, "life": self.game["players"][idx]["life"]})
            self._sync_ui_from_game()
            self._save_game()

        btnrow = ttk.Frame(card)
        btnrow.grid(row=2, column=0, sticky="w", pady=(6, 0))
        for txt, d in [("-5", -5), ("-1", -1), ("+1", 1), ("+5", 5)]:
            ttk.Button(btnrow, text=txt, command=lambda dd=d: set_life(dd)).pack(side="left", padx=3)

        pe = ttk.Frame(card)
        pe.grid(row=3, column=0, sticky="w", pady=(10, 0))

        ttk.Label(pe, text=f"Poison: {p['poison']}").pack(side="left")

        def poison(delta):
            self._snapshot()
            self.game["players"][idx]["poison"] = max(0, self.game["players"][idx]["poison"] + delta)
            self._log_event("poison_change", {"player": idx, "delta": delta, "poison": self.game["players"][idx]["poison"]})
            self._sync_ui_from_game()
            self._save_game()

        ttk.Button(pe, text="-", command=lambda: poison(-1)).pack(side="left", padx=4)
        ttk.Button(pe, text="+", command=lambda: poison(1)).pack(side="left", padx=4)

        ttk.Label(pe, text=f"   Energy: {p['energy']}").pack(side="left")

        def energy(delta):
            self._snapshot()
            self.game["players"][idx]["energy"] = max(0, self.game["players"][idx]["energy"] + delta)
            self._log_event("energy_change", {"player": idx, "delta": delta, "energy": self.game["players"][idx]["energy"]})
            self._sync_ui_from_game()
            self._save_game()

        ttk.Button(pe, text="-", command=lambda: energy(-1)).pack(side="left", padx=4)
        ttk.Button(pe, text="+", command=lambda: energy(1)).pack(side="left", padx=4)

        tax = ttk.Frame(card)
        tax.grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Label(tax, text=f"Impuesto comandante: +{p['cmd_tax']}").pack(side="left")

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

        ttk.Button(tax, text="Commander murió (+2)", command=cmd_died).pack(side="left", padx=6)
        ttk.Button(tax, text="Reset impuesto", command=cmd_tax_reset).pack(side="left", padx=6)

        return card

    def _build_cmd_damage_panel(self, parent):
        players = self.game["meta"]["players"]
        frame = ttk.Frame(parent, padding=10, relief="ridge")
        ttk.Label(frame, text="Commander Damage", style="Header.TLabel").pack(anchor="w")

        grid = ttk.Frame(frame)
        grid.pack(fill="x", pady=(6, 0))

        ttk.Label(grid, text="Ataque\\Defensa").grid(row=0, column=0, padx=4, pady=4)
        for d in range(players):
            ttk.Label(grid, text=f"J{d+1}", style="Small.TLabel").grid(row=0, column=d+1, padx=4, pady=4)

        for a in range(players):
            ttk.Label(grid, text=f"J{a+1}").grid(row=a+1, column=0, padx=4, pady=4, sticky="e")
            for d in range(players):
                if a == d:
                    ttk.Label(grid, text="—").grid(row=a+1, column=d+1, padx=4, pady=4)
                    continue

                val = self.game["cmd_damage"][a][d]
                cell = ttk.Frame(grid)
                cell.grid(row=a+1, column=d+1, padx=3, pady=3)

                ttk.Label(cell, text=str(val)).pack(side="left", padx=4)

                def add(att=a, de=d, delta=1):
                    self._snapshot()
                    self.game["cmd_damage"][att][de] = max(0, self.game["cmd_damage"][att][de] + delta)
                    self._log_event("cmd_damage_change", {"attacker": att, "defender": de, "delta": delta, "value": self.game["cmd_damage"][att][de]})
                    self._sync_ui_from_game()
                    self._save_game()

                ttk.Button(cell, text="-", width=2, command=lambda att=a, de=d: add(att, de, -1)).pack(side="left")
                ttk.Button(cell, text="+", width=2, command=lambda att=a, de=d: add(att, de, 1)).pack(side="left")

        ttk.Button(frame, text="Reset commander damage", command=self._reset_cmd_damage).pack(anchor="w", pady=(10, 0))
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
        container = ttk.Frame(self.players_frame)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        self._build_player_card(container, 0).grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._build_player_card(container, 1).grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self._build_cmd_damage_panel(container).grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

    def _build_players_4(self):
        container = ttk.Frame(self.players_frame)
        container.pack(fill="both", expand=True)
        for c in range(2):
            container.columnconfigure(c, weight=1)
        for r in range(3):
            container.rowconfigure(r, weight=1)

        for i in range(4):
            self._build_player_card(container, i).grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)

        self._build_cmd_damage_panel(container).grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

    # --------------------- Judge tab ---------------------
    def _build_tab_judge(self):
        root = ttk.Frame(self.tab_judge, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        left = ttk.Frame(root, padding=10, relief="ridge")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(left, text="Juez (Reglas)", style="Header.TLabel").pack(anchor="w")

        self.rules_entry = ttk.Entry(left)
        self.rules_entry.pack(fill="x", pady=(8, 6))

        ttk.Button(left, text="Responder (IA)", command=self._judge_answer).pack(anchor="w")

        out_wrap = ttk.Frame(left)
        out_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.rules_out = tk.Text(out_wrap, height=18, wrap="word", font=("Segoe UI", 10))
        ys = ttk.Scrollbar(out_wrap, orient="vertical", command=self.rules_out.yview)
        self.rules_out.configure(yscrollcommand=ys.set)
        self.rules_out.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")

        right = ttk.Frame(root, padding=10, relief="ridge")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(right, text="Cartas (Scryfall + Gatherer)", style="Header.TLabel").pack(anchor="w")

        self.card_entry = ttk.Entry(right)
        self.card_entry.pack(fill="x", pady=(8, 6))
        btns = ttk.Frame(right)
        btns.pack(fill="x")
        ttk.Button(btns, text="Buscar carta", command=self._card_search).pack(side="left")
        ttk.Button(btns, text="Abrir Gatherer", command=self._open_gatherer).pack(side="left", padx=8)

        card_wrap = ttk.Frame(right)
        card_wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.card_out = tk.Text(card_wrap, height=18, wrap="word", font=("Segoe UI", 10))
        ys2 = ttk.Scrollbar(card_wrap, orient="vertical", command=self.card_out.yview)
        self.card_out.configure(yscrollcommand=ys2.set)
        self.card_out.pack(side="left", fill="both", expand=True)
        ys2.pack(side="right", fill="y")

        hint = ttk.Label(
            root,
            text="Modo juez: Commander por defecto (si no indicas otro formato). No muestra fuentes, solo respuesta.",
            style="Small.TLabel",
        )
        hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

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
        root = ttk.Frame(self.tab_music, padding=10)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Música (local)", style="Header.TLabel").pack(anchor="w")
        ttk.Label(root, text=f"Carpeta: {MUSIC_DIR}", style="Small.TLabel").pack(anchor="w", pady=(0, 10))

        controls = ttk.Frame(root)
        controls.pack(fill="x")

        ttk.Button(controls, text="Cargar carpeta", command=self._music_load_folder).pack(side="left")
        ttk.Button(controls, text="⏯ Play/Pause", command=self._music_toggle).pack(side="left", padx=8)
        ttk.Button(controls, text="⏭ Next", command=self._music_next).pack(side="left")
        ttk.Button(controls, text="🔀 Shuffle", command=self._music_shuffle).pack(side="left", padx=8)

        ttk.Label(controls, text="Volumen").pack(side="left", padx=(20, 6))
        self.vol_var = tk.DoubleVar(value=self.music_volume)
        vol = ttk.Scale(controls, from_=0.0, to=1.0, variable=self.vol_var, command=self._music_set_volume)
        vol.pack(side="left", fill="x", expand=True)

        self.music_status = ttk.Label(root, text="Backend: (no cargado)", style="Small.TLabel")
        self.music_status.pack(anchor="w", pady=(10, 0))

        self.music_list = tk.Listbox(root, height=18)
        self.music_list.pack(fill="both", expand=True, pady=(8, 0))

        hint = ttk.Label(
            root,
            text="Nota: para mp3/ogg recomiendo instalar pygame:  pip install pygame\nSi no, Treta lo indicará aquí.",
            style="Small.TLabel",
        )
        hint.pack(anchor="w", pady=(10, 0))

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
            self.music_status.config(text="Backend: NO disponible. Instala pygame (pip install pygame).")
            return

        exts = (".mp3", ".ogg", ".wav")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        self.music_playlist = files
        self.music_current_idx = 0
        self.music_list.delete(0, "end")
        for f in files:
            self.music_list.insert("end", os.path.basename(f))

        self.music_status.config(text=f"Backend: pygame · {len(files)} pistas cargadas")
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
            self.music_status.config(text=f"Reproduciendo: {os.path.basename(path)}")
            self.music_list.selection_clear(0, "end")
            self.music_list.selection_set(idx)
            self.music_list.see(idx)
            self._log_event("music_play", {"track": path})
        except Exception as e:
            self.music_status.config(text=f"Error música: {e}")

    def _music_toggle(self):
        if not self._music_try_init():
            self.music_status.config(text="Backend: NO disponible. Instala pygame (pip install pygame).")
            return
        if not self.music_playlist:
            self.music_status.config(text="Carga una carpeta con música primero.")
            return
        try:
            import pygame
            if self.music_playing:
                pygame.mixer.music.pause()
                self.music_playing = False
                self.music_status.config(text="Pausado")
                self._log_event("music_pause", {})
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                    self.music_playing = True
                    self.music_status.config(text="Reanudado")
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
        root = ttk.Frame(self.tab_history, padding=10)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Historial de eventos", style="Header.TLabel").pack(anchor="w")

        btns = ttk.Frame(root)
        btns.pack(fill="x", pady=(10, 6))
        ttk.Button(btns, text="Recargar", command=self._reload_history_ui).pack(side="left")
        ttk.Button(btns, text="Guardar partida…", command=self._save_as).pack(side="left", padx=8)
        ttk.Button(btns, text="Cargar partida…", command=self._load_from).pack(side="left")

        self.history_text = tk.Text(root, wrap="word")
        self.history_text.pack(fill="both", expand=True)
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
