import os, json, time, threading, queue, subprocess
import sys
import argparse
import difflib
import re
import unicodedata
import wave
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import sounddevice as sd
import pyttsx3

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

WhisperModel = None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None
from faster_whisper import WhisperModel

from ui_theme import (
    MECH_BG,
    MECH_BORDER,
    MECH_BTN_HOVER,
    MECH_CARD,
    MECH_CARD_2,
    MECH_MUTED,
    MECH_OK,
    MECH_PANEL,
    MECH_RED,
    MECH_RED_DARK,
    MECH_TEXT,
)

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

BRIDGE_DIR = os.path.join(BASE_DIR, "bridge")
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIONS_DIR = os.path.join(BASE_DIR, "actions")
LOG_DIR = os.path.join(BASE_DIR, "logs")

LOG_PATH = os.path.join(LOG_DIR, "treta.log")
PENDING_CONFIRM_PATH = os.path.join(BRIDGE_DIR, "pending_confirm.json")

STATE_PATH = os.path.join(DATA_DIR, "state.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.jsonl")
DIARY_PATH = os.path.join(DATA_DIR, "diary.jsonl")
INTENTS_PATH = os.path.join(DATA_DIR, "intents.jsonl")
HISTORY_PATH = os.path.join(DATA_DIR, "history.jsonl")

CONFIRM_TIMEOUT_SEC = 5

DEFAULT_PANEL_BUTTONS_LEFT = [
    {"label": "Modo Trabajo", "action": "mode_work"},
    {"label": "Modo Perfil", "action": "mode_profile"},
    {"label": "Apuntar idea", "action": "idea_prompt"},
    {"label": "Reposo", "action": "presence_sleep"},
]

DEFAULT_PANEL_BUTTONS_RIGHT = [
    {"label": "Modo Magic", "action": "mode_magic"},
    {"label": "Modo Caos", "action": "mode_chaos"},
    {"label": "Presencia ON", "action": "presence_start"},
    {"label": "Apagar PC (DESACTIVADO)", "action": "shutdown_safe", "needs_confirm": True},
]

DEFAULT_ACTIONS = {
    "presence_start": {"type": "ps1", "path": "actions/presence_start.ps1"},
    "presence_sleep": {"type": "ps1", "path": "actions/presence_sleep.ps1"},
    "mode_work": {"type": "ps1", "path": "actions/mode_work.ps1"},
    "mode_magic": {"type": "ps1", "path": "actions/mode_magic.ps1"},
    "mode_chaos": {"type": "ps1", "path": "actions/mode_chaos.ps1"},
    "mode_profile": {"type": "py", "path": "actions/mode_profile.py"},
    "shutdown_safe": {"type": "ps1", "path": "actions/shutdown_safe.ps1"},
}


def _ensure_dirs():
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ACTIONS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    lowered = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if not audio.size or gain_db == 0:
        return audio
    gain = float(10 ** (gain_db / 20))
    boosted = audio * gain
    return np.clip(boosted, -1.0, 1.0)


def fuzzy_contains(text: str, pattern: str, threshold: float) -> bool:
    text_words = text.split()
    pattern_words = pattern.split()
    if not text_words or not pattern_words:
        return False
    if len(text_words) < len(pattern_words):
        candidate = " ".join(text_words)
        return difflib.SequenceMatcher(None, candidate, pattern).ratio() >= threshold
    for idx in range(len(text_words) - len(pattern_words) + 1):
        candidate = " ".join(text_words[idx : idx + len(pattern_words)])
        if difflib.SequenceMatcher(None, candidate, pattern).ratio() >= threshold:
            return True
    return False


def parse_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
def load_local_intents(cfg: dict) -> list[dict]:
    intents = cfg.get("local_intents", [])
    return intents if isinstance(intents, list) else []


def resolve_timezone(cfg: dict) -> tuple[datetime, str]:
    tz_name = (cfg.get("time_zone") or "local").strip()
    if not tz_name or tz_name.lower() == "local":
        return datetime.now(), ""
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return datetime.now(), ""
    try:
        return datetime.now(tz=ZoneInfo(tz_name)), f"({tz_name})"
    except Exception:
        return datetime.now(), ""


def render_local_response(template: str, cfg: dict) -> str:
    now, tz_label = resolve_timezone(cfg)
    weekday_map = {
        "monday": "lunes",
        "tuesday": "martes",
        "wednesday": "miércoles",
        "thursday": "jueves",
        "friday": "viernes",
        "saturday": "sábado",
        "sunday": "domingo",
    }
    weekday_es = weekday_map.get(now.strftime("%A").lower(), now.strftime("%A"))
    text = (
        template.replace("{time}", now.strftime("%H:%M"))
        .replace("{date}", now.strftime("%d/%m/%Y"))
        .replace("{weekday}", weekday_es)
        .replace("{timezone}", tz_label)
    )
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,!?])", r"\1", text)
def is_time_request(text: str) -> bool:
    if "hora" not in text:
        return False
    patterns = [
        r"\bque\s+hora\s+es\b",
        r"\bque\s+hora\s+son\b",
        r"\bque\s+hora\b",
        r"\bdime\s+la\s+hora\b",
        r"\bme\s+dices?\s+la\s+hora\b",
        r"\bhora\s+actual\b",
        r"\bla\s+hora\s+actual\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        return cfg
    actions = cfg.get("actions")
    if not isinstance(actions, dict):
        actions = {}
        cfg["actions"] = actions
    for key, spec in DEFAULT_ACTIONS.items():
        actions.setdefault(key, dict(spec))

    left_buttons = cfg.get("panel_buttons_left")
    if not isinstance(left_buttons, list):
        left_buttons = []
        cfg["panel_buttons_left"] = left_buttons
    existing_left = {btn.get("action") for btn in left_buttons if isinstance(btn, dict)}
    for btn in DEFAULT_PANEL_BUTTONS_LEFT:
        if btn["action"] not in existing_left:
            left_buttons.append(dict(btn))

    right_buttons = cfg.get("panel_buttons_right")
    if not isinstance(right_buttons, list):
        right_buttons = []
        cfg["panel_buttons_right"] = right_buttons
    existing_right = {btn.get("action") for btn in right_buttons if isinstance(btn, dict)}
    for btn in DEFAULT_PANEL_BUTTONS_RIGHT:
        if btn["action"] not in existing_right:
            right_buttons.append(dict(btn))

    return cfg


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_in = x.shape[0]
    dur = n_in / float(sr_in)
    n_out = int(dur * sr_out)
    if n_out <= 0:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0, dur, num=n_in, endpoint=False)
    t_out = np.linspace(0, dur, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def pick_input_device(cfg: dict) -> int | None:
    idx = cfg.get("mic_device_index", None)
    name_contains = cfg.get("mic_device_name_contains", None)

    if idx is not None:
        return idx

    if name_contains:
        needle = str(name_contains).lower()
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0 and needle in d["name"].lower():
                    return i
        except Exception:
            pass

    return None


def load_whisper_model(cfg: dict) -> tuple[object | None, str | None]:
    try:
        from faster_whisper import WhisperModel as LocalWhisperModel
    except Exception as exc:
        return None, str(exc)
    try:
        model = LocalWhisperModel(cfg.get("model", "small"), device="cpu", compute_type="int8")
    except Exception as exc:
        return None, str(exc)
    return model, None


class TretaPanel(tk.Tk):
    def __init__(self, debug: bool = False):
        super().__init__()
        _ensure_dirs()

        self.cfg = load_config()
        self.debug = debug
        self._last_vad_seconds: float | None = None
        self.locale = (self.cfg.get("language") or "es").strip().lower()
        self.history_path = HISTORY_PATH
        self.history_keep_days = int(self.cfg.get("history_keep_days", 90))
        self.history_max_entries = int(self.cfg.get("history_max_entries", 10000))
        self._history_write_count = 0

        self.title("TRETA — Panel de Control")
        self.geometry("1100x650")
        self.minsize(980, 600)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.configure(bg=MECH_BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=MECH_BG, foreground=MECH_TEXT)
        style.configure("TFrame", background=MECH_BG)
        style.configure("Panel.TFrame", background=MECH_PANEL)
        style.configure("Card.TFrame", background=MECH_CARD)
        style.configure("TLabel", background=MECH_BG, foreground=MECH_TEXT, padding=(4, 2))
        style.configure("Card.TLabel", background=MECH_CARD, foreground=MECH_TEXT, padding=(4, 2))
        style.configure(
            "Header.TLabel",
            background=MECH_PANEL,
            foreground=MECH_TEXT,
            font=("Segoe UI", 12, "bold"),
            padding=(4, 2),
        )
        style.configure(
            "TButton",
            padding=(10, 6),
            background=MECH_PANEL,
            foreground=MECH_TEXT,
            bordercolor=MECH_BORDER,
        )
        style.map(
            "TButton",
            background=[("active", MECH_BTN_HOVER), ("pressed", MECH_RED_DARK)],
            foreground=[("disabled", MECH_MUTED)],
        )
        style.configure(
            "Success.TButton",
            background=MECH_OK,
            foreground=MECH_BG,
            bordercolor=MECH_OK,
        )
        style.configure(
            "Danger.TButton",
            background=MECH_RED,
            foreground=MECH_TEXT,
            bordercolor=MECH_RED,
        )
        style.configure(
            "TProgressbar",
            background=MECH_RED,
            troughcolor=MECH_CARD_2,
            bordercolor=MECH_BORDER,
            lightcolor=MECH_RED,
            darkcolor=MECH_RED,
        )

        # IA (OpenAI)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        # Whisper local (lazy init)
        self.whisper = None
        self._whisper_error = None
        self._whisper_loaded = False
        self._whisper_disabled = sys.version_info < (3, 10)
        # Whisper local
        self.whisper, self._whisper_error = load_whisper_model(self.cfg)
        if WhisperModel:
            self.whisper = WhisperModel(self.cfg.get("model", "small"), device="cpu", compute_type="int8")
        else:
            self.whisper = None

        # Estado vivo
        self.state = read_json(STATE_PATH, {
            "mood": "calm",
            "last_interaction": now_iso(),
            "daily_interactions": 0,
            "mode": "normal"
        })
        self._save_state()

        # Audio / voz
        self.listening = False
        self.stop_event = threading.Event()
        self.speaking = threading.Event()
        self.mic_index = pick_input_device(self.cfg)
        self._last_audio_sr: int | None = None
        self._resolved_sr: int | None = None
        self._resolved_ch: int = 1

        # Captura de idea
        self._idea_capture_next = False

        # TTS
        self.tts_q: "queue.Queue[str | None]" = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        self._build_ui()
        self._presence_startup()

        self.after(700, self._poll_pending_confirm)
        self.after(8000, self._alive_tick)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Copy ----------
    def _copy(self, key: str, **kwargs) -> str:
        copy = {
            "confirm_prompt": {
                "es": "¿Confirmas la acción? Sí/No.",
                "en": "Confirm the action? Yes/No.",
            },
            "confirm_yes": {
                "es": "Hecho.",
                "en": "Done.",
            },
            "confirm_no": {
                "es": "Cancelado. Sin cambios.",
                "en": "Canceled. No changes made.",
            },
            "confirm_timeout": {
                "es": "Sin respuesta. Cancelado por seguridad.",
                "en": "No response. Canceled for safety.",
            },
            "confirm_followup": {
                "es": "Necesito un Sí o No.",
                "en": "Please answer Yes or No.",
            },
            "mode_not_found": {
                "es": "Error técnico: modo no encontrado. Prueba: {suggestions}.",
                "en": "Technical error: mode not found. Try: {suggestions}.",
            },
            "mode_activated": {
                "es": "Modo {mode} activado.",
                "en": "Mode {mode} activated.",
            },
            "stt_failed": {
                "es": "No he entendido eso. Inténtalo de nuevo.",
                "en": "I didn't catch that. Please try again.",
            },
            "insult_boundary": {
                "es": "No te ayudo si insultas. Reformúlalo.",
                "en": "I can't help if you insult me. Please rephrase.",
            },
        }
        lang = "en" if self.locale.startswith("en") else "es"
        template = copy.get(key, {}).get(lang, "")
        return template.format(**kwargs).strip()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10, style="Card.TFrame")
        root.pack(fill="both", expand=True)

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=0)
        root.rowconfigure(1, weight=1)

        self.left = ttk.Frame(root, style="Panel.TFrame")
        self.left.grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(0, 8))
        ttk.Label(self.left, text="Acciones", style="Header.TLabel").pack(pady=(0, 8))

        self.right = ttk.Frame(root, style="Panel.TFrame")
        self.right.grid(row=0, column=2, rowspan=3, sticky="nse", padx=(8, 0))
        ttk.Label(self.right, text="Sistema", style="Header.TLabel").pack(pady=(0, 8))

        top = ttk.Frame(root, style="Card.TFrame")
        top.grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 8))
        top.columnconfigure(2, weight=1)

        self.btn_listen = ttk.Button(top, text="🎙️ Escuchar", command=self.toggle_listen)
        self.btn_listen.grid(row=0, column=0, sticky="w")

        self.status = ttk.Label(top, text="Estado: en espera", style="Card.TLabel")
        self.status.grid(row=0, column=1, sticky="w", padx=10)

        self.meter = ttk.Progressbar(top, length=260, mode="determinate")
        self.meter.grid(row=0, column=3, sticky="e")

        center = ttk.Frame(root, style="Card.TFrame")
        center.grid(row=1, column=1, sticky="nsew", padx=8)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.text = scrolledtext.ScrolledText(
            center,
            wrap="word",
            font=("Segoe UI", 11),
            bg="#121b21",
            fg="#d6dde3",
            insertbackground="#d6dde3",
            selectbackground="#1b2831",
            highlightbackground="#2b3842",
            highlightcolor="#2b3842",
        )
        self.text.grid(row=0, column=0, sticky="nsew")

        self.confirm_frame = ttk.Frame(root, padding=10, style="Card.TFrame")
        self.confirm_frame.grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.confirm_frame.columnconfigure(0, weight=1)

        self.confirm_label = ttk.Label(self.confirm_frame, text="", style="Card.TLabel")
        self.confirm_label.grid(row=0, column=0, sticky="w")

        self.btn_confirm_yes = ttk.Button(
            self.confirm_frame,
            text="✅ Ejecutar",
            style="Success.TButton",
            command=self._confirm_yes,
        )
        self.btn_confirm_no = ttk.Button(
            self.confirm_frame,
            text="❌ Cancelar",
            style="Danger.TButton",
            command=self._confirm_no,
        )
        self.btn_confirm_yes.grid(row=0, column=1, padx=6)
        self.btn_confirm_no.grid(row=0, column=2)

        self._hide_confirm()

        bottom = ttk.Frame(root, style="Card.TFrame")
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(
            bottom,
            text="Treta Panel: voz + botones. Acciones peligrosas requieren confirmación.",
            style="Card.TLabel",
        ).pack(side="left")

        self._build_side_buttons()

        self._log("Treta Panel listo.\n")
        if self.whisper is None:
            detail = f" Detalle: {self._whisper_error}" if getattr(self, "_whisper_error", None) else ""
            self._log(
                "⚠ Falta dependencia de STT local (faster-whisper/pyav). El STT local no está disponible."
                f"{detail}\n"
            )
            self._log("⚠ Falta dependencia faster-whisper/pyav. El STT local no está disponible.\n")
        if not self.client:
            if OpenAI is None:
                self._log("⚠ Falta dependencia OpenAI. La voz y acciones pueden funcionar igual.\n")
            else:
                self._log("⚠ Falta OPENAI_API_KEY (solo afecta a respuestas IA). La voz y acciones pueden funcionar igual.\n")
        mic_desc = "default del sistema"
        if self.mic_index is not None:
            mic_desc = str(self.mic_index)
            try:
                mic_dev = sd.query_devices(self.mic_index)
                mic_desc = (
                    f"{self.mic_index} ({mic_dev.get('name', 'desconocido')}, "
                    f"default_sr={mic_dev.get('default_samplerate', 'n/a')})"
                )
            except Exception:
                pass
        self._resolve_audio_config()
        self._log(f"🎤 Mic: {mic_desc}\n")
        self._log(
            f"🎚️ sample_rate: {self._resolved_sr or 'auto'} | channels: {self._resolved_ch}\n"
        )
        self._log(f"🔊 input_gain_db: {self.cfg.get('input_gain_db', 0.0)}\n")
        self._log(f"🧠 stt_initial_prompt: {self.cfg.get('stt_initial_prompt', '')}\n")
        self._log(f"📈 min_input_rms: {self.cfg.get('min_input_rms', 0.0)}\n")
        self._log(f"📈 max_input_rms: {self.cfg.get('max_input_rms', 1.0)}\n")
        self._log(
            "🎯 thresholds: "
            f"noise_calib_sec={self.cfg.get('noise_calib_sec', 0.6)}, "
            f"end_silence_sec={self.cfg.get('end_silence_sec', 1.2)}, "
            f"thresh_mult={self.cfg.get('thresh_mult', 2.8)}\n"
        )
        if self.debug:
            self._log("🐛 Debug mode activo (logs/treta.log, data/debug_last.wav).\n")
        self._log("\n")

    def _build_side_buttons(self):
        # limpia
        for w in self.left.winfo_children()[1:]:
            w.destroy()
        for w in self.right.winfo_children()[1:]:
            w.destroy()

        for b in self.cfg.get("panel_buttons_left", []):
            ttk.Button(self.left, text=b["label"], command=lambda bb=b: self._button_action(bb)).pack(fill="x", pady=4)

        for b in self.cfg.get("panel_buttons_right", []):
            ttk.Button(self.right, text=b["label"], command=lambda bb=b: self._button_action(bb)).pack(fill="x", pady=4)

    def _resolve_audio_config(self):
        sr_cfg = parse_int(self.cfg.get("sample_rate", 48000))
        ch_cfg = parse_int(self.cfg.get("channels", 1))
        resolved_sr = sr_cfg
        resolved_ch = ch_cfg or 1
        try:
            if self.mic_index is not None:
                dev = sd.query_devices(self.mic_index)
                if resolved_sr is None:
                    resolved_sr = int(dev.get("default_samplerate", 48000))
                if ch_cfg is None:
                    resolved_ch = 1 if dev.get("max_input_channels", 1) >= 1 else 1
        except Exception:
            if resolved_sr is None:
                resolved_sr = 48000
        self._resolved_sr = resolved_sr or 48000
        self._resolved_ch = resolved_ch or 1

    # ---------- Logging ----------
    def _log(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            pass

    def _set_status(self, s: str):
        self.status.config(text=f"Estado: {s}")

    def _log_timing(self, stage: str, seconds: float):
        self._log(f"⏱️ {stage}: {seconds:.2f}s\n")

    def _write_debug_wav(self, audio: np.ndarray, sr: int):
        if not self.debug:
            return
        if audio is None or audio.size == 0:
            return
        path = os.path.join(DATA_DIR, "debug_last.wav")
        a = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm16 = (a * 32767.0).astype(np.int16)
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())
        except Exception as e:
            self._log(f"⚠ No pude guardar debug_last.wav: {e}\n")

    # ---------- History ----------
    def _append_history(self, event: dict) -> None:
        append_jsonl(self.history_path, event)
        self._history_write_count += 1
        if self._history_write_count % 50 == 0:
            self._prune_history()

    def _prune_history(self) -> None:
        if not os.path.exists(self.history_path):
            return
        cutoff = datetime.now() - timedelta(days=self.history_keep_days)
        kept: list[dict] = []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = obj.get("ts")
                    if ts:
                        try:
                            if datetime.fromisoformat(ts) < cutoff:
                                continue
                        except Exception:
                            pass
                    kept.append(obj)
        except Exception:
            return
        if len(kept) > self.history_max_entries:
            kept = kept[-self.history_max_entries :]
        tmp = self.history_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for obj in kept:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp, self.history_path)

    def _log_turn_history(
        self,
        user_text: str,
        assistant_text: str,
        latencies: dict,
        error_code: str | None = None,
    ) -> None:
        event = {
            "ts": now_iso(),
            "event": "turn",
            "mode": self.state.get("mode", "normal"),
            "user_text": user_text,
            "assistant_text": assistant_text,
            "latency_ms": latencies,
            "error_code": error_code,
        }
        self._append_history(event)

    def _log_event(self, event: str, payload: dict) -> None:
        self._append_history(
            {
                "ts": now_iso(),
                "event": event,
                "mode": self.state.get("mode", "normal"),
                **payload,
            }
        )

    # ---------- TTS ----------
    def _tts_worker(self):
        engine = pyttsx3.init(driverName="sapi5")
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        while True:
            msg = self.tts_q.get()
            if msg is None:
                break
            try:
                self.speaking.set()
                t0 = time.perf_counter()
                engine.say(msg)
                engine.runAndWait()
                self._log_timing("TTS", time.perf_counter() - t0)
            except Exception:
                pass
            finally:
                time.sleep(0.35)
                self.speaking.clear()

    def speak(self, text: str):
        if not text:
            return
        if self._discreet_mode():
            return
        s = text.strip()
        while s:
            self.tts_q.put(s[:260])
            s = s[260:]

    def _discreet_mode(self) -> bool:
        return bool(self.cfg.get("discreet_mode", False))

    def _ensure_whisper(self) -> None:
        if self._whisper_loaded:
            return
        self._whisper_loaded = True
        if self._whisper_disabled:
            self._log(
                f"⚠ Python {sys.version.split()[0]} detectado. "
                "El STT local requiere Python 3.10+.\n"
            )
            return
        self.whisper, self._whisper_error = load_whisper_model(self.cfg)
        if self.whisper is None:
            detail = f" Detalle: {self._whisper_error}" if self._whisper_error else ""
            self._log(
                "⚠ Falta dependencia de STT local (faster-whisper/pyav). El STT local no está disponible."
                f"{detail}\n"
            )

    # ---------- Estado vivo ----------
    def _save_state(self):
        write_json(STATE_PATH, self.state)

    def _touch_interaction(self):
        self.state["last_interaction"] = now_iso()
        self.state["daily_interactions"] = int(self.state.get("daily_interactions", 0)) + 1
        self._save_state()

    def _presence_startup(self):
        self._log("🟦 Sistema Treta operativo.\n")
        self._diary("presence", "startup")
        self._run_action_id("presence_start", allow_confirm=False, source="system")

    def _alive_tick(self):
        try:
            last = datetime.fromisoformat(self.state.get("last_interaction", now_iso()))
        except Exception:
            last = datetime.now()

        idle = datetime.now() - last
        hour = datetime.now().hour

        # mood por hora
        if 0 <= hour <= 6:
            self.state["mood"] = "night"
        elif 7 <= hour <= 12:
            self.state["mood"] = "morning"
        elif 13 <= hour <= 19:
            self.state["mood"] = "day"
        else:
            self.state["mood"] = "evening"

        # nudges suaves
        if idle > timedelta(hours=2) and 10 <= hour <= 23:
            self._log("🫧 Marian, llevas rato sin hablar conmigo. ¿Agua / pausa? (yo solo digo 👀)\n")
            self._diary("nudge", "break_hint")
            self._touch_interaction()

        self._save_state()
        self.after(8000, self._alive_tick)

    # ---------- Diario / Ideas ----------
    def _diary(self, typ: str, value: str):
        append_jsonl(DIARY_PATH, {"ts": now_iso(), "type": typ, "value": value})

    def _add_idea(self, text: str, tag: str = "random"):
        append_jsonl(IDEAS_PATH, {"ts": now_iso(), "text": text, "tag": tag})
        self._diary("idea", tag)

    # ---------- Acciones ----------
    def _button_action(self, btn_cfg: dict):
        action_id = btn_cfg.get("action")
        needs_confirm = bool(btn_cfg.get("needs_confirm", False))

        if action_id == "idea_prompt":
            self._log("🧠 Vale. Dime la idea (y la guardo).\n")
            self.speak("Dime la idea y la guardo.")
            self._diary("mode", "idea_capture")
            self._idea_capture_next = True
            return

        if needs_confirm:
            self._request_confirm(action_id, self._copy("confirm_prompt"))
            return

        self._run_action_id(action_id, allow_confirm=False, source="ui")

    def _mode_label_from_action(self, action_id: str) -> str:
        return action_id.replace("mode_", "").replace("_", " ").strip()


    def _mode_label_from_action(self, action_id: str) -> str:
        return action_id.replace("mode_", "").replace("_", " ").strip()


    def _mode_label_from_action(self, action_id: str) -> str:
        return action_id.replace("mode_", "").replace("_", " ").strip()

    def _run_action_id(self, action_id: str, allow_confirm: bool = True, source: str = "ui"):
        actions = self.cfg.get("actions", {})
        spec = actions.get(action_id)
        if not spec:
            self._log(f"⚠ Acción no definida: {action_id}\n")
            return

        if action_id == "mode_profile":
            from modes.profile import open_profile_mode

            open_profile_mode(self)
            return

        self._diary("action", action_id)

        typ = spec.get("type", "ps1")
        rel = spec.get("path", "")
        path = os.path.join(BASE_DIR, rel)

        if typ == "ps1":
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", path]
        elif typ == "bat":
            cmd = [path]
        else:
            self._log(f"⚠ Tipo no soportado: {typ}\n")
            return

        try:
            subprocess.Popen(cmd, cwd=BASE_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
            self._log(f"▶ Ejecutado: {action_id}\n")
        except Exception as e:
            self._log(f"❌ Error ejecutando {action_id}: {e}\n")
            return

        if action_id.startswith("mode_"):
            mode_name = self._mode_label_from_action(action_id)
            self.state["mode"] = mode_name
            self._save_state()
            self._log_event(
                "mode_activated",
                {"mode_name": mode_name, "source": source},
            )

    # ---------- Confirmaciones ----------
    def _request_confirm(self, action_id: str, text: str | None):
        prompt = text or self._copy("confirm_prompt")
        expires_at = (datetime.now() + timedelta(seconds=CONFIRM_TIMEOUT_SEC)).isoformat(timespec="seconds")
        write_json(
            PENDING_CONFIRM_PATH,
            {
                "ts": now_iso(),
                "action": action_id,
                "text": prompt,
                "expires_at": expires_at,
                "attempts": 0,
            },
        )
        self._show_confirm(prompt)
        self._log(f"⚠ Confirmación requerida: {prompt}\n")
        self.speak(prompt)
        self._log_event(
            "confirmation_prompted",
            {"action": action_id, "text": prompt},
        )

    def _poll_pending_confirm(self):
        if self._check_confirm_timeout():
            self.after(700, self._poll_pending_confirm)
            return
        if os.path.exists(PENDING_CONFIRM_PATH):
            obj = read_json(PENDING_CONFIRM_PATH, None)
            if obj and obj.get("text"):
                self._show_confirm(obj["text"])
        else:
            self._hide_confirm()
        self.after(700, self._poll_pending_confirm)

    def _check_confirm_timeout(self) -> bool:
        if not os.path.exists(PENDING_CONFIRM_PATH):
            return False
        obj = read_json(PENDING_CONFIRM_PATH, None)
        if not obj:
            return False
        expires_at = obj.get("expires_at")
        if not expires_at:
            return False
        try:
            expired = datetime.now() > datetime.fromisoformat(expires_at)
        except Exception:
            expired = False
        if not expired:
            return False
        try:
            os.remove(PENDING_CONFIRM_PATH)
        except Exception:
            pass
        self._hide_confirm()
        msg = self._copy("confirm_timeout")
        self._log(f"⏱️ Confirmación expirada. {msg}\n")
        self.speak(msg)
        self._log_event(
            "confirmation_resolved",
            {"action": obj.get("action"), "result": "timeout"},
        )
        return True

    def _parse_confirmation_response(self, text: str) -> str | None:
        t = normalize_text(text)
        yes_terms = {"si", "sí", "vale", "ok", "okay", "yes"}
        no_terms = {"no", "cancelar", "cancela", "cancelado", "never mind", "stop"}
        if any(term in t.split() or term in t for term in yes_terms):
            return "yes"
        if any(term in t.split() or term in t for term in no_terms):
            return "no"
        return None

    def _handle_confirmation_response(self, text: str) -> tuple[bool, str, str | None]:
        if not os.path.exists(PENDING_CONFIRM_PATH):
            return False, "", None
        obj = read_json(PENDING_CONFIRM_PATH, None)
        if not obj:
            return False, "", None
        action_id = obj.get("action")
        response = self._parse_confirmation_response(text)
        if response == "yes":
            try:
                os.remove(PENDING_CONFIRM_PATH)
            except Exception:
                pass
            self._hide_confirm()
            msg = self._copy("confirm_yes")
            self._log(f"✅ Confirmado.\n")
            self.speak(msg)
            if action_id:
                self._run_action_id(action_id, allow_confirm=False, source="voice")
            self._log_event(
                "confirmation_resolved",
                {"action": action_id, "result": "yes"},
            )
            return True, msg, None
        if response == "no":
            try:
                os.remove(PENDING_CONFIRM_PATH)
            except Exception:
                pass
            self._hide_confirm()
            msg = self._copy("confirm_no")
            self._log("❌ Cancelado.\n")
            self.speak(msg)
            self._diary("confirm", "cancel")
            self._log_event(
                "confirmation_resolved",
                {"action": action_id, "result": "no"},
            )
            return True, msg, None
        attempts = int(obj.get("attempts", 0))
        if attempts < 1:
            obj["attempts"] = attempts + 1
            write_json(PENDING_CONFIRM_PATH, obj)
            msg = self._copy("confirm_followup")
            self._log(f"⚠ Confirmación: {msg}\n")
            self.speak(msg)
            return True, msg, None
        try:
            os.remove(PENDING_CONFIRM_PATH)
        except Exception:
            pass
        self._hide_confirm()
        msg = self._copy("confirm_timeout")
        self._log("❌ Cancelado por seguridad.\n")
        self.speak(msg)
        self._log_event(
            "confirmation_resolved",
            {"action": action_id, "result": "cancel"},
        )
        return True, msg, "CONFIRM_TIMEOUT"

    def _show_confirm(self, text: str):
        self.confirm_label.config(text=text)
        self.confirm_frame.grid()
        self.btn_confirm_yes.state(["!disabled"])
        self.btn_confirm_no.state(["!disabled"])

    def _hide_confirm(self):
        self.confirm_label.config(text="")
        self.confirm_frame.grid_remove()

    def _confirm_yes(self):
        obj = read_json(PENDING_CONFIRM_PATH, None)
        if not obj:
            self._hide_confirm()
            return
        action_id = obj.get("action")
        try:
            os.remove(PENDING_CONFIRM_PATH)
        except Exception:
            pass
        self._hide_confirm()
        msg = self._copy("confirm_yes")
        self._log("✅ Confirmado.\n")
        self.speak(msg)
        if action_id:
            self._run_action_id(action_id, allow_confirm=False, source="ui")
        self._log_event(
            "confirmation_resolved",
            {"action": action_id, "result": "yes"},
        )

    def _confirm_no(self):
        try:
            os.remove(PENDING_CONFIRM_PATH)
        except Exception:
            pass
        self._hide_confirm()
        msg = self._copy("confirm_no")
        self._log("❌ Cancelado.\n")
        self.speak(msg)
        self._diary("confirm", "cancel")
        self._log_event(
            "confirmation_resolved",
            {"action": None, "result": "no"},
        )

    # ---------- Voz ----------
    def toggle_listen(self):
        if self.listening:
            self.stop_event.set()
            self.btn_listen.config(text="🎙️ Escuchar")
            self._set_status("parando…")
            return

        if self.speaking.is_set():
            self._log("⏳ Estoy hablando; espera un segundo.\n")
            return

        self.listening = True
        self.stop_event.clear()
        self.btn_listen.config(text="⏹️ Parar")
        self._set_status("escuchando…")
        threading.Thread(target=self._listen_flow, daemon=True).start()

    def _listen_flow(self):
        try:
            total_t0 = time.perf_counter()
            capture_t0 = time.perf_counter()
            audio = self._record_until_silence()
            capture_sec = time.perf_counter() - capture_t0
            self._log_timing("capture", capture_sec)
            if self._last_vad_seconds is not None:
                self._log_timing("VAD", self._last_vad_seconds)
            self.after(0, lambda: self.meter.configure(value=0))

            sr = int(self._last_audio_sr or self._resolved_sr or 48000)
            self._write_debug_wav(audio, sr)
            if audio is None or len(audio) < int(sr * 0.3):
                self._finish_listen()
                return
            level = rms(audio)
            min_rms = parse_float(self.cfg.get("min_input_rms", 0.0), 0.0) or 0.0
            max_rms = parse_float(self.cfg.get("max_input_rms", 1.0), 1.0) or 1.0
            if level < min_rms:
                self._log("⚠️ Audio muy bajo. Revisa el micro o sube input_gain_db.\n")
                self.speak("No te oigo bien. Sube el volumen del micro o acércate.")
                self._finish_listen()
                return
            if level > max_rms:
                self._log("⚠️ Audio saturado. Baja el volumen del micro.\n")

            self._set_status("transcribiendo…")
            stt_t0 = time.perf_counter()
            text = self._transcribe(audio)
            stt_sec = time.perf_counter() - stt_t0
            self._log_timing("STT", stt_sec)
            if not text:
                msg = self._copy("stt_failed")
                self._log("📝 Oído: (nada claro)\n\n")
                self.speak(msg)
                self._log_turn_history(
                    "",
                    msg,
                    {
                        "capture_ms": int(capture_sec * 1000),
                        "stt_ms": int(stt_sec * 1000),
                        "total_ms": int((time.perf_counter() - total_t0) * 1000),
                    },
                    error_code="STT_FAILED",
                )
                self._finish_listen()
                return

            self._touch_interaction()
            self._log(f"📝 Oído: {text}\n")

            if self._idea_capture_next:
                self._idea_capture_next = False
                self._add_idea(text)
                self._log("📌 Idea guardada.\n\n")
                msg = "Idea guardada."
                self.speak(msg)
                self._log_turn_history(
                    text,
                    msg,
                    {
                        "capture_ms": int(capture_sec * 1000),
                        "stt_ms": int(stt_sec * 1000),
                        "total_ms": int((time.perf_counter() - total_t0) * 1000),
                    },
                )
                self._finish_listen()
                return

            handled, response_text, error_code = self._maybe_handle_intent(text)
            if handled:
                self._log("\n")
                self._log_turn_history(
                    text,
                    response_text,
                    {
                        "capture_ms": int(capture_sec * 1000),
                        "stt_ms": int(stt_sec * 1000),
                        "total_ms": int((time.perf_counter() - total_t0) * 1000),
                    },
                    error_code=error_code,
                )
                self._finish_listen()
                return

            # Respuesta IA (si hay API key)
            if self.client:
                self._set_status("pensando…")
                think_t0 = time.perf_counter()
                answer = self._ask_chatgpt(text)
                think_sec = time.perf_counter() - think_t0
                self._log(f"🤖 Treta: {answer}\n\n")
                self._set_status("hablando…")
                self.speak(answer)
                self._log_turn_history(
                    text,
                    answer,
                    {
                        "capture_ms": int(capture_sec * 1000),
                        "stt_ms": int(stt_sec * 1000),
                        "think_ms": int(think_sec * 1000),
                        "total_ms": int((time.perf_counter() - total_t0) * 1000),
                    },
                )
            else:
                msg = "Comando registrado. Puedo ejecutar acciones y guardar ideas."
                self._log(f"🤖 Treta: (sin API Key) {msg}\n\n")
                self._log_turn_history(
                    text,
                    msg,
                    {
                        "capture_ms": int(capture_sec * 1000),
                        "stt_ms": int(stt_sec * 1000),
                        "total_ms": int((time.perf_counter() - total_t0) * 1000),
                    },
                )

        except Exception as e:
            self._log(f"❌ Error: {e}\n\n")
        finally:
            self._finish_listen()

    def _finish_listen(self):
        self.listening = False
        self.stop_event.clear()
        self.btn_listen.config(text="🎙️ Escuchar")
        self._set_status("en espera")

    def _available_modes(self) -> dict[str, dict]:
        actions = {**DEFAULT_ACTIONS, **self.cfg.get("actions", {})}
        modes: dict[str, dict] = {}
        for action_id in actions.keys():
            if not action_id.startswith("mode_"):
                continue
            label = self._mode_label_from_action(action_id)
            modes[normalize_text(label)] = {"action_id": action_id, "label": label}
        return modes

    def _handle_mode_command(self, raw_text: str) -> tuple[bool, str, str | None]:
        raw_norm = normalize_text(raw_text).strip()
        match = re.match(r"^treta[, ]+ejecuta modo\s+(.+)$", raw_norm)
        if not match:
            return False, "", None
        requested = match.group(1).strip(" .,!¡¿?")
        requested_norm = normalize_text(requested)
        modes = self._available_modes()
        if requested_norm in modes:
            action_id = modes[requested_norm]["action_id"]
            label = modes[requested_norm]["label"]
            response = self._copy("mode_activated", mode=label)
            self._log(f"🎛️ Modo solicitado: {label}\n")
            self.speak(response)
            self._run_action_id(action_id, allow_confirm=False, source="voice")
            return True, response, None
        suggestions = difflib.get_close_matches(requested_norm, list(modes.keys()), n=3, cutoff=0.5)
        suggestion_labels = [modes[key]["label"] for key in suggestions]
        if suggestion_labels:
            response = self._copy("mode_not_found", suggestions=", ".join(suggestion_labels))
        else:
            response = self._copy("mode_not_found", suggestions="—")
        self._log(f"⚠ Modo no encontrado: {requested}\n")
        self.speak(response)
        self._log_event(
            "mode_not_found",
            {"requested_name": requested, "suggestions": suggestion_labels},
        )
        return True, response, "MODE_NOT_FOUND"

    def _is_direct_insult(self, text: str) -> bool:
        t = normalize_text(text)
        insults = {
            "idiota",
            "imbecil",
            "estupido",
            "estupida",
            "tonto",
            "tonta",
            "gilipollas",
            "mierda",
            "puta",
            "puto",
            "fuck",
            "shit",
            "asshole",
        }
        if not any(word in t for word in insults):
            return False
        return "treta" in t or "eres" in t or "tu" in t

    def _maybe_handle_intent(self, text: str) -> tuple[bool, str, str | None]:
        raw = text.strip()
        raw_norm = normalize_text(raw)

        self._check_confirm_timeout()

        handled_confirm, confirm_msg, confirm_error = self._handle_confirmation_response(raw_norm)
        if handled_confirm:
            return True, confirm_msg, confirm_error

        if self._is_direct_insult(raw_norm):
            msg = self._copy("insult_boundary")
            self._log(f"🚫 {msg}\n")
            self.speak(msg)
            return True, msg, "INSULT_BLOCK"

        handled_mode, mode_msg, mode_error = self._handle_mode_command(raw)
        if handled_mode:
            return True, mode_msg, mode_error

        t = raw.lower().strip()
        wake = (self.cfg.get("wake_word") or "").lower().strip()
        if wake and wake in t:
            t = t.replace(wake, "").strip(" ,.:;!?¡¿")
        t_norm = normalize_text(t)

        matched, response = self._handle_local_intent(t_norm)
        if matched:
            return True, response, None

        if is_time_request(t_norm):
            now = datetime.now().strftime("%H:%M")
            answer = f"Son las {now}."
            self._log(f"🕒 Hora local: {now}\n")
            self.speak(answer)
            return True, answer, None

        for rule in self.cfg.get("voice_commands", []):
            m = (rule.get("match") or "").lower()
            if m and m in t:
                action = rule.get("action")
                needs_confirm = bool(rule.get("needs_confirm", False))

                self._log(f"🎛️ Comando detectado: {m} → {action}\n")
                self._diary("voice_command", action or m)

                if action == "idea_prompt":
                    msg = "Dime la idea y la guardo."
                    self._log("🧠 Dime la idea (la guardo).\n")
                    self.speak(msg)
                    self._idea_capture_next = True
                    return True, msg, None

                if needs_confirm:
                    prompt = self._copy("confirm_prompt")
                    self._request_confirm(action, prompt)
                    return True, prompt, None

                if action:
                    self._run_action_id(action, allow_confirm=False, source="voice")
                    return True, "", None

        return False, "", None

    def _log_intent(self, payload: dict):
        payload = {"ts": now_iso(), **payload}
        append_jsonl(INTENTS_PATH, payload)

    def _handle_local_intent(self, text: str) -> tuple[bool, str]:
        intents = load_local_intents(self.cfg)
        threshold = float(self.cfg.get("fuzzy_match_threshold", 0.84))
        for intent in intents:
            name = str(intent.get("name") or "unknown")
            patterns = intent.get("patterns", [])
            response = str(intent.get("response") or "")
            if not response or not isinstance(patterns, list):
                continue
            for pattern in patterns:
                pattern_norm = normalize_text(str(pattern))
                matched = False
                if not pattern_norm:
                    continue
                if pattern_norm.startswith("re:"):
                    expr = pattern_norm[3:]
                    if re.search(expr, text):
                        matched = True
                elif pattern_norm in text:
                    matched = True
                elif fuzzy_contains(text, pattern_norm, threshold):
                    matched = True
                if matched:
                    answer = render_local_response(response, self.cfg)
                    self._log(f"🎯 Intento local: {name} (patrón: {pattern_norm})\n")
                    self._log_intent(
                        {
                            "source": "panel",
                            "intent": name,
                            "pattern": pattern_norm,
                            "text": text,
                            "response": answer,
                        }
                    )
                    self.speak(answer)
                    return True, answer
        return False, ""

    # ---------- Audio capture ----------
    def _record_until_silence(self):
        sr = int(self._resolved_sr or 48000)
        ch = int(self._resolved_ch or 1)
        max_sec = float(self.cfg.get("max_record_sec", 90))
        noise_calib = float(self.cfg.get("noise_calib_sec", 0.6))
        end_silence = float(self.cfg.get("end_silence_sec", 1.2))
        thresh_mult = float(self.cfg.get("thresh_mult", 2.8))
        gain_db = float(self.cfg.get("input_gain_db", 0.0))

        # calibración
        self._set_status("calibrando ruido…")
        noise = self._record_fixed(noise_calib, sr, ch)
        if noise is None:
            return None

        noise_level = rms(noise)
        thresh = max(noise_level * thresh_mult, 0.006)

        chunks = []
        silence_run = 0.0
        t0 = time.time()

        def callback(indata, frames, time_info, status):
            chunks.append(indata.copy())

        with sd.InputStream(
            samplerate=sr,
            channels=ch,
            dtype="float32",
            device=self.mic_index,
            callback=callback,
        ):
            vad_t0 = time.perf_counter()
            last_meter = 0.0
            while True:
                if self.stop_event.is_set():
                    break
                if self.speaking.is_set():
                    break
                if (time.time() - t0) > max_sec:
                    break

                time.sleep(0.05)

                if chunks:
                    x = chunks[-1].squeeze()
                    level = rms(x)

                    nowt = time.time()
                    if nowt - last_meter > 0.12:
                        last_meter = nowt
                        v = min(100, level * 1800)
                        self.after(0, lambda vv=v: self.meter.configure(value=vv))

                    if level < thresh:
                        silence_run += 0.05
                    else:
                        silence_run = 0.0

                    if silence_run >= end_silence and (time.time() - t0) > 0.8:
                        break
            self._last_vad_seconds = time.perf_counter() - vad_t0

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).astype(np.float32)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio[:, 0]
        self._last_audio_sr = sr
        audio = apply_gain(audio.squeeze(), gain_db)
        return audio

    def _record_fixed(self, seconds: float, sr: int, ch: int):
        frames = int(sr * seconds)
        try:
            data = sd.rec(frames, samplerate=sr, channels=ch, dtype="float32", device=self.mic_index)
            sd.wait()
            x = data.astype(np.float32)
            if x.ndim == 2 and x.shape[1] > 1:
                x = x[:, 0]
            return apply_gain(x.squeeze(), float(self.cfg.get("input_gain_db", 0.0)))
        except Exception as e:
            self._log(f"❌ Error grabando audio: {e}\n")
            return None

    def _transcribe(self, audio: np.ndarray) -> str:
        self._ensure_whisper()
        if self.whisper is None:
            return ""
        sr_in = int(self._last_audio_sr or self._resolved_sr or 48000)
        audio_16k = resample_linear(audio, sr_in, 16000)
        prompt = (self.cfg.get("stt_initial_prompt") or "").strip()
        kwargs = dict(language=self.cfg.get("language", "es"), vad_filter=True)
        if prompt:
            kwargs["initial_prompt"] = prompt
        segments, _ = self.whisper.transcribe(audio_16k, **kwargs)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _ask_chatgpt(self, user_text: str) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.cfg.get("system_prompt", "")},
                {"role": "user", "content": user_text},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    def on_close(self):
        try:
            self.tts_q.put(None)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treta Panel")
    parser.add_argument("--debug", action="store_true", help="Guarda debug_last.wav y logs detallados.")
    args = parser.parse_args()
    TretaPanel(debug=args.debug).mainloop()
