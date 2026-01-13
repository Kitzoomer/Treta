import os, json, time, threading, queue, subprocess
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

from openai import OpenAI
from faster_whisper import WhisperModel

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


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


class TretaPanel(tk.Tk):
    def __init__(self, debug: bool = False):
        super().__init__()
        _ensure_dirs()

        self.cfg = load_config()
        self.debug = debug
        self._last_vad_seconds: float | None = None

        self.title("TRETA — Panel de Control")
        self.geometry("1100x650")
        self.minsize(980, 600)

        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            style.theme_use("clam")
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(4, 2))

        # IA (OpenAI)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Whisper local
        self.whisper = WhisperModel(self.cfg.get("model", "small"), device="cpu", compute_type="int8")

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

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=0)
        root.rowconfigure(1, weight=1)

        self.left = ttk.Frame(root)
        self.left.grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(0, 8))
        ttk.Label(self.left, text="Acciones", font=("Segoe UI", 12, "bold")).pack(pady=(0, 8))

        self.right = ttk.Frame(root)
        self.right.grid(row=0, column=2, rowspan=3, sticky="nse", padx=(8, 0))
        ttk.Label(self.right, text="Sistema", font=("Segoe UI", 12, "bold")).pack(pady=(0, 8))

        top = ttk.Frame(root)
        top.grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 8))
        top.columnconfigure(2, weight=1)

        self.btn_listen = ttk.Button(top, text="🎙️ Escuchar", command=self.toggle_listen)
        self.btn_listen.grid(row=0, column=0, sticky="w")

        self.status = ttk.Label(top, text="Estado: en espera")
        self.status.grid(row=0, column=1, sticky="w", padx=10)

        self.meter = ttk.Progressbar(top, length=260, mode="determinate")
        self.meter.grid(row=0, column=3, sticky="e")

        center = ttk.Frame(root)
        center.grid(row=1, column=1, sticky="nsew", padx=8)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.text = scrolledtext.ScrolledText(center, wrap="word", font=("Segoe UI", 11))
        self.text.grid(row=0, column=0, sticky="nsew")

        self.confirm_frame = ttk.Frame(root, padding=10)
        self.confirm_frame.grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        self.confirm_frame.columnconfigure(0, weight=1)

        self.confirm_label = ttk.Label(self.confirm_frame, text="")
        self.confirm_label.grid(row=0, column=0, sticky="w")

        self.btn_confirm_yes = ttk.Button(self.confirm_frame, text="✅ Ejecutar", command=self._confirm_yes)
        self.btn_confirm_no = ttk.Button(self.confirm_frame, text="❌ Cancelar", command=self._confirm_no)
        self.btn_confirm_yes.grid(row=0, column=1, padx=6)
        self.btn_confirm_no.grid(row=0, column=2)

        self._hide_confirm()

        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(bottom, text="Treta Panel: voz + botones. Acciones peligrosas requieren confirmación.").pack(side="left")

        self._build_side_buttons()

        self._log("Treta Panel listo.\n")
        if not self.client:
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
        s = text.strip()
        while s:
            self.tts_q.put(s[:260])
            s = s[260:]

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
        self._run_action_id("presence_start", allow_confirm=False)

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
            self._request_confirm(action_id, f"¿Confirmas ejecutar: {action_id}?")
            return

        self._run_action_id(action_id, allow_confirm=False)

    def _run_action_id(self, action_id: str, allow_confirm: bool = True):
        actions = self.cfg.get("actions", {})
        spec = actions.get(action_id)
        if not spec:
            self._log(f"⚠ Acción no definida: {action_id}\n")
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

    # ---------- Confirmaciones ----------
    def _request_confirm(self, action_id: str, text: str):
        write_json(PENDING_CONFIRM_PATH, {"ts": now_iso(), "action": action_id, "text": text})
        self._show_confirm(text)
        self._log(f"⚠ Confirmación requerida: {text}\n")
        self.speak(text)

    def _poll_pending_confirm(self):
        if os.path.exists(PENDING_CONFIRM_PATH):
            obj = read_json(PENDING_CONFIRM_PATH, None)
            if obj and obj.get("text"):
                self._show_confirm(obj["text"])
        else:
            self._hide_confirm()
        self.after(700, self._poll_pending_confirm)

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
        self._log("✅ Confirmado.\n")
        self.speak("Confirmado.")
        if action_id:
            self._run_action_id(action_id, allow_confirm=False)

    def _confirm_no(self):
        try:
            os.remove(PENDING_CONFIRM_PATH)
        except Exception:
            pass
        self._hide_confirm()
        self._log("❌ Cancelado.\n")
        self.speak("Cancelado.")
        self._diary("confirm", "cancel")

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
            capture_t0 = time.perf_counter()
            audio = self._record_until_silence()
            self._log_timing("capture", time.perf_counter() - capture_t0)
            if self._last_vad_seconds is not None:
                self._log_timing("VAD", self._last_vad_seconds)
            self.after(0, lambda: self.meter.configure(value=0))

            sr = int(self.cfg.get("sample_rate", 48000))
            self._write_debug_wav(audio, sr)
            if audio is None or len(audio) < int(sr * 0.3):
                self._finish_listen()
                return

            self._set_status("transcribiendo…")
            stt_t0 = time.perf_counter()
            text = self._transcribe(audio)
            self._log_timing("STT", time.perf_counter() - stt_t0)
            if not text:
                self._log("📝 Oído: (nada claro)\n\n")
                self._finish_listen()
                return

            self._touch_interaction()
            self._log(f"📝 Oído: {text}\n")

            if self._idea_capture_next:
                self._idea_capture_next = False
                self._add_idea(text)
                self._log("📌 Idea guardada.\n\n")
                self.speak("Idea guardada.")
                self._finish_listen()
                return

            if self._maybe_handle_intent(text):
                self._log("\n")
                self._finish_listen()
                return

            # Respuesta IA (si hay API key)
            if self.client:
                self._set_status("pensando…")
                answer = self._ask_chatgpt(text)
                self._log(f"🤖 Treta: {answer}\n\n")
                self._set_status("hablando…")
                self.speak(answer)
            else:
                self._log("🤖 Treta: (sin API Key) Comando registrado. Puedo ejecutar acciones y guardar ideas.\n\n")

        except Exception as e:
            self._log(f"❌ Error: {e}\n\n")
        finally:
            self._finish_listen()

    def _finish_listen(self):
        self.listening = False
        self.stop_event.clear()
        self.btn_listen.config(text="🎙️ Escuchar")
        self._set_status("en espera")

    def _maybe_handle_intent(self, text: str) -> bool:
        t = text.lower().strip()
        wake = (self.cfg.get("wake_word") or "").lower().strip()
        if wake and wake in t:
            t = t.replace(wake, "").strip(" ,.:;!?¡¿")
        t_norm = normalize_text(t)

        if self._handle_local_intent(t_norm):
            return True

        for rule in self.cfg.get("voice_commands", []):
            m = (rule.get("match") or "").lower()
            if m and m in t:
                action = rule.get("action")
                needs_confirm = bool(rule.get("needs_confirm", False))

                self._log(f"🎛️ Comando detectado: {m} → {action}\n")
                self._diary("voice_command", action or m)

                if action == "idea_prompt":
                    self._log("🧠 Dime la idea (la guardo).\n")
                    self.speak("Dime la idea y la guardo.")
                    self._idea_capture_next = True
                    return True

                if needs_confirm:
                    self._request_confirm(action, f"¿Confirmas ejecutar {action}?")
                    return True

                if action:
                    self._run_action_id(action, allow_confirm=False)
                    return True

        return False

    def _log_intent(self, payload: dict):
        payload = {"ts": now_iso(), **payload}
        append_jsonl(INTENTS_PATH, payload)

    def _handle_local_intent(self, text: str) -> bool:
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
                    return True
        return False

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
