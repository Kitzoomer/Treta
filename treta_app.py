import os
import json
import time
import wave
import queue
import threading
import tempfile
import unicodedata
import difflib

import numpy as np
import sounddevice as sd
import pyttsx3

import customtkinter as ctk
from tkinter import messagebox

from openai import OpenAI

from modes.registry import ModePosition, get_modes, modes_for_position
from ui_theme import (
    MECH_BG,
    MECH_BORDER,
    MECH_BTN_HOVER,
    MECH_CARD,
    MECH_CARD_2,
    MECH_DANGER,
    MECH_MUTED,
    MECH_OK,
    MECH_PANEL,
    MECH_RED,
    MECH_RED_DARK,
    MECH_TEXT,
    apply_treta_theme,
)

# ===========================
# CONFIG
# ===========================
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
APP_BUILD = "wakeword-fix-2"


def load_config() -> dict:
    # ✅ Robustez Windows: soporta UTF-8 con BOM (utf-8-sig)
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)



def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


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


def resolve_audio_settings(cfg: dict, mic_index: int | None) -> tuple[int, int]:
    sr = parse_int(cfg.get("sample_rate", None))
    ch = parse_int(cfg.get("channels", None))

    if sr is None or ch is None:
        try:
            dev = sd.query_devices(mic_index) if mic_index is not None else sd.query_devices()
            if sr is None:
                sr = int(dev.get("default_samplerate", 48000))
            if ch is None:
                ch = max(1, int(dev.get("max_input_channels", 1)))
        except Exception:
            sr = sr or 48000
            ch = ch or 1

    return int(sr), max(1, int(ch))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def wake_word_matches(text: str, candidates: list[str], threshold: float) -> bool:
    if not text or not candidates:
        return False
    words = text.split()
    for candidate in candidates:
        if candidate in words:
            return True
    for candidate in candidates:
        if not words:
            break
        for word in words:
            if difflib.SequenceMatcher(None, word, candidate).ratio() >= threshold:
                return True
    return False


def pick_input_device(cfg: dict) -> int | None:
    """
    1) Si config.json tiene mic_device_index -> usa ese.
    2) Si mic_device_name_contains -> busca por nombre.
    3) Si no, intenta auto-detectar FIFINE y prioriza default_sr ~ 48000.
    4) Si no encuentra nada -> None (default del sistema).
    """
    idx = cfg.get("mic_device_index", None)
    name_contains = cfg.get("mic_device_name_contains", None)

    if idx is not None:
        return int(idx)

    try:
        devs = sd.query_devices()
    except Exception:
        return None

    # 2) nombre contiene
    if name_contains:
        needle = str(name_contains).lower()
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0 and needle in str(d.get("name", "")).lower():
                return i

    # 3) auto FIFINE (prioriza 48k)
    candidates = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = str(d.get("name", "")).lower()
        if "fifine" in name or "microphone" in name or "micrófono" in name:
            sr = float(d.get("default_samplerate", 0) or 0)
            candidates.append((abs(sr - 48000.0), -sr, i))

    if candidates:
        candidates.sort()
        return candidates[0][2]

    return None


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Convierte float32 mono [-1..1] a WAV PCM16 en memoria.
    """
    a = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm16 = (a * 32767.0).astype(np.int16)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        path = tf.name

    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


class TretaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Base CTk
        apply_treta_theme()

        self.cfg = load_config()
        self.base_dir = BASE_DIR

        self.title("TRETA — Panel de Control")
        self.geometry("1500x860")
        self.minsize(1100, 650)
        self.configure(fg_color=MECH_BG)

        # Layout 3 columnas
        self.grid_columnconfigure(0, weight=0)  # izquierda
        self.grid_columnconfigure(1, weight=1)  # centro
        self.grid_columnconfigure(2, weight=0)  # derecha
        self.grid_rowconfigure(0, weight=1)

        # OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Audio / states
        self.listening = False
        self.stop_event = threading.Event()
        self.speaking = threading.Event()
        self.mic_index = pick_input_device(self.cfg)

        # ✅ Anti-ruido / Anti-youtube: no empezamos a grabar hasta detectar voz
        self.start_voice_sec = 0.20   # necesita 200ms de voz para arrancar
        self.min_record_sec  = 1.00   # no cortar antes de 1s

        # Wake word / always-on
        self.hotword_enabled = True
        self.hotword_stop = threading.Event()
        self.hotword_busy = threading.Event()
        self.wake_word = (self.cfg.get("wake_word", "treta") or "treta").strip().lower()
        self.wake_word_variants = self.cfg.get("wake_word_variants", [])
        self.wake_word_threshold = float(self.cfg.get("wake_word_fuzzy_threshold", 0.86))
        self.wake_debug = bool(self.cfg.get("wake_debug", False))
        if not isinstance(self.wake_word_variants, list):
            self.wake_word_variants = []
        self.wake_word_variants = [
            normalize_text(str(item)).strip()
            for item in self.wake_word_variants
            if str(item).strip()
        ]
        self.wake_word_variants = [w for w in self.wake_word_variants if w and w != self.wake_word]

        # ✅ FIX CRÍTICO: ruta robusta del modelo Vosk
        cfg_path = self.cfg.get("vosk_model_path", os.path.join("models", "vosk-es"))
        cfg_path = str(cfg_path).strip()
        if os.path.isabs(cfg_path):
            self.vosk_model_path = cfg_path
        else:
            self.vosk_model_path = os.path.normpath(os.path.join(BASE_DIR, cfg_path))

        self.wake_cooldown_sec = float(self.cfg.get("wake_cooldown_sec", 2.0))
        self.wake_sample_rate = int(self.cfg.get("wake_sample_rate", 16000))

        # Magic process
        self.magic_proc = None

        # TTS en hilo (con autorecovery)
        self.tts_q: "queue.Queue[str | None]" = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # Modes
        self.modes = list(get_modes())

        # UI
        self._build_left()
        self._build_center()
        self._build_right()

        # Logs iniciales
        self._log("Treta Panel listo.\n")
        self._log(f"Build: {APP_BUILD}\n")
        self._log(f"Archivo: {__file__}\n")
        if not api_key:
            self._log("⚠ Falta OPENAI_API_KEY. (Sin esto no hay voz/IA)\n")
        else:
            self._log("✅ OPENAI_API_KEY detectada.\n")
        self._log(f"🎤 Mic: {self.mic_index if self.mic_index is not None else 'default del sistema'}\n")
        wake_words = ", ".join([self.wake_word.upper(), *[w.upper() for w in self.wake_word_variants]])
        self._log(f"🟠 Wake-word: '{wake_words}' (always-on)\n")
        self._log(f"🟠 Vosk model path: {self.vosk_model_path}\n\n")
        self._log("🧠 Sistema Treta operativo.\n")
        self._log("▶ Ejecutado: presence_start\n")

        # Arrancar wake-word en segundo plano
        self.hotword_thread = threading.Thread(target=self._hotword_worker, daemon=True)
        self.hotword_thread.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------------------------
    # UI helpers
    # --------------------------
    def _big_btn(self, parent, text, command, fg=None, hover=None):
        return ctk.CTkButton(
            parent,
            text=text,
            height=62,                 # grande táctil
            corner_radius=16,
            fg_color=(fg or MECH_CARD),
            hover_color=(hover or MECH_BTN_HOVER),
            border_width=1,
            border_color=MECH_BORDER,
            text_color=MECH_TEXT,
            font=("Segoe UI", 15, "bold"),
            command=command,
        )

    def _log(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def _set_status(self, s: str):
        self.status.configure(text=f"Estado: {s}")

    def _ui_status(self, s: str):
        self.after(0, lambda: self._set_status(s))

    def _ui_finish(self):
        def _do():
            self.listening = False
            self.stop_event.clear()
            self.btn_listen.configure(text="🎙️ Escuchar", fg_color=MECH_RED, hover_color=MECH_RED_DARK)
            self._set_status("en espera")
            try:
                self.meter.set(0)
            except Exception:
                pass
        self.after(0, _do)

    # --------------------------
    # Build UI
    # --------------------------
    def _build_left(self):
        self.left = ctk.CTkFrame(self, fg_color=MECH_PANEL, corner_radius=18)
        self.left.grid(row=0, column=0, sticky="nsw", padx=(18, 10), pady=18)
        self.left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.left, text="Acciones",
            text_color=MECH_TEXT,
            font=("Segoe UI", 20, "bold")
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 10))

        row = 1
        for mode in modes_for_position(self.modes, ModePosition.LEFT):
            btn = self._big_btn(self.left, mode.title, lambda m=mode: self._run_mode(m))
            btn.grid(row=row, column=0, sticky="ew", padx=16, pady=(10, 12))
            row += 1

        self.btn_idea = self._big_btn(self.left, "Apuntar idea", self._idea_prompt)
        self.btn_idea.grid(row=row, column=0, sticky="ew", padx=16, pady=12)
        row += 1

        self.btn_sleep = self._big_btn(self.left, "Reposo", self._presence_sleep)
        self.btn_sleep.grid(row=row, column=0, sticky="ew", padx=16, pady=12)

        self.left.grid_rowconfigure(99, weight=1)

    def _build_center(self):
        self.center = ctk.CTkFrame(self, fg_color=MECH_PANEL, corner_radius=18)
        self.center.grid(row=0, column=1, sticky="nsew", padx=10, pady=18)
        self.center.grid_columnconfigure(0, weight=1)
        self.center.grid_rowconfigure(2, weight=1)

        topbar = ctk.CTkFrame(self.center, fg_color=MECH_CARD_2, corner_radius=16)
        topbar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        topbar.grid_columnconfigure(3, weight=1)

        self.btn_listen = ctk.CTkButton(
            topbar,
            text="🎙️ Escuchar",
            height=56,
            corner_radius=16,
            fg_color=MECH_RED,
            hover_color=MECH_RED_DARK,
            text_color="white",
            font=("Segoe UI", 15, "bold"),
            command=self.toggle_listen,
        )
        self.btn_listen.grid(row=0, column=0, padx=14, pady=12)

        self.status = ctk.CTkLabel(
            topbar,
            text="Estado: en espera",
            text_color=MECH_MUTED,
            font=("Segoe UI", 14, "bold")
        )
        self.status.grid(row=0, column=1, padx=10)

        # Indicador simple
        self.dot = ctk.CTkLabel(topbar, text="●", text_color=MECH_OK, font=("Segoe UI", 18, "bold"))
        self.dot.grid(row=0, column=2, padx=(10, 4))

        self.meter = ctk.CTkProgressBar(
            topbar,
            height=18,
            corner_radius=999,
            fg_color="#1b242b",
            progress_color="#2d6ea3"
        )
        self.meter.grid(row=0, column=3, sticky="ew", padx=(10, 14))
        self.meter.set(0)

        # Log box
        card = ctk.CTkFrame(
            self.center,
            fg_color=MECH_CARD,
            corner_radius=18,
            border_width=1,
            border_color=MECH_BORDER
        )
        card.grid(row=2, column=0, sticky="nsew", padx=16, pady=(6, 12))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        self.text = ctk.CTkTextbox(
            card,
            fg_color="#0b0f12",
            text_color=MECH_TEXT,
            corner_radius=14,
            border_width=1,
            border_color=MECH_BORDER,
            font=("Consolas", 13),
            wrap="word"
        )
        self.text.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        footer = ctk.CTkLabel(
            self.center,
            text="Treta Panel: wake-word + voz + botones. Acciones peligrosas requieren confirmación.",
            text_color=MECH_MUTED,
            font=("Segoe UI", 12)
        )
        footer.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 14))

    def _build_right(self):
        self.right = ctk.CTkFrame(self, fg_color=MECH_PANEL, corner_radius=18)
        self.right.grid(row=0, column=2, sticky="nse", padx=(10, 18), pady=18)
        self.right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.right, text="Sistema",
            text_color=MECH_TEXT,
            font=("Segoe UI", 20, "bold")
        ).grid(row=0, column=0, sticky="e", padx=16, pady=(16, 10))

        row = 1
        for mode in modes_for_position(self.modes, ModePosition.RIGHT):
            btn = self._big_btn(self.right, mode.title, lambda m=mode: self._run_mode(m))
            btn.grid(row=row, column=0, sticky="ew", padx=16, pady=(10, 12))
            row += 1

        self.btn_presence = self._big_btn(self.right, "Presencia ON", self._presence_on)
        self.btn_presence.grid(row=row, column=0, sticky="ew", padx=16, pady=12)
        row += 1

        self.btn_shutdown = self._big_btn(
            self.right,
            "⛔ Apagar PC (DESACTIVADO)",
            self._shutdown_placeholder,
            fg=MECH_DANGER,
            hover="#3a3f44"
        )
        self.btn_shutdown.grid(row=row, column=0, sticky="ew", padx=16, pady=(12, 18))

        self.right.grid_rowconfigure(99, weight=1)

    # --------------------------
    # Actions
    # --------------------------
    def _run_mode(self, mode):
        try:
            mode.handler(self)
        except Exception as exc:
            messagebox.showerror("Modo", f"No pude abrir {mode.title}.\n\nDetalle: {exc}")

    def _idea_prompt(self):
        self._log("▶ Ejecutado: idea_prompt\n")

    def _presence_sleep(self):
        self._log("▶ Ejecutado: presence_sleep\n")

    def _presence_on(self):
        self._log("▶ Ejecutado: presence_start\n")

    def _shutdown_placeholder(self):
        messagebox.showinfo("Apagar PC", "Shutdown sigue DESACTIVADO por seguridad.")

    # --------------------------
    # TTS (con auto-recovery)
    # --------------------------
    def _tts_worker(self):
        engine = None

        def _init_engine():
            eng = pyttsx3.init(driverName="sapi5")
            eng.setProperty("rate", 175)
            eng.setProperty("volume", 1.0)
            return eng

        engine = _init_engine()

        while True:
            msg = self.tts_q.get()
            if msg is None:
                break
            if not str(msg).strip():
                continue

            try:
                self.speaking.set()
                engine.say(msg)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)
                try:
                    engine = _init_engine()
                    engine.say(msg)
                    engine.runAndWait()
                except Exception as e2:
                    print("TTS recovery failed:", e2)
            finally:
                time.sleep(0.35)  # mini cooldown anti-eco
                self.speaking.clear()

    def speak(self, text: str):
        if not text:
            return
        s = text.strip()
        while s:
            self.tts_q.put(s[:260])
            s = s[260:]

    def _speak_blocking(self, text: str, timeout: float = 25.0):
        if not text or not str(text).strip():
            return
        self.speak(text)

        # Espera a que arranque speaking (máx 2s)
        t0 = time.time()
        while not self.speaking.is_set() and (time.time() - t0) < 2.0:
            time.sleep(0.02)

        # Espera a que termine speaking (máx timeout)
        t1 = time.time()
        while self.speaking.is_set() and (time.time() - t1) < timeout:
            time.sleep(0.05)

    # --------------------------
    # Guard rails: filtrar audio "no humano" típico
    # --------------------------
    def _looks_like_system_audio(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        # Frases típicas de subtítulos / audio de vídeo
        bad_markers = [
            "subtítulos realizados por la comunidad",
            "amara.org",
            "subtitles by the amara",
            "subtitles by amara",
        ]
        return any(m in t for m in bad_markers)

    # --------------------------
    # WAKE WORD (always-on) usando VOSK
    # --------------------------
    def _hotword_worker(self):
        try:
            from vosk import Model, KaldiRecognizer, SetLogLevel
        except Exception:
            self.after(0, lambda: self._log("⚠ Wake-word: falta 'vosk'. Instala: py -m pip install vosk\n"))
            self.hotword_enabled = False
            return

        # baja el spam de logs de Vosk
        try:
            SetLogLevel(-1)
        except Exception:
            pass

        if not os.path.exists(self.vosk_model_path):
            self.after(
                0,
                lambda: self._log(
                    "⚠ Wake-word: no encuentro modelo Vosk en:\n"
                    f"   {self.vosk_model_path}\n"
                    "   Crea la carpeta y pon un modelo ES dentro.\n"
                )
            )
            self.hotword_enabled = False
            return

        try:
            model = Model(self.vosk_model_path)
        except Exception as e:
            self.after(0, lambda: self._log(f"⚠ Wake-word: no pude cargar modelo Vosk: {e}\n"))
            self.hotword_enabled = False
            return

        wake_sr = int(self.wake_sample_rate)
        wake_ch = 1
        wake_word_norm = normalize_text(self.wake_word)
        wake_candidates = [wake_word_norm, *self.wake_word_variants]

        grammar_words = [self.wake_word, *self.wake_word_variants]
        grammar_words = [w for w in grammar_words if w]
        grammar = json.dumps([*grammar_words, "[unk]"])
        rec = KaldiRecognizer(model, wake_sr, grammar)
        try:
            rec.SetWords(False)
        except Exception:
            pass

        last_trigger = 0.0
        last_debug_log = 0.0

        self.after(0, lambda: self._log("🟠 Wake-word ON: escuchando en segundo plano…\n"))

        def _should_listen():
            if self.hotword_stop.is_set():
                return False
            if not self.hotword_enabled:
                return False
            # No oír wake-word mientras Treta habla (evita autotrigger + evita calibración sucia)
            if self.speaking.is_set():
                return False
            # No interferir con modo manual ni comando en curso
            if self.listening or self.hotword_busy.is_set():
                return False
            return True

        stream = None
        try:
            try:
                stream = sd.RawInputStream(
                    samplerate=wake_sr,
                    blocksize=8000,   # ~0.5s
                    dtype="int16",
                    channels=wake_ch,
                    device=self.mic_index,
                )
            except Exception as e:
                self.after(
                    0,
                    lambda: self._log(
                        f"⚠ Wake-word: no pude abrir micro a {wake_sr} Hz ({e}). Reintentando con default…\n"
                    ),
                )
                dev = sd.query_devices(self.mic_index) if self.mic_index is not None else sd.query_devices()
                wake_sr = int(dev.get("default_samplerate", wake_sr))
                rec = KaldiRecognizer(model, wake_sr, grammar)
                try:
                    rec.SetWords(False)
                except Exception:
                    pass
                stream = sd.RawInputStream(
                    samplerate=wake_sr,
                    blocksize=8000,
                    dtype="int16",
                    channels=wake_ch,
                    device=self.mic_index,
                )

            with stream:
                while not self.hotword_stop.is_set():
                    if not _should_listen():
                        time.sleep(0.05)
                        continue

                    data, _ = stream.read(8000)
                    if not data:
                        continue

                    try:
                        is_final = rec.AcceptWaveform(data)
                        partial = ""
                        final = ""
                        if is_final:
                            try:
                                j = json.loads(rec.Result())
                                final = (j.get("text", "") or "").strip()
                            except Exception:
                                final = ""
                        try:
                            j = json.loads(rec.PartialResult())
                            partial = (j.get("partial", "") or "").strip()
                        except Exception:
                            partial = ""
                    except Exception:
                        continue

                    partial_norm = normalize_text(partial)
                    final_norm = normalize_text(final)
                    if self.wake_debug and (partial_norm or final_norm):
                        now = time.time()
                        if now - last_debug_log >= 0.6:
                            last_debug_log = now
                            self.after(
                                0,
                                lambda pn=partial_norm, fn=final_norm: self._log(
                                    f"🧪 Wake-debug: partial='{pn}' final='{fn}'\n"
                                ),
                            )
                    matched = wake_word_matches(partial_norm, wake_candidates, self.wake_word_threshold)
                    if not matched and final_norm:
                        matched = wake_word_matches(final_norm, wake_candidates, self.wake_word_threshold)
                    if matched:
                        now = time.time()
                        if now - last_trigger < self.wake_cooldown_sec:
                            continue
                        last_trigger = now

                        self.hotword_busy.set()
                        self.after(0, lambda: self._log(f"🟠 Wake-word detectada: {self.wake_word.upper()}\n"))
                        self.after(0, lambda: self._set_status("wake-word detectada…"))

                        time.sleep(0.15)  # margen para que no meta la palabra en la grabación
                        threading.Thread(target=self._wake_command_flow, daemon=True).start()

        except Exception as e:
            self.after(0, lambda: self._log(f"⚠ Wake-word stream error: {e}\n"))
        finally:
            self.hotword_busy.clear()
            self.after(0, lambda: self._set_status("en espera"))

    def _wake_command_flow(self):
        try:
            if not self.client:
                self.after(0, lambda: self._log("⚠ Wake-word: falta OPENAI_API_KEY.\n"))
                return

            # Esperar a que termine de hablar (anti-autocalibración)
            t0 = time.time()
            while self.speaking.is_set() and (time.time() - t0) < 10.0:
                time.sleep(0.05)
            time.sleep(0.20)

            self._ui_status("escuchando comando…")
            audio = self._record_until_silence()
            self.after(0, lambda: self.meter.set(0))

            sr, _ = resolve_audio_settings(self.cfg, self.mic_index)
            if audio is None or len(audio) < int(sr * 0.25):
                self.after(0, lambda: self._log("📝 Comando: (nada claro)\n\n"))
                return

            self._ui_status("transcribiendo…")
            text = self._transcribe_openai(audio, sr)
            if not text:
                self.after(0, lambda: self._log("📝 Comando: (vacío)\n\n"))
                return

            t = text.strip()

            # Guard-rail: si parece audio de sistema (subtítulos, Amara…), pedir repetir
            if self._looks_like_system_audio(t):
                msg = "He oído audio de un vídeo, no tu voz. Repite el comando con el PC en silencio."
                self.after(0, lambda: self._log(f"⚠ Filtrado: {t}\n"))
                self.after(0, lambda: self._log(f"🤖 Treta: {msg}\n\n"))
                self._ui_status("hablando…")
                self._speak_blocking(msg)
                return

            # Si Whisper incluye "treta ..." al inicio, quitarlo
            if t.lower().startswith(self.wake_word + " "):
                t = t[len(self.wake_word):].strip(" ,.:;-")

            self.after(0, lambda: self._log(f"📝 Comando: {t}\n"))

            self._ui_status("pensando…")
            answer = self._ask_chatgpt(t)
            if answer:
                self.after(0, lambda: self._log(f"🤖 Treta: {answer}\n\n"))
                self._ui_status("hablando…")
                self._speak_blocking(answer)
            else:
                self.after(0, lambda: self._log("🤖 Treta: (sin respuesta)\n\n"))

        except Exception as e:
            self.after(0, lambda: self._log(f"❌ Error wake-command: {e}\n"))
        finally:
            self.hotword_busy.clear()
            self.after(0, lambda: self._set_status("en espera"))

    # --------------------------
    # VOZ manual: Toggle (botón) — se conserva
    # --------------------------
    def toggle_listen(self):
        # STOP
        if self.listening:
            self.stop_event.set()
            self._set_status("parando…")
            self.btn_listen.configure(text="🎙️ Escuchar", fg_color=MECH_RED, hover_color=MECH_RED_DARK)
            self._log("⏹️ Voz: parada.\n")
            self.listening = False
            return

        # START
        if not self.client:
            messagebox.showwarning("Falta API Key", "Configura OPENAI_API_KEY y reinicia Treta.")
            return

        # Si estoy hablando, esperar un poco para que la calibración no capture altavoz
        if self.speaking.is_set():
            self._log("⏳ Estoy hablando; espero para escuchar bien…\n")
            t0 = time.time()
            while self.speaking.is_set() and (time.time() - t0) < 8.0:
                time.sleep(0.05)
            time.sleep(0.20)

        self.listening = True
        self.stop_event.clear()
        self.btn_listen.configure(text="⏹️ Parar", fg_color="#3a3f44", hover_color="#4a5057")
        self._set_status("escuchando…")
        self._log("🎙️ Voz: escuchando…\n")

        threading.Thread(target=self._listen_flow, daemon=True).start()

    def _listen_flow(self):
        try:
            audio = self._record_until_silence()
            self.after(0, lambda: self.meter.set(0))

            sr, _ = resolve_audio_settings(self.cfg, self.mic_index)
            if audio is None or len(audio) < int(sr * 0.25):
                self._ui_finish()
                return

            self._ui_status("transcribiendo…")
            text = self._transcribe_openai(audio, sr)
            if not text:
                self._log("📝 Oído: (nada claro)\n\n")
                self._ui_finish()
                return

            # Guard-rail también en modo manual
            if self._looks_like_system_audio(text):
                msg = "Estoy oyendo audio de vídeo. Repite con el PC en silencio."
                self._log(f"⚠ Filtrado: {text}\n")
                self._log(f"🤖 Treta: {msg}\n\n")
                self._ui_status("hablando…")
                self._speak_blocking(msg)
                self._ui_finish()
                return

            self._log(f"📝 Oído: {text}\n")

            self._ui_status("pensando…")
            answer = self._ask_chatgpt(text)
            if answer:
                self._log(f"🤖 Treta: {answer}\n\n")
                self._ui_status("hablando…")
                self._speak_blocking(answer)
            else:
                self._log("🤖 Treta: (sin respuesta)\n\n")

        except Exception as e:
            self._log(f"❌ Error en escucha: {e}\n")
        finally:
            self._ui_finish()

    def _record_fixed(self, seconds: float, sr: int, ch: int) -> np.ndarray | None:
        frames = int(sr * seconds)
        try:
            data = sd.rec(
                frames,
                samplerate=sr,
                channels=ch,
                dtype="float32",
                device=self.mic_index,
            )
            sd.wait()
            x = data.astype(np.float32)
            if x.ndim == 2 and x.shape[1] > 1:
                x = x[:, 0]
            return x.squeeze()
        except Exception as e:
            self._log(f"❌ Error grabando audio: {e}\n")
            return None

    def _record_until_silence(self) -> np.ndarray | None:
        sr, ch = resolve_audio_settings(self.cfg, self.mic_index)

        max_sec = float(self.cfg.get("max_record_sec", 20))
        end_silence = float(self.cfg.get("end_silence_sec", 1.8))
        calib = float(self.cfg.get("noise_calib_sec", 1.0))
        mult = float(self.cfg.get("thresh_mult", 2.6))

        chunk_sec = 0.10
        start_need = float(getattr(self, "start_voice_sec", 0.20))
        min_rec = float(getattr(self, "min_record_sec", 1.0))

        self.after(0, lambda: self._log("🔇 Calibrando ruido (1s)… silencio.\n"))
        noise = self._record_fixed(calib, sr, ch)
        if noise is None:
            return None

        noise_level = _rms(noise)
        thresh = max(0.0020, noise_level * mult)

        chunks = []
        t0 = time.time()
        started = False
        voice_run = 0.0
        silence_run = 0.0
        last_meter = 0.0

        while not self.stop_event.is_set():
            elapsed = time.time() - t0
            if elapsed > max_sec:
                break

            x = self._record_fixed(chunk_sec, sr, ch)
            if x is None:
                break

            if x.ndim > 1:
                x = x[:, 0]
            x = x.astype(np.float32).squeeze()

            level = _rms(x)

            now = time.time()
            if now - last_meter > 0.10:
                last_meter = now
                v = min(1.0, level * 18.0)
                self.after(0, lambda vv=v: self.meter.set(vv))

            if not started:
                if level >= thresh:
                    voice_run += chunk_sec
                    if voice_run >= start_need:
                        started = True
                        chunks.append(x)
                    continue
                else:
                    voice_run = 0.0
                    continue

            chunks.append(x)

            if level < thresh:
                silence_run += chunk_sec
            else:
                silence_run = 0.0

            if elapsed >= min_rec and silence_run >= end_silence:
                break

        if not chunks:
            return None

        audio = np.concatenate([c.reshape(-1) for c in chunks]).astype(np.float32)
        if len(audio) < int(sr * 0.40):
            return None

        return audio

    def _transcribe_openai(self, audio: np.ndarray, sr: int) -> str:
        if not self.client:
            return ""

        wav_bytes = audio_to_wav_bytes(audio, sr)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tf.write(wav_bytes)
            path = tf.name

        try:
            with open(path, "rb") as f:
                resp = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=self.cfg.get("language", "es"),
                    temperature=0,
                    prompt="Transcribe SOLO voz humana en español. Ignora música, vídeos y audio de fondo. Si no hay voz clara, devuelve vacío."
                )
            return (getattr(resp, "text", "") or "").strip()
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def _ask_chatgpt(self, user_text: str) -> str:
        if not self.client:
            return ""
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.cfg.get("system_prompt", "Responde en español.")},
                {"role": "user", "content": user_text},
            ],
            temperature=0.4,
        )
        return (resp.choices[0].message.content or "").strip()

    # --------------------------
    # Close
    # --------------------------
    def on_close(self):
        try:
            self.hotword_stop.set()
        except Exception:
            pass
        try:
            self.stop_event.set()
        except Exception:
            pass
        try:
            self.tts_q.put(None)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    TretaApp().mainloop()
