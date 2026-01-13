import os, json, time, threading, queue, subprocess
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

LOG_PATH = os.path.join(BRIDGE_DIR, "log.txt")
PENDING_CONFIRM_PATH = os.path.join(BRIDGE_DIR, "pending_confirm.json")

STATE_PATH = os.path.join(DATA_DIR, "state.json")
IDEAS_PATH = os.path.join(DATA_DIR, "ideas.jsonl")
DIARY_PATH = os.path.join(DATA_DIR, "diary.jsonl")


def _ensure_dirs():
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ACTIONS_DIR, exist_ok=True)


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
    def __init__(self):
        super().__init__()
        _ensure_dirs()

        self.cfg = load_config()

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
        self._log(f"🎤 Mic: {self.mic_index if self.mic_index is not None else 'default del sistema'}\n\n")

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
                engine.say(msg)
                engine.runAndWait()
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
            audio = self._record_until_silence()
            self.after(0, lambda: self.meter.configure(value=0))

            sr = int(self.cfg.get("sample_rate", 48000))
            if audio is None or len(audio) < int(sr * 0.3):
                self._finish_listen()
                return

            self._set_status("transcribiendo…")
            text = self._transcribe(audio)
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

    # ---------- Audio capture ----------
    def _record_until_silence(self):
        sr = int(self.cfg.get("sample_rate", 48000))
        ch = int(self.cfg.get("channels", 1))
        max_sec = float(self.cfg.get("max_record_sec", 90))
        noise_calib = float(self.cfg.get("noise_calib_sec", 0.6))
        end_silence = float(self.cfg.get("end_silence_sec", 1.2))
        thresh_mult = float(self.cfg.get("thresh_mult", 2.8))

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

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).astype(np.float32)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio[:, 0]
        return audio.squeeze()

    def _record_fixed(self, seconds: float, sr: int, ch: int):
        frames = int(sr * seconds)
        try:
            data = sd.rec(frames, samplerate=sr, channels=ch, dtype="float32", device=self.mic_index)
            sd.wait()
            x = data.astype(np.float32)
            if x.ndim == 2 and x.shape[1] > 1:
                x = x[:, 0]
            return x.squeeze()
        except Exception as e:
            self._log(f"❌ Error grabando audio: {e}\n")
            return None

    def _transcribe(self, audio: np.ndarray) -> str:
        sr_in = int(self.cfg.get("sample_rate", 48000))
        audio_16k = resample_linear(audio, sr_in, 16000)
        segments, _ = self.whisper.transcribe(audio_16k, language=self.cfg.get("language", "es"), vad_filter=True)
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
    TretaPanel().mainloop()
