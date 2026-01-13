import os
import json
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import sounddevice as sd
import pyttsx3

from openai import OpenAI
from faster_whisper import WhisperModel


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


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
    """
    Devuelve índice de micro si se puede escoger, o None para el default del sistema.
    - Si cfg["mic_device_index"] != null, usa ese.
    - Si cfg["mic_device_name_contains"] tiene texto, busca un dispositivo cuyo nombre lo contenga.
    """
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


class TretaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()

        # Ventana
        self.title("Treta — Asistente de voz")
        self.geometry("820x580")
        self.minsize(720, 520)

        # Tema/estilo (mejora visual)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            style.theme_use("clam")
        style.configure("TButton", padding=(10, 6))
        style.configure("TLabel", padding=(4, 2))

        # OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Whisper local
        self.whisper = WhisperModel(self.cfg["model"], device="cpu", compute_type="int8")

        # Estado
        self.listening = False
        self.stop_event = threading.Event()

        # Anti-eco: no escuchar mientras habla
        self.speaking = threading.Event()

        # Elegir micro (por índice o por nombre)
        self.mic_index = pick_input_device(self.cfg)

        # TTS en hilo
        self.tts_q: "queue.Queue[str | None]" = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # UI
        self._build_ui()

        # Logs iniciales
        self._log("Treta lista.\n")
        if not api_key:
            self._log("⚠ Falta OPENAI_API_KEY. Configúrala y reinicia Treta.\n")
        else:
            self._log("✅ OPENAI_API_KEY detectada.\n")

        # Info de micro
        self._log(
            f"🎤 Mic configurado: {self.mic_index if self.mic_index is not None else 'default del sistema'}\n\n"
        )

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        self.btn = ttk.Button(top, text="🎙️ Escuchar", command=self.toggle_listen)
        self.btn.pack(side="left")

        self.status = ttk.Label(top, text="Estado: en espera")
        self.status.pack(side="left", padx=10)

        self.meter = ttk.Progressbar(top, length=260, mode="determinate")
        self.meter.pack(side="right")

        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        self.text = scrolledtext.ScrolledText(body, wrap="word", font=("Segoe UI", 11))
        self.text.pack(fill="both", expand=True)

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Label(bottom, text="Modo Toggle: clic para empezar, clic para parar.").pack(side="left")

    def _log(self, s: str):
        self.text.insert("end", s)
        self.text.see("end")

    def _set_status(self, s: str):
        self.status.config(text=f"Estado: {s}")

    def _ui_status(self, s: str):
        self.after(0, lambda: self._set_status(s))

    def _ui_finish(self):
        def _do():
            self.listening = False
            self.stop_event.clear()
            self.btn.config(text="🎙️ Escuchar")
            self._set_status("en espera")
        self.after(0, _do)

    # ---------------- TTS ----------------
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
            except Exception as e:
                print("TTS error:", e)
            finally:
                # cooldown anti-eco
                time.sleep(0.4)
                self.speaking.clear()

    def speak(self, text: str):
        if not text:
            return
        s = text.strip()
        while s:
            self.tts_q.put(s[:260])
            s = s[260:]

    # ---------------- Toggle ----------------
    def toggle_listen(self):
        if self.listening:
            self.stop_event.set()
            self.btn.config(text="🎙️ Escuchar")
            self._set_status("parando…")
            return

        if not self.client:
            messagebox.showwarning("Falta API Key", "Configura OPENAI_API_KEY y reinicia Treta.")
            return

        if self.speaking.is_set():
            self._log("⏳ Estoy hablando; espera un segundo.\n")
            return

        self.listening = True
        self.stop_event.clear()
        self.btn.config(text="⏹️ Parar")
        self._set_status("escuchando…")

        threading.Thread(target=self._listen_flow, daemon=True).start()

    def _listen_flow(self):
        try:
            audio = self._record_until_silence()
            self.after(0, lambda: self.meter.configure(value=0))

            sr = int(self.cfg["sample_rate"])
            if audio is None or len(audio) < int(sr * 0.3):
                self._ui_finish()
                return

            self._ui_status("transcribiendo…")
            text = self._transcribe(audio)
            if not text:
                self._log("📝 Oído: (nada claro)\n\n")
                self._ui_finish()
                return

            self._log(f"📝 Oído: {text}\n")
            self._ui_status("pensando…")
            answer = self._ask_chatgpt(text)
            self._log(f"🤖 Treta: {answer}\n\n")

            self._ui_status("hablando…")
            self.speak(answer)

        except Exception as e:
            self._log(f"❌ Error: {e}\n\n")
        finally:
            self._ui_finish()

    # ---------------- Grabación hasta silencio ----------------
    def _record_until_silence(self) -> np.ndarray | None:
        sr = int(self.cfg["sample_rate"])
        ch = int(self.cfg["channels"])
        max_sec = float(self.cfg["max_record_sec"])
        noise_calib = float(self.cfg["noise_calib_sec"])
        end_silence = float(self.cfg["end_silence_sec"])
        thresh_mult = float(self.cfg["thresh_mult"])

        # 1) calibración de ruido
        self._ui_status("calibrando ruido…")
        noise = self._record_fixed(noise_calib, sr, ch)
        if noise is None:
            return None

        noise_level = rms(noise)
        thresh = max(noise_level * thresh_mult, 0.006)

        # 2) grabar hasta silencio o stop
        self._ui_status("escuchando…")
        chunks = []
        silence_run = 0.0
        t0 = time.time()

        def callback(indata, frames, time_info, status):
            if status:
                pass
            chunks.append(indata.copy())

        device = self.mic_index  # None => default del sistema

        with sd.InputStream(
            samplerate=sr,
            channels=ch,
            dtype="float32",
            device=device,
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

                    now = time.time()
                    if now - last_meter > 0.12:
                        last_meter = now
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

        # mono
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio[:, 0]

        return audio.squeeze()

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

    # ---------------- Whisper + ChatGPT ----------------
    def _transcribe(self, audio: np.ndarray) -> str:
        sr_in = int(self.cfg["sample_rate"])
        audio_16k = resample_linear(audio, sr_in, 16000)

        segments, _ = self.whisper.transcribe(
            audio_16k,
            language=self.cfg["language"],
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _ask_chatgpt(self, user_text: str) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.cfg["system_prompt"]},
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
    TretaApp().mainloop()
