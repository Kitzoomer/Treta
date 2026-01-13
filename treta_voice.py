from __future__ import annotations

import os
import queue
import time
import threading
import argparse
import json
import logging
import wave


LANGUAGE = "es"
WAKE_WORDS = ["treta"]

MIC_DEVICE = 19      # ✅ tu micro nuevo
CHANNELS = 1
BLOCK_SEC = 5.0

# Anti-eco: no grabar mientras habla y un poco después
TTS_COOLDOWN_SEC = 0.8

whisper = None
client = None

audio_q = queue.Queue()

tts_q: "queue.Queue[str | None]" = queue.Queue()
tts_ready = threading.Event()
tts_stop = threading.Event()

tts_speaking = threading.Event()
tts_last_end = 0.0

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "treta.log")
DEBUG_WAV_PATH = os.path.join(BASE_DIR, "data", "debug_last.wav")

logger = logging.getLogger("treta")


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def _setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _log_timing(stage: str, seconds: float):
    logger.info("%s: %.2fs", stage, seconds)


def _write_debug_wav(audio: np.ndarray, sr: int, debug: bool):
    import numpy as np

    if not debug:
        return
    if audio is None or audio.size == 0:
        return
    os.makedirs(os.path.dirname(DEBUG_WAV_PATH), exist_ok=True)
    a = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm16 = (a * 32767.0).astype(np.int16)
    with wave.open(DEBUG_WAV_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _rms(x: np.ndarray) -> float:
    import numpy as np

    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _run_vad_pass(audio: np.ndarray, sr: int, cfg: dict) -> bool:
    if audio is None or audio.size == 0:
        return False
    noise_sec = float(cfg.get("noise_calib_sec", 0.2))
    thresh_mult = float(cfg.get("thresh_mult", 2.8))
    noise_frames = max(1, int(sr * noise_sec))
    noise_slice = audio[:noise_frames]
    baseline = _rms(noise_slice)
    threshold = max(baseline * thresh_mult, 0.006)
    block = max(1, int(sr * 0.05))
    for i in range(0, len(audio), block):
        if _rms(audio[i:i + block]) >= threshold:
            return True
    return False


def _resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    import numpy as np

    sr_out = 16000
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


def tts_worker():
    """Hilo dedicado a TTS. Inicializa COM para evitar silencios/bloqueos en Windows."""
    global tts_last_end
    import pyttsx3
    import pythoncom
    try:
        pythoncom.CoInitialize()
        engine = pyttsx3.init(driverName="sapi5")
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices")
        print(f"🔊 TTS listo. Voces detectadas: {len(voices)}")
        tts_ready.set()

        while not tts_stop.is_set():
            try:
                text = tts_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                tts_speaking.set()
                t0 = time.perf_counter()
                print(f"🗣️ TTS leyendo ({len(text)} chars)...")
                engine.say(text)
                engine.runAndWait()
                print("✅ TTS terminó frase")
                _log_timing("TTS", time.perf_counter() - t0)
            except Exception as e:
                print("⚠️ TTS error:", e)
            finally:
                tts_last_end = time.time()
                tts_speaking.clear()

    except Exception as e:
        print("❌ TTS worker no pudo iniciar:", e)
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


def speak(text: str):
    """Encola texto para el hilo TTS (troceado para estabilidad)."""
    if not text:
        return
    s = text.strip()
    while s:
        tts_q.put(s[:240])
        s = s[240:]


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_q.put(indata.copy())


def _try_open_inputstream(device: int, sr: int | None, channels: int) -> "sd.InputStream":
    """
    Intenta abrir InputStream. sr=None => deja que PortAudio use el samplerate por defecto.
    """
    import sounddevice as sd

    kwargs = dict(
        channels=channels,
        dtype="float32",
        device=device,
        callback=audio_callback,
    )
    if sr is not None:
        kwargs["samplerate"] = sr
    return sd.InputStream(**kwargs)


def _drain_queue(q: queue.Queue, max_items: int = 9999):
    """Vacía una cola sin bloquear."""
    for _ in range(max_items):
        try:
            q.get_nowait()
        except queue.Empty:
            break


def record_block(seconds: float) -> tuple[np.ndarray, int]:
    """
    Graba un bloque y devuelve (audio, samplerate_usado).
    Robusto: prueba varias combinaciones para evitar -9996.
    """
    import numpy as np
    import sounddevice as sd
    # Limpia cola por si quedó basura
    _drain_queue(audio_q)

    candidates = [
        (None, CHANNELS),
        (48000, CHANNELS),
        (16000, CHANNELS),
        # por si el dispositivo realmente es estéreo obligatorio:
        (None, 2),
        (48000, 2),
        (16000, 2),
    ]

    last_err = None

    for sr, ch in candidates:
        try:
            with _try_open_inputstream(MIC_DEVICE, sr, ch):
                chunks = []
                t0 = time.time()
                while time.time() - t0 < seconds:
                    try:
                        chunks.append(audio_q.get(timeout=0.5))
                    except queue.Empty:
                        pass

            if not chunks:
                used_sr = sr if sr is not None else 48000
                return np.zeros((int(used_sr * seconds),), dtype=np.float32), used_sr

            audio = np.concatenate(chunks, axis=0).astype(np.float32)

            # Si grabamos 2 canales, nos quedamos con el primero
            if audio.ndim == 2 and audio.shape[1] > 1:
                audio = audio[:, 0]
            else:
                audio = audio.squeeze()

            used_sr = sr if sr is not None else 48000
            print(f"🎛️ Grabación OK con sr={sr if sr is not None else 'auto'} ch={ch}")
            return audio, used_sr

        except sd.PortAudioError as e:
            last_err = e
            print(f"⚠️ No pude abrir micro={MIC_DEVICE} sr={sr} ch={ch}: {e}")

    raise sd.PortAudioError(
        f"Error abriendo InputStream en todas las combinaciones. Último error: {last_err}"
    )


def transcribe(audio_16k: np.ndarray) -> str:
    segments, _ = whisper.transcribe(audio_16k, language=LANGUAGE, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def extract_query(text: str):
    lower = text.lower()
    for w in WAKE_WORDS:
        idx = lower.find(w)
        if idx != -1:
            after = lower[idx + len(w):].strip(" ,.:;!?¡¿")
            return after if after else None
    return None


def ask_chatgpt(user_query: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres Treta la Torreta: técnica, clara y creativa. Responde en español."},
            {"role": "user", "content": user_query},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    global tts_last_end
    global whisper
    global client

    parser = argparse.ArgumentParser(description="Treta Voice")
    parser.add_argument("--debug", action="store_true", help="Guarda debug_last.wav y logs detallados.")
    parser.add_argument(
        "--debug-dummy",
        action="store_true",
        help="Simula capture/VAD/STT/TTS sin micro y genera logs.",
    )
    args = parser.parse_args()

    cfg = _load_config()
    _setup_logging()
    logger.info("Treta Voice iniciando.")
    logger.info("sample_rate: %s", cfg.get("sample_rate", "unknown"))
    logger.info("mic_device_index: %s", cfg.get("mic_device_index", "unknown"))
    logger.info("mic_device_name_contains: %s", cfg.get("mic_device_name_contains", "unknown"))
    logger.info(
        "thresholds: noise_calib_sec=%s, end_silence_sec=%s, thresh_mult=%s",
        cfg.get("noise_calib_sec", "unknown"),
        cfg.get("end_silence_sec", "unknown"),
        cfg.get("thresh_mult", "unknown"),
    )
    if args.debug:
        logger.info("Debug mode activo (logs/treta.log, data/debug_last.wav).")
    if args.debug_dummy:
        logger.info("Debug dummy activo (simulación sin mic).")
        stages = [
            ("capture", 0.25),
            ("VAD", 0.05),
            ("STT", 0.35),
            ("TTS", 0.15),
        ]
        for stage, seconds in stages:
            t0 = time.perf_counter()
            time.sleep(seconds)
            _log_timing(stage, time.perf_counter() - t0)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY. Configúrala con setx y abre una terminal nueva.")
        return

    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    from openai import OpenAI

    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Limpieza inicial (por si quedaron cosas de ejecuciones previas)
    _drain_queue(audio_q)
    _drain_queue(tts_q)

    th = threading.Thread(target=tts_worker, daemon=True)
    th.start()

    if not tts_ready.wait(timeout=3.0):
        print("❌ TTS no quedó listo a tiempo.")
        return

    # Info del micro
    try:
        dev = sd.query_devices(MIC_DEVICE)
        print(f"🎤 Usando micro {MIC_DEVICE}: {dev['name']} (inputs={dev['max_input_channels']})")
    except Exception as e:
        print("⚠️ No pude consultar el micro seleccionado:", e)

    speak("Treta en línea. Voz verificada. Dime: Treta, y tu pregunta.")
    time.sleep(0.2)

    print("🎙️ Treta escuchando. Di: 'Treta, ...' (Ctrl+C para salir)")

    try:
        while True:
            # Antieco
            if tts_speaking.is_set() or (time.time() - tts_last_end) < TTS_COOLDOWN_SEC:
                time.sleep(0.05)
                continue

            capture_t0 = time.perf_counter()
            audio, sr_used = record_block(BLOCK_SEC)
            _log_timing("capture", time.perf_counter() - capture_t0)
            vad_t0 = time.perf_counter()
            _run_vad_pass(audio, sr_used, cfg)
            _log_timing("VAD", time.perf_counter() - vad_t0)
            _write_debug_wav(audio, sr_used, args.debug)
            audio_16k = _resample_to_16k(audio, sr_used)

            stt_t0 = time.perf_counter()
            text = transcribe(audio_16k)
            _log_timing("STT", time.perf_counter() - stt_t0)
            if not text:
                continue

            print("📝 Oído:", text)
            q = extract_query(text)

            if q is None and "treta" in text.lower():
                speak("Te escucho. Dime tu pregunta.")
                continue

            if q:
                speak("Procesando.")
                answer = ask_chatgpt(q)
                print("🤖 Treta:", answer)
                time.sleep(0.2)
                speak(answer)

    except KeyboardInterrupt:
        speak("Treta apagando escucha.")
    finally:
        tts_stop.set()
        tts_q.put(None)


if __name__ == "__main__":
    main()
