import os
import queue
import time
import threading
import numpy as np
import sounddevice as sd

from faster_whisper import WhisperModel
from openai import OpenAI
import pyttsx3
import pythoncom


LANGUAGE = "es"
WAKE_WORDS = ["treta"]

MIC_DEVICE = 19      # ✅ tu micro nuevo
CHANNELS = 1
BLOCK_SEC = 5.0

# Anti-eco: no grabar mientras habla y un poco después
TTS_COOLDOWN_SEC = 0.8

whisper = WhisperModel("small", device="cpu", compute_type="int8")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

audio_q = queue.Queue()

tts_q: "queue.Queue[str | None]" = queue.Queue()
tts_ready = threading.Event()
tts_stop = threading.Event()

tts_speaking = threading.Event()
tts_last_end = 0.0


def _resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
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
                print(f"🗣️ TTS leyendo ({len(text)} chars)...")
                engine.say(text)
                engine.runAndWait()
                print("✅ TTS terminó frase")
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


def _try_open_inputstream(device: int, sr: int | None, channels: int) -> sd.InputStream:
    """
    Intenta abrir InputStream. sr=None => deja que PortAudio use el samplerate por defecto.
    """
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

    if not os.environ.get("OPENAI_API_KEY"):
        print("Falta OPENAI_API_KEY. Configúrala con setx y abre una terminal nueva.")
        return

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

            audio, sr_used = record_block(BLOCK_SEC)
            audio_16k = _resample_to_16k(audio, sr_used)

            text = transcribe(audio_16k)
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

