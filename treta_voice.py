from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TARGET_SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(TARGET_SAMPLE_RATE * FRAME_MS / 1000)

# Anti-eco: no grabar mientras habla y un poco después
TTS_COOLDOWN_SEC = 0.8


CONFIG_PATH = Path(os.environ.get("TRETA_CONFIG", BASE_DIR / "config.json"))


def load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


_config = load_config(CONFIG_PATH)

LANGUAGE = "es"
WAKE_WORDS = ["treta"]

MIC_NAME_CONTAINS = (_config.get("mic_device_name_contains") or "").strip()
MAX_RECORD_SEC = float(_config.get("max_record_sec", 20))
END_SILENCE_SEC = float(_config.get("end_silence_sec", 1.8))

whisper = None
client: OpenAI | None = None

audio_q = queue.Queue()

tts_q: "queue.Queue[str | None]" = queue.Queue()
tts_ready = threading.Event()
tts_stop = threading.Event()

tts_speaking = threading.Event()
tts_last_end = 0.0

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "treta.log"
logger = logging.getLogger("treta")


def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _log_timing(stage: str, seconds: float):
    logger.info("%s: %.2fs", stage, seconds)




def _resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    sr_out = TARGET_SAMPLE_RATE
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


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def list_input_devices() -> list[tuple[int, dict]]:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append((idx, dev))
    return devices


def print_input_devices() -> None:
    for idx, dev in list_input_devices():
        print(
            idx,
            "|",
            dev["name"],
            "| inputs:",
            dev["max_input_channels"],
            "| default_sr:",
            dev.get("default_samplerate"),
        )


def find_device_by_name_contains(name_contains: str) -> int | None:
    if not name_contains:
        return None
    needle = name_contains.lower()
    matches: list[tuple[int, float]] = []
    for idx, dev in list_input_devices():
        if needle in dev.get("name", "").lower():
            default_sr = float(dev.get("default_samplerate") or 0)
            matches.append((idx, default_sr))
    if matches:
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[0][0]
    return None


def default_input_device() -> int | None:
    default_device = sd.default.device[0]
    if default_device is not None and default_device >= 0:
        return int(default_device)
    devices = list_input_devices()
    if devices:
        return devices[0][0]
    return None


def vad_flags_to_segments(flags: list[bool], frame_ms: int = FRAME_MS) -> list[dict]:
    if not flags:
        return []
    segments: list[dict] = []
    frame_sec = frame_ms / 1000.0
    current_type = "speech" if flags[0] else "silence"
    start_idx = 0
    for idx, flag in enumerate(flags[1:], start=1):
        next_type = "speech" if flag else "silence"
        if next_type != current_type:
            segments.append(
                {"type": current_type, "start": start_idx * frame_sec, "end": idx * frame_sec}
            )
            start_idx = idx
            current_type = next_type
    segments.append(
        {"type": current_type, "start": start_idx * frame_sec, "end": len(flags) * frame_sec}
    )
    return segments


def save_debug_audio(audio_16k: np.ndarray, path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pcm16 = _float_to_pcm16(audio_16k)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm16)


def save_debug_segments(segments: list[dict], path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def tts_worker():
    """Hilo dedicado a TTS. Inicializa COM para evitar silencios/bloqueos en Windows."""
    global tts_last_end
    try:
        import pyttsx3
        import pythoncom

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
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:
            pass


def speak(text: str):
    """Encola texto para el hilo TTS (troceado para estabilidad)."""
    if not text:
        return
    s = text.strip()
    if not s:
        return
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


def record_utterance(
    max_record_sec: float,
    end_silence_sec: float,
    device: int,
    channels: int,
    vad,
    preferred_sr: int | None = None,
) -> tuple[np.ndarray, list[dict], bool]:
    """
    Graba hasta detectar fin de frase usando VAD y hangover.
    Devuelve (audio_16k, segments, speech_detected).
    """
    # Limpia cola por si quedó basura
    _drain_queue(audio_q)

    candidates = []
    if preferred_sr is not None:
        candidates.append((preferred_sr, channels))
    candidates.extend(
        [
            (None, channels),
            (48000, channels),
            (16000, channels),
        ]
    )
    # por si el dispositivo realmente es estéreo obligatorio:
    candidates.extend([(None, 2), (48000, 2), (16000, 2)])

    last_err = None
    dev_info = sd.query_devices(device)
    default_sr = int(dev_info.get("default_samplerate", 48000))
    max_frames = max(1, int(max_record_sec * 1000 / FRAME_MS))
    end_silence_frames = max(1, int(end_silence_sec * 1000 / FRAME_MS))

    for sr, ch in candidates:
        try:
            used_sr = int(sr if sr is not None else default_sr)
            with _try_open_inputstream(device, sr, ch):
                audio_parts: list[np.ndarray] = []
                frame_buffer = np.zeros((0,), dtype=np.float32)
                vad_flags: list[bool] = []
                frames_processed = 0
                speech_detected = False
                last_speech_frame = None
                stop = False
                start_time = time.time()

                while frames_processed < max_frames and not stop:
                    try:
                        chunk = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        if time.time() - start_time > max_record_sec + 1.0:
                            break
                        continue

                    if chunk.ndim == 2 and chunk.shape[1] > 1:
                        chunk = chunk[:, 0]
                    chunk = chunk.squeeze().astype(np.float32)
                    chunk_16k = _resample_to_16k(chunk, used_sr)
                    if chunk_16k.size == 0:
                        continue

                    audio_parts.append(chunk_16k)
                    frame_buffer = np.concatenate([frame_buffer, chunk_16k])

                    while frame_buffer.shape[0] >= FRAME_SAMPLES and frames_processed < max_frames:
                        frame = frame_buffer[:FRAME_SAMPLES]
                        frame_buffer = frame_buffer[FRAME_SAMPLES:]
                        is_speech = vad.is_speech(_float_to_pcm16(frame), TARGET_SAMPLE_RATE)
                        vad_flags.append(is_speech)
                        frames_processed += 1

                        if is_speech:
                            speech_detected = True
                            last_speech_frame = frames_processed
                        if speech_detected and last_speech_frame is not None:
                            if frames_processed - last_speech_frame >= end_silence_frames:
                                stop = True
                                break

                if not audio_parts:
                    return np.zeros((0,), dtype=np.float32), [], False

                audio_16k = np.concatenate(audio_parts, axis=0)
                if frames_processed > 0:
                    audio_16k = audio_16k[:frames_processed * FRAME_SAMPLES]
                segments = vad_flags_to_segments(vad_flags)
                print(f"🎛️ Grabación OK con sr={sr if sr is not None else 'auto'} ch={ch}")
                return audio_16k, segments, speech_detected

        except sd.PortAudioError as e:
            last_err = e
            print(f"⚠️ No pude abrir micro={device} sr={sr} ch={ch}: {e}")

    raise sd.PortAudioError(
        f"Error abriendo InputStream en todas las combinaciones. Último error: {last_err}"
    )


def transcribe(audio_16k: np.ndarray, gain_db: float = 0.0, prompt: str | None = None) -> str:
    global whisper
    if whisper is None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError:
            print("⚠️ faster_whisper no está instalado. Instálalo para habilitar la transcripción.")
            return ""
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
    audio_16k = apply_gain(audio_16k, gain_db)
    kwargs = dict(language=LANGUAGE, vad_filter=True)
    if prompt:
        kwargs["initial_prompt"] = prompt
    segments, _ = whisper.transcribe(audio_16k, **kwargs)
    return " ".join(seg.text.strip() for seg in segments).strip()


def extract_query(text: str):
    lower = text.lower()
    for w in WAKE_WORDS:
        idx = lower.find(w)
        if idx != -1:
            after = lower[idx + len(w):].strip(" ,.:;!?¡¿")
            return after if after else None
    return None


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if not audio.size or gain_db == 0:
        return audio
    gain = float(10 ** (gain_db / 20))
    boosted = audio * gain
    return np.clip(boosted, -1.0, 1.0)


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


def ask_chatgpt(user_query: str) -> str:
    global client
    if not os.environ.get("OPENAI_API_KEY"):
        return "API key no configurada."
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres Treta la Torreta: técnica, clara y creativa. Responde en español.",
                },
                {"role": "user", "content": user_query},
            ],
            temperature=0.4,
        )
    except Exception:
        logger.exception("Error consultando OpenAI")
        return "Tuve un problema al consultar el modelo. Intenta de nuevo."
    return (resp.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treta voice assistant")
    parser.add_argument("--list-mics", action="store_true", help="Listar micrófonos disponibles y salir.")
    parser.add_argument("--mic", type=int, help="ID de micrófono a usar (override).")
    parser.add_argument("--debug", action="store_true", help="Guardar audio/segmentos de depuración.")
    parser.add_argument(
        "--debug-dummy",
        action="store_true",
        help="Simula capture/VAD/STT/TTS sin micro y genera logs.",
    )
    return parser.parse_args()


def main():
    global tts_last_end

    cfg = load_config(CONFIG_PATH)
    _setup_logging()
    args = parse_args()
    logger.info("Treta Voice iniciando.")
    logger.info("sample_rate: %s", cfg.get("sample_rate", "unknown"))
    logger.info("channels: %s", cfg.get("channels", "unknown"))
    logger.info("input_gain_db: %s", cfg.get("input_gain_db", 0.0))
    logger.info("stt_initial_prompt: %s", cfg.get("stt_initial_prompt", ""))
    logger.info("min_input_rms: %s", cfg.get("min_input_rms", 0.0))
    logger.info("max_input_rms: %s", cfg.get("max_input_rms", 1.0))
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

    if args.list_mics:
        print_input_devices()
        return

    from faster_whisper import WhisperModel

    whisper = WhisperModel("small", device="cpu", compute_type="int8")

    # Limpieza inicial (por si quedaron cosas de ejecuciones previas)
    _drain_queue(audio_q)
    _drain_queue(tts_q)

    th = threading.Thread(target=tts_worker, daemon=True)
    th.start()

    if not tts_ready.wait(timeout=3.0):
        print("❌ TTS no quedó listo a tiempo.")
        return

    vad = webrtcvad.Vad(2)
    channels = parse_int(cfg.get("channels", 1)) or 1
    preferred_sr = parse_int(cfg.get("sample_rate", None))
    input_gain_db = float(cfg.get("input_gain_db", 0.0))
    min_rms = parse_float(cfg.get("min_input_rms", 0.0), 0.0) or 0.0
    max_rms = parse_float(cfg.get("max_input_rms", 1.0), 1.0) or 1.0
    stt_prompt = (cfg.get("stt_initial_prompt") or "").strip() or None

    if args.mic is not None:
        mic_device = args.mic
    else:
        mic_device = find_device_by_name_contains(MIC_NAME_CONTAINS)
        if mic_device is None:
            if MIC_NAME_CONTAINS:
                print(
                    f"⚠️ No encontré micros que contengan '{MIC_NAME_CONTAINS}'. Usando el dispositivo por defecto."
                )
            mic_device = default_input_device()
        else:
            print(
                f"🎚️ Micro seleccionado por nombre (contains='{MIC_NAME_CONTAINS}'): id {mic_device}."
            )
    if mic_device is None:
        print("❌ No se encontró ningún dispositivo de entrada.")
        print("🧾 Dispositivos de entrada disponibles:")
        print_input_devices()
        return

    # Info del micro
    try:
        dev = sd.query_devices(mic_device)
        print(
            f"🎤 Usando micro {mic_device}: {dev['name']} "
            f"(inputs={dev['max_input_channels']}, default_sr={dev.get('default_samplerate', 'n/a')})"
        )
    except Exception as e:
        print(f"⚠️ No pude consultar el micro {mic_device}: {e}")
        fallback_device = default_input_device()
        if fallback_device is not None and fallback_device != mic_device:
            try:
                dev = sd.query_devices(fallback_device)
                mic_device = fallback_device
                print(f"🎤 Usando micro {mic_device}: {dev['name']} (inputs={dev['max_input_channels']})")
            except Exception as err:
                print(f"❌ No pude consultar el micro por defecto {fallback_device}: {err}")
                print("🧾 Dispositivos de entrada disponibles:")
                print_input_devices()
                return
        else:
            print("🧾 Dispositivos de entrada disponibles:")
            print_input_devices()
            return

    speak("Treta en línea. Voz verificada. Dime: Treta, y tu pregunta.")
    time.sleep(0.2)

    print("🎙️ Treta escuchando. Di: 'Treta, ...' (Ctrl+C para salir)")

    empty_streak = 0
    try:
        while True:
            # Antieco
            if tts_speaking.is_set() or (time.time() - tts_last_end) < TTS_COOLDOWN_SEC:
                time.sleep(0.05)
                continue

            try:
                audio_16k, segments, speech_detected = record_utterance(
                    MAX_RECORD_SEC,
                    END_SILENCE_SEC,
                    mic_device,
                    channels,
                    vad,
                    preferred_sr,
                )
            except sd.PortAudioError as e:
                fallback_device = default_input_device()
                if fallback_device is None or fallback_device == mic_device:
                    print(f"❌ No se pudo abrir el micro {mic_device}: {e}")
                    print("🧾 Dispositivos de entrada disponibles:")
                    print_input_devices()
                    continue
                print(f"⚠️ No se pudo abrir el micro {mic_device}: {e}")
                print(f"🔁 Reintentando con el dispositivo por defecto {fallback_device}.")
                try:
                    audio_16k, segments, speech_detected = record_utterance(
                        MAX_RECORD_SEC,
                        END_SILENCE_SEC,
                        fallback_device,
                        channels,
                        vad,
                        preferred_sr,
                    )
                    mic_device = fallback_device
                except sd.PortAudioError as err:
                    print(f"❌ No se pudo abrir el micro por defecto {fallback_device}: {err}")
                    print("🧾 Dispositivos de entrada disponibles:")
                    print_input_devices()
                    continue
            if args.debug:
                save_debug_audio(audio_16k, DATA_DIR / "debug_last.wav")
                save_debug_segments(segments, DATA_DIR / "debug_last_segments.json")
            if not speech_detected or audio_16k.size == 0:
                continue
            level = float(np.sqrt(np.mean(audio_16k * audio_16k) + 1e-12))
            if level < min_rms:
                print("⚠️ Audio muy bajo. Revisa el micro o sube input_gain_db.")
                speak("No te oigo bien. Sube el volumen del micro o acércate.")
                continue
            if level > max_rms:
                print("⚠️ Audio saturado. Baja el volumen del micro.")

            stt_t0 = time.perf_counter()
            text = transcribe(audio_16k, gain_db=input_gain_db, prompt=stt_prompt)
            _log_timing("STT", time.perf_counter() - stt_t0)
            if not text:
                empty_streak += 1
                if empty_streak >= 3:
                    speak("No te he entendido. Repite tu pregunta, por favor.")
                    empty_streak = 0
                continue

            print("📝 Oído:", text)
            q = extract_query(text)
            empty_streak = 0

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
