"""Datos que la interfaz muestra en cada estado: el audio del visor y el resultado del analisis."""

from __future__ import annotations

import base64
import hashlib
import io

import librosa
import numpy as np
import soundfile as sf

from src.inferencia import (
    ERROR,
    MODEL_LABELS,
    PROTOTYPE_TAG,
    InferenceError,
    analyze_cycle,
    check_input,
    decode_wav,
    predict,
    trim,
)
from src.preprocesamiento import resample_audio

CLASS_NAMES = {"NORMAL": "Normal", "CREPITANTE": "Crepitante", "SIBILANCIA": "Sibilancia"}
ENVELOPE_BINS = 1600
SPECTROGRAM_FRAMES = 900
SPECTROGRAM_BANDS = 96
SPECTROGRAM_RANGE_DB = 65.0
SPECTROGRAM_TICKS_HZ = (100, 500, 2000)
PLAYBACK_MAX_RATE = 16000


def envelope(audio: np.ndarray, bins: int = ENVELOPE_BINS) -> tuple[list[int], list[int]]:
    """Minimo y maximo de la senal por columna, en enteros de -127 a 127 respecto al pico."""
    audio = np.asarray(audio, dtype=np.float64)
    bins = max(1, min(bins, len(audio)))
    edges = np.linspace(0, len(audio), bins + 1).astype(int)[:-1]
    peak = float(np.max(np.abs(audio))) or 1.0
    low = np.minimum.reduceat(audio, edges) / peak
    high = np.maximum.reduceat(audio, edges) / peak
    return np.round(127 * low).astype(int).tolist(), np.round(127 * high).astype(int).tolist()


def spectrogram(audio: np.ndarray, sample_rate: int, cfg) -> dict:
    """Espectrograma en dB de la banda que usa el modelo, con la frecuencia en escala logaritmica.

    La escala logaritmica reparte mejor los sonidos respiratorios, que se concentran en los graves.
    Cada celda ocupa un byte.
    """
    p = cfg.preprocessing
    low_hz, high_hz = float(p.bandpass_low_hz), float(p.bandpass_high_hz)
    signal = resample_audio(np.asarray(audio, dtype=np.float64), sample_rate, p.target_sr, p.resample_kaiser_beta)
    n_fft = 512
    hop = max(32, int(np.ceil(len(signal) / SPECTROGRAM_FRAMES)))
    power = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    bands = np.geomspace(low_hz, high_hz, SPECTROGRAM_BANDS) * n_fft / p.target_sr
    below = np.floor(bands).astype(int)
    weight = (bands - below)[:, None]
    level = 10.0 * np.log10((1.0 - weight) * power[below] + weight * power[below + 1] + 1e-12)
    top = float(level.max())
    scaled = np.clip((level - (top - SPECTROGRAM_RANGE_DB)) / SPECTROGRAM_RANGE_DB, 0.0, 1.0)
    cells = np.round(255 * scaled).astype(np.uint8)
    ticks = [{"label": f"{hz / 1000:g} kHz" if hz >= 1000 else f"{hz:g}",
              "at": float(np.log(hz / low_hz) / np.log(high_hz / low_hz))}
             for hz in SPECTROGRAM_TICKS_HZ if low_hz <= hz <= high_hz]
    return {"rows": int(cells.shape[0]), "cols": int(cells.shape[1]), "ticks": ticks,
            "cells": base64.b64encode(cells.tobytes()).decode("ascii")}


def playback_wav(audio: np.ndarray, sample_rate: int, cfg) -> str:
    """WAV de 16 bits para el reproductor del navegador, como URL de datos."""
    signal = np.asarray(audio, dtype=np.float64)
    if sample_rate > PLAYBACK_MAX_RATE:
        signal = resample_audio(signal, sample_rate, PLAYBACK_MAX_RATE, cfg.preprocessing.resample_kaiser_beta)
        sample_rate = PLAYBACK_MAX_RATE
    peak = float(np.max(np.abs(signal))) or 1.0
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(signal / max(peak, 1.0), -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def audio_view(data: bytes, cfg) -> dict:
    """Todo lo que el visor necesita de un archivo: envolvente, espectrograma y audio para reproducir."""
    audio, sample_rate = decode_wav(data)
    low, high = envelope(audio)
    return {
        "id": hashlib.sha256(data).hexdigest()[:16],
        "duration": len(audio) / sample_rate,
        "sample_rate": int(sample_rate),
        "low": low,
        "high": high,
        "spectrogram": spectrogram(audio, sample_rate, cfg),
        "wav": playback_wav(audio, sample_rate, cfg),
    }


def model_options(models: dict) -> list[dict]:
    options = []
    for key in ("prototype", "best_model"):
        if key not in models:
            continue
        bundle = models[key]
        members = bundle.metadata.get("members") or []
        kind = f"{len(members)} clasificadores" if members else "SVM"
        options.append({"key": key, "name": MODEL_LABELS[key],
                        "spec": f"{kind} · {len(bundle.feature_names)} características"})
    return options


def result_view(prediction, annotation: str | None) -> dict:
    label = prediction.label
    view = {
        "label": label,
        "name": CLASS_NAMES[label],
        "probabilities": [{"key": key, "name": CLASS_NAMES[key], "value": float(value)}
                          for key, value in prediction.probabilities.items()],
        "votes": None,
        "doubt": None,
        "annotation": None,
    }
    if prediction.votes:
        view["votes"] = {"agree": sum(vote == label for _, vote in prediction.votes), "total": len(prediction.votes)}
    if prediction.probability_label != label:
        view["doubt"] = CLASS_NAMES[prediction.probability_label]
    if annotation:
        view["annotation"] = {"name": CLASS_NAMES[annotation], "match": annotation == label}
    return view


def new_session(default_model: str) -> dict:
    return {"file": None, "name": "", "example": None, "annotation": None, "model": default_model,
            "selection": None, "analyzed": None, "seq": 0, "seen": set(), "error": ""}


def apply_events(session: dict, events, examples, examples_dir, model_keys) -> bool:
    """Aplica al estado lo que llego de la interfaz. Devuelve True si algo cambio."""
    if not events:
        return False
    changed = False

    def fresh(event) -> bool:
        stamp = event.get("t") if isinstance(event, dict) else None
        if stamp is None or stamp in session["seen"]:
            return False
        session["seen"].add(stamp)
        return True

    def load(data: bytes, name: str, example, annotation) -> None:
        session.update(file=data, name=name, example=example, annotation=annotation, selection=None,
                       analyzed=None, error="")

    upload = events.get("file")
    if upload and fresh(upload):
        try:
            data = base64.b64decode(upload["data"])
            decode_wav(data)
            load(data, str(upload.get("name", "audio.wav")), None, None)
        except (InferenceError, ValueError, KeyError) as error:
            session["error"] = str(error) if isinstance(error, InferenceError) else "No se pudo leer el archivo."
        changed = True

    example = events.get("example")
    if example and fresh(example) and 0 <= int(example["index"]) < len(examples):
        row = examples.iloc[int(example["index"])]
        load((examples_dir / row["file"]).read_bytes(), str(row["file"]), int(example["index"]), str(row["label"]))
        changed = True

    model = events.get("model")
    if model in model_keys and model != session["model"]:
        session["model"] = model
        changed = True

    current = audio_id(session)
    selection = events.get("selection")
    if selection and current and selection.get("id") == current:
        span = (round(float(selection["start"]), 3), round(float(selection["end"]), 3))
        if span != session["selection"]:
            session["selection"] = span
            session["analyzed"] = None
            changed = True

    analyze = events.get("analyze")
    if analyze and fresh(analyze) and current and analyze.get("id") == current:
        session["analyzed"] = (current, session["selection"])
        changed = True

    if changed:
        session["seq"] += 1
    return changed


def audio_id(session: dict) -> str | None:
    return hashlib.sha256(session["file"]).hexdigest()[:16] if session["file"] else None


def analyze(data: bytes, start: float, end: float, cfg, models: dict):
    """Avisos sobre el tramo y, si se puede clasificar, la prediccion de cada modelo."""
    audio, sample_rate = decode_wav(data)
    segment = trim(audio, sample_rate, start, end)
    checks = check_input(segment, sample_rate, cfg)
    if any(check.level == ERROR for check in checks):
        return checks, {}
    analysis = analyze_cycle(segment, sample_rate, cfg)
    return checks, {key: predict(bundle, analysis) for key, bundle in models.items()}


def build_payload(session: dict, cfg, models: dict, examples, view, run, max_upload_mb: int = 50) -> dict:
    """Lo que la interfaz dibuja: ``view(bytes)`` describe el audio y ``run(bytes, inicio, fin)`` lo analiza."""
    data = {
        "tag": PROTOTYPE_TAG,
        "seq": session["seq"],
        "stage": "idle",
        "audio": None,
        "selection": None,
        "example": session["example"],
        "examples": [{"title": f"{CLASS_NAMES[row.label]} · {row.duration_sec:.2f} s".replace(".", ",")}
                     for row in examples.itertuples()],
        "models": model_options(models),
        "model": session["model"],
        "result": None,
        "messages": [],
        "error": session["error"],
        "max_mb": max_upload_mb,
        "min_span": cfg.app.min_duration_sec,
    }
    if session["file"] is None:
        return data
    audio = view(session["file"])
    start, end = session["selection"] or (0.0, audio["duration"])
    data.update(stage="loaded", audio={**audio, "name": session["name"]}, selection=[start, end])
    if session["analyzed"] != (audio["id"], session["selection"]):
        return data
    data["stage"] = "result"
    try:
        checks, predictions = run(session["file"], start, end)
    except InferenceError as error:
        data["messages"] = [str(error)]
        return data
    data["messages"] = [check.message for check in checks]
    if predictions:
        annotation = session["annotation"] if session["selection"] is None else None
        data["result"] = result_view(predictions[session["model"]], annotation)
    return data
