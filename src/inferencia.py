"""Inferencia sobre un ciclo nuevo (Fase 12).

Un WAV que entra por la interfaz recorre exactamente la ruta de entrenamiento::

    audio -> mono -> remuestreo -> pasa-banda -> normalizacion RMS
          -> 4 representaciones + descriptores -> pipeline congelado (escalador + clasificador)

No se reajusta nada: el ``StandardScaler`` es el que quedo guardado dentro de cada
pipeline. La clase que se muestra es la de ``model.predict`` (la regla evaluada en la
Fase 7); las probabilidades son una estimacion aparte y pueden no tener su maximo en
esa clase. No hay segmentacion automatica: se clasifica el tramo que el usuario entrega.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from src.configuracion import Config
from src.descriptores import compute_descriptors
from src.extraccion_caracteristicas import (
    FeatureExtractionError,
    extract_representations,
    frame_params,
    log_mel_spectrogram,
)
from src.evaluacion_final import EvaluationSetupError
from src.preprocesamiento import PreprocessingError, PreprocessingResult, preprocess_audio, rms, to_mono
from src.robustez import MODEL_FILES, feature_columns, load_frozen_models

PROTOTYPE_TAG = "Prototipo académico de investigación"
MODEL_LABELS = {"prototype": "Mejorado", "best_model": "Experimental"}
MEMBER_NAMES = {"random_forest": "Random Forest", "svm_rbf": "SVM", "knn": "KNN"}
ERROR, WARNING = "error", "warning"


class InferenceError(ValueError):
    """El archivo no se puede leer o el tramo no se puede clasificar."""


@dataclass(frozen=True)
class Check:
    """Aviso sobre el audio de entrada."""

    level: str      # "error" impide clasificar; "warning" solo advierte
    message: str


@dataclass
class ModelBundle:
    """Modelo congelado listo para predecir."""

    key: str
    model: Any
    feature_names: list[str]
    metadata: dict[str, Any]
    verified: bool   # el archivo coincide con el sellado antes de abrir el test


@dataclass
class CycleAnalysis:
    """Todo lo que la interfaz necesita de un ciclo ya procesado."""

    audio: np.ndarray               # tramo original en mono, a su frecuencia nativa
    sample_rate: int
    processed: PreprocessingResult  # senal que "escucha" el modelo
    features: pd.DataFrame          # 1 fila x 225 columnas con nombre
    log_mel: np.ndarray             # (bandas, tramas) en dB, de la senal preprocesada
    hop_sec: float
    duration_sec: float
    elapsed_ms: float


@dataclass
class Prediction:
    model_key: str
    label: str
    probabilities: dict[str, float]
    probability_label: str                    # clase de mayor probabilidad (puede diferir de ``label``)
    votes: list[tuple[str, str]] | None       # (miembro, clase) si el modelo es un voto
    elapsed_ms: float


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Lee un WAV desde bytes y lo devuelve en mono (float32) con su frecuencia de muestreo.

    Raises:
        InferenceError: el archivo no es un audio legible, esta vacio o tiene valores no finitos.
    """
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (RuntimeError, sf.LibsndfileError, ValueError) as exc:
        raise InferenceError("No se pudo leer el archivo. Debe ser un audio WAV válido.") from exc
    if audio.size == 0:
        raise InferenceError("El archivo no contiene muestras de audio.")
    if not np.all(np.isfinite(audio)):
        raise InferenceError("El archivo contiene valores no válidos (NaN o infinito).")
    mono = audio if audio.ndim == 1 else to_mono(audio).astype(np.float32)
    return mono, int(sample_rate)


def trim(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Tramo ``[start_sec, end_sec)`` del audio (recorte manual que elige el usuario)."""
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(len(audio), int(round(end_sec * sample_rate)))
    if end <= start:
        raise InferenceError("El tramo elegido está vacío.")
    return audio[start:end]


def minimum_duration_sec(cfg: Config) -> float:
    """Duracion minima para que las caracteristicas se puedan calcular (limite tecnico)."""
    return frame_params(cfg.features, cfg.preprocessing.target_sr).min_samples / cfg.preprocessing.target_sr


def check_input(audio: np.ndarray, sample_rate: int, cfg: Config) -> list[Check]:
    """Revisa si el tramo se parece a un ciclo como los de entrenamiento. No modifica el audio."""
    app, checks = cfg.app, []
    duration = len(audio) / sample_rate

    def fmt(value: float) -> str:
        return f"{value:.1f}".replace(".", ",")

    if sample_rate < 2 * cfg.preprocessing.bandpass_high_hz:
        checks.append(Check(ERROR, f"La frecuencia de muestreo es de {sample_rate} Hz. Se necesitan al menos "
                                   f"{int(2 * cfg.preprocessing.bandpass_high_hz)} Hz para cubrir la banda de "
                                   f"{int(cfg.preprocessing.bandpass_low_hz)} a "
                                   f"{int(cfg.preprocessing.bandpass_high_hz)} Hz que usa el modelo."))
    if duration < minimum_duration_sec(cfg):
        checks.append(Check(ERROR, f"El tramo dura {duration:.2f} s y es demasiado corto para analizarlo.".replace(
            ".", ",", 1)))
    elif duration < app.min_duration_sec:
        checks.append(Check(WARNING, f"El tramo dura {duration:.2f} s, menos que el ciclo más corto con que se "
                                     f"entrenó el modelo ({fmt(app.min_duration_sec)} s).".replace(".", ",", 1)))
    if duration > app.max_duration_sec:
        checks.append(Check(WARNING, f"El tramo dura {fmt(duration)} s, más que cualquier ciclo de entrenamiento "
                                     f"({fmt(app.max_duration_sec)} s). Es probable que contenga varias "
                                     "respiraciones; el resultado no es confiable."))
    elif duration > app.typical_max_duration_sec:
        checks.append(Check(WARNING, f"El tramo dura {fmt(duration)} s. Casi todos los ciclos de entrenamiento duran "
                                     f"menos de {fmt(app.typical_max_duration_sec)} s; revise que sea una sola "
                                     "respiración."))
    if float(np.mean(np.abs(audio) >= cfg.audit.clipping_level)) > app.clipping_warning_fraction:
        checks.append(Check(WARNING, "La grabación está saturada: se grabó con el volumen tan alto que el sonido "
                                     "sale distorsionado, y eso puede cambiar el resultado."))
    if rms(audio) < cfg.preprocessing.silence_rms_threshold:
        checks.append(Check(ERROR, "El tramo está prácticamente en silencio."))
    return checks


def analyze_cycle(audio: np.ndarray, sample_rate: int, cfg: Config) -> CycleAnalysis:
    """Preprocesa y extrae las caracteristicas de un ciclo con la ruta de entrenamiento.

    Raises:
        InferenceError: el tramo no admite el preprocesamiento o la extraccion.
    """
    started = time.perf_counter()
    try:
        processed = preprocess_audio(audio, sample_rate, cfg.preprocessing)
        if processed.is_silent:
            raise InferenceError("No hay señal en la banda de 100 a 2 000 Hz: el tramo no se puede clasificar.")
        parts = extract_representations(processed.audio, cfg.features, processed.sample_rate)
        descriptors = compute_descriptors(processed.audio, processed.sample_rate, cfg.descriptors, cfg.features)
    except (PreprocessingError, FeatureExtractionError) as exc:
        raise InferenceError(f"No se pudo analizar el tramo: {exc}") from exc
    vector = np.concatenate([parts[name] for name in cfg.features.representations] + [descriptors])
    params = frame_params(cfg.features, processed.sample_rate)
    return CycleAnalysis(
        audio=audio, sample_rate=int(sample_rate), processed=processed,
        features=pd.DataFrame([vector], columns=feature_columns(cfg)),
        log_mel=log_mel_spectrogram(processed.audio, cfg.features, params),
        hop_sec=params.hop_length / processed.sample_rate,
        duration_sec=len(audio) / sample_rate,
        elapsed_ms=1000.0 * (time.perf_counter() - started),
    )


def load_models(cfg: Config) -> dict[str, ModelBundle]:
    """Carga los modelos congelados; si existe el sello de la Fase 7, comprueba que coincidan.

    Raises:
        InferenceError: no hay ningun modelo entrenado en ``models/``.
    """
    try:
        verified = {key: ModelBundle(key, model, names, meta, True)
                    for key, (model, names, meta) in load_frozen_models(cfg).items()}
    except (FileNotFoundError, EvaluationSetupError, KeyError):
        verified = {}
    if verified:
        return verified
    metadata_files = {"best_model": "metadatos_modelo.json", "prototype": "metadatos_modelo_prototipo.json"}
    bundles = {}
    for key, file_name in MODEL_FILES.items():
        path = cfg.paths.model_dir / file_name
        if path.is_file():
            meta = json.loads((cfg.paths.model_dir / metadata_files[key]).read_text(encoding="utf-8"))
            bundles[key] = ModelBundle(key, joblib.load(path), list(meta["feature_names"]), meta, False)
    if not bundles:
        raise InferenceError("No hay modelos entrenados en la carpeta models/. Ejecute antes la Fase 7.")
    return bundles


def predict(bundle: ModelBundle, analysis: CycleAnalysis) -> Prediction:
    """Aplica un modelo congelado a un ciclo ya analizado."""
    classes = list(bundle.metadata["classes"])
    X = analysis.features[bundle.feature_names]
    started = time.perf_counter()
    code = int(bundle.model.predict(X)[0])
    proba = np.asarray(bundle.model.predict_proba(X)[0], dtype=float)
    votes = None
    if hasattr(bundle.model, "member_predictions"):
        member_codes = bundle.model.member_predictions(X)[0]
        votes = [(MEMBER_NAMES.get(name, name), classes[int(c)])
                 for (name, _), c in zip(bundle.model.fitted_, member_codes)]
    return Prediction(
        model_key=bundle.key, label=classes[code],
        probabilities={name: float(p) for name, p in zip(classes, proba)},
        probability_label=classes[int(np.argmax(proba))], votes=votes,
        elapsed_ms=1000.0 * (time.perf_counter() - started),
    )
