"""Descriptores acusticos especificos de sibilancias y crepitantes (MEJORA PROPUESTA).

No forman parte del experimento principal (las 4 representaciones). Se calculan
sobre la misma senal preprocesada (``preprocess_audio``) y con el mismo STFT, y se
guardan en una cache propia (``data/features/descriptors.parquet``) con su huella.

Idea fisica (umbrales fijados a priori en ``config.yaml``, seccion ``descriptors``):

* **Sibilancia = tono sostenido.** En cada trama se mide la prominencia local del pico
  espectral mas alto: ``P_dB(f) - mediana_movil_en_frecuencia(P_dB)(f)``. Una trama
  es "tonal" si la prominencia maxima supera el umbral; se mide ademas la racha mas
  larga de tramas tonales consecutivas cuya frecuencia dominante casi no cambia.
* **Crepitante = transitorio breve.** Se calcula la envolvente RMS en tramas de 5 ms
  y se cuentan los maximos locales que superan ``mediana + k * MAD`` (MAD robusta).

El resto son descriptores estandar de forma espectral y temporal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from src.configuracion import Config, DescriptorConfig, FeatureConfig, seconds_to_samples
from src.extraccion_caracteristicas import FeatureExtractionError, StaleFeaturesError, frame_params
from src.preprocesamiento import preprocess_audio

EPS = 1e-12
DESCRIPTORS = "descriptors"
METADATA_NAME = "metadatos_descriptores.json"


def descriptor_names(dcfg: DescriptorConfig) -> list[str]:
    """Nombres de las columnas, en el orden de :func:`compute_descriptors`."""
    bands = [f"band_fraction_{int(lo)}_{int(hi)}" for lo, hi in zip(dcfg.band_edges_hz[:-1], dcfg.band_edges_hz[1:])]
    return [
        "tonal_prom_mean", "tonal_prom_p90", "tonal_prom_max", "tonal_frame_fraction", "tonal_run_sec",
        "tonal_run_fraction", "dom_freq_median", "dom_freq_std", "flatness_mean", "flatness_p10",
        "centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean", "entropy_mean", "entropy_std",
        *bands,
        "wave_kurtosis", "wave_skewness", "crest_factor_db", "zcr_mean", "zcr_std",
        "peak_rate_per_sec", "peak_height_mean_db", "peak_height_max_db",
        "env_kurtosis", "env_cv", "env_p95_median_db", "log_duration",
    ]


def longest_stable_run(tonal: np.ndarray, dominant_bin: np.ndarray, tolerance_bins: int) -> int:
    """Racha mas larga de tramas tonales consecutivas con frecuencia dominante estable."""
    best = run = 0
    for t in range(len(tonal)):
        if tonal[t]:
            stable = run > 0 and abs(int(dominant_bin[t]) - int(dominant_bin[t - 1])) <= tolerance_bins
            run = run + 1 if stable else 1
        else:
            run = 0
        best = max(best, run)
    return best


def _safe_moment(values: np.ndarray, function: Any) -> float:
    """Curtosis/asimetria con resguardo para senales constantes."""
    return float(function(values)) if float(np.std(values)) > EPS else 0.0


def compute_descriptors(audio: np.ndarray, sample_rate: int, dcfg: DescriptorConfig,
                        fcfg: FeatureConfig) -> np.ndarray:
    """Vector de descriptores de un ciclo ya preprocesado (orden de :func:`descriptor_names`).

    Raises:
        FeatureExtractionError: ciclo mas corto que una ventana STFT o resultado no finito.
    """
    params = frame_params(fcfg, sample_rate)
    if len(audio) < params.n_fft:
        raise FeatureExtractionError(f"Ciclo de {len(audio)} muestras; se requieren >= {params.n_fft}.")
    duration = len(audio) / sample_rate
    hop_sec = params.hop_length / sample_rate

    # espectro en la banda de interes
    power = np.abs(librosa.stft(audio, n_fft=params.n_fft, hop_length=params.hop_length, win_length=params.n_fft,
                                window=fcfg.stft_window, center=False)) ** 2
    freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate)
    in_band = (freqs >= dcfg.band_edges_hz[0]) & (freqs <= dcfg.band_edges_hz[-1])
    P, f = power[in_band], freqs[in_band]
    total = P.sum(axis=0)
    p = np.where(total > 0, P / np.where(total > 0, total, 1.0), 1.0 / len(f))

    flatness = np.exp(np.mean(np.log(P + EPS), axis=0)) / (np.mean(P, axis=0) + EPS)
    entropy = -np.sum(p * np.log2(p + EPS), axis=0) / np.log2(len(f))
    centroid = np.sum(f[:, None] * p, axis=0)
    bandwidth = np.sqrt(np.sum((f[:, None] - centroid) ** 2 * p, axis=0))
    rolloff = f[np.argmax(np.cumsum(p, axis=0) >= dcfg.rolloff_fraction, axis=0)]

    # tonalidad (sibilancias)
    p_db = 10.0 * np.log10(P + EPS)
    prominence = p_db - median_filter(p_db, size=(dcfg.tonal_median_filter_bins, 1), mode="nearest")
    tonal_prom = prominence.max(axis=0)
    dominant_bin = prominence.argmax(axis=0)
    tonal = tonal_prom >= dcfg.tonal_prominence_db
    run_sec = longest_stable_run(tonal, dominant_bin, dcfg.tonal_freq_tolerance_bins) * hop_sec
    dom_freq = f[dominant_bin]

    # energia por bandas
    mean_spectrum = P.mean(axis=1)
    spectrum_total = float(mean_spectrum.sum())
    n_bands = len(dcfg.band_edges_hz) - 1
    fractions = []
    for i, (low, high) in enumerate(zip(dcfg.band_edges_hz[:-1], dcfg.band_edges_hz[1:])):
        mask = (f >= low) & ((f <= high) if i == n_bands - 1 else (f < high))
        fractions.append(float(mean_spectrum[mask].sum() / spectrum_total) if spectrum_total > 0 else 1.0 / n_bands)

    # forma temporal y transitorios (crepitantes)
    level = float(np.sqrt(np.mean(audio**2)))
    crest_db = 20.0 * np.log10((float(np.max(np.abs(audio))) + EPS) / (level + EPS))
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=seconds_to_samples(dcfg.zcr_frame_sec, sample_rate),
        hop_length=seconds_to_samples(dcfg.zcr_hop_sec, sample_rate), center=False)[0]
    env_hop = seconds_to_samples(dcfg.envelope_hop_sec, sample_rate)
    envelope = librosa.feature.rms(y=audio, frame_length=seconds_to_samples(dcfg.envelope_frame_sec, sample_rate),
                                   hop_length=env_hop, center=False, dtype=np.float64)[0]
    median = float(np.median(envelope))
    mad = 1.4826 * float(np.median(np.abs(envelope - median)))
    peak_rate, peak_mean_db, peak_max_db = 0.0, 0.0, 0.0
    if mad > EPS and len(envelope) >= 3:
        distance = max(1, int(round(dcfg.peak_min_distance_sec / (env_hop / sample_rate))))
        peaks, _ = find_peaks(envelope, height=median + dcfg.peak_mad_factor * mad, distance=distance)
        if len(peaks):
            heights_db = 20.0 * np.log10(envelope[peaks] / (median + EPS))
            peak_rate = len(peaks) / duration
            peak_mean_db, peak_max_db = float(heights_db.mean()), float(heights_db.max())
    env_mean = float(envelope.mean())

    values = np.array([
        tonal_prom.mean(), np.percentile(tonal_prom, 90), tonal_prom.max(), tonal.mean(), run_sec,
        min(1.0, run_sec / duration), np.median(dom_freq), dom_freq.std(), flatness.mean(),
        np.percentile(flatness, 10), centroid.mean(), centroid.std(), bandwidth.mean(), rolloff.mean(),
        entropy.mean(), entropy.std(),
        *fractions,
        _safe_moment(audio, kurtosis), _safe_moment(audio, skew), crest_db, zcr.mean(), zcr.std(),
        peak_rate, peak_mean_db, peak_max_db,
        _safe_moment(envelope, kurtosis), float(envelope.std()) / (env_mean + EPS),
        20.0 * np.log10((float(np.percentile(envelope, 95)) + EPS) / (median + EPS)), np.log(duration),
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FeatureExtractionError("Los descriptores produjeron NaN o Inf.")
    return values


def extract_descriptors_from_audio(audio: np.ndarray, sample_rate: int, cfg: Config) -> tuple[dict[str, np.ndarray], Any]:
    """Preprocesa (misma cadena de entrenamiento) y calcula los descriptores de un ciclo."""
    processed = preprocess_audio(audio, sample_rate, cfg.preprocessing)
    vector = compute_descriptors(processed.audio, processed.sample_rate, cfg.descriptors, cfg.features)
    return {DESCRIPTORS: vector}, processed


def descriptor_fingerprint(cfg: Config) -> str:
    """Huella SHA-256 de todo lo que determina los descriptores."""
    payload = json.dumps({
        "preprocessing": cfg.preprocessing.as_dict(),
        "stft": [cfg.features.stft_window_sec, cfg.features.stft_hop_sec, cfg.features.stft_window],
        "descriptors": cfg.descriptors.as_dict(),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_descriptors(index: pd.DataFrame, matrix: pd.DataFrame, features_dir: Path, metadata: dict[str, Any]) -> Path:
    """Guarda ``descriptors.parquet`` (cycle_id + columnas) y sus metadatos."""
    main_index = pd.read_parquet(features_dir / "indice.parquet", columns=["cycle_id"])
    if not main_index["cycle_id"].equals(index["cycle_id"]):
        raise StaleFeaturesError("Los descriptores no cubren las mismas filas que la cache principal.")
    path = features_dir / f"{DESCRIPTORS}.parquet"
    pd.concat([index[["cycle_id"]].reset_index(drop=True), matrix.reset_index(drop=True)], axis=1).to_parquet(
        path, index=False)
    (features_dir / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def load_descriptors(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga ``(indice, X)`` de los descriptores verificando huella, filas y columnas."""
    features_dir = cfg.paths.features_dir
    meta_path = features_dir / METADATA_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"No existe {meta_path}. Ejecute: python scripts/extraer_descriptores.py")
    if json.loads(meta_path.read_text(encoding="utf-8")).get("descriptor_fingerprint") != descriptor_fingerprint(cfg):
        raise StaleFeaturesError("La cache de descriptores no corresponde a config.yaml actual. "
                                 "Vuelva a ejecutar: python scripts/extraer_descriptores.py")
    index = pd.read_parquet(features_dir / "indice.parquet")
    matrix = pd.read_parquet(features_dir / f"{DESCRIPTORS}.parquet")
    if not matrix["cycle_id"].equals(index["cycle_id"]):
        raise StaleFeaturesError("Las filas de descriptors.parquet no coinciden con indice.parquet.")
    if list(matrix.columns[1:]) != descriptor_names(cfg.descriptors):
        raise StaleFeaturesError("Columnas inesperadas en descriptors.parquet.")
    return index, matrix.drop(columns="cycle_id")
