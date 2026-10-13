"""Extraccion de las 4 representaciones acusticas como vectores de dimension fija (Fase 4).

Cada ciclo se lee y se preprocesa una sola vez (:func:`src.preprocesamiento.preprocess_audio`)
y de la senal resultante (mono, TARGET_SR, 100-2000 Hz, RMS = 0,1) se obtienen:

========  ==============================================================  =====
Nombre    Vector                                                          Dim.
========  ==============================================================  =====
mfcc      media y desviacion temporal de c0..c12                          26
mel       media y desviacion temporal de cada banda del log-Mel (dB)      128
wavelet   por sub-banda A7, D7..D1: media, desviacion, energia relativa,   32
          entropia normalizada
rms       mean, std, median, min, max, p25, p75 del RMS por trama          7
========  ==============================================================  =====

Las matrices tiempo-frecuencia se calculan con ``center=False`` (sin relleno): toda
trama contiene solo senal real. La MFCC es la DCT-II ortonormal del mismo log-Mel
usado para ``mel``: el STFT se calcula una sola vez.

Formulas de la representacion wavelet (``c`` = coeficientes de la sub-banda ``j``,
``N_j`` = numero de coeficientes)::

    media_j    = (1/N_j) * sum_k c_k
    desv_j     = sqrt((1/N_j) * sum_k (c_k - media_j)^2)
    E_j        = sum_k c_k^2 ;  energia_rel_j = E_j / sum_i E_i
    q_k        = c_k^2 / E_j ;  entropia_j = -sum_k q_k log2 q_k / log2(N_j)   (0 log 0 = 0)

La energia relativa y la entropia normalizada no dependen de la duracion del ciclo.
La extension de bordes es ``zero`` (ver ``config.yaml``): con ``symmetric`` los
artefactos de borde dominaban la energia de las sub-bandas gruesas (A7, D7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import pywt
from joblib import Parallel, delayed
from tqdm import tqdm

from src.configuracion import Config, FeatureConfig, PreprocessingConfig, seconds_to_samples
from src.preprocesamiento import PreprocessingError, preprocess_audio
from src.segmentacion import SegmentationError, iter_cycle_segments

RMS_STATS: tuple[str, ...] = ("mean", "std", "median", "min", "max", "p25", "p75")
WAVELET_STATS: tuple[str, ...] = ("mean", "std", "relenergy", "entropy")
INDEX_COLUMNS: tuple[str, ...] = (
    "cycle_id", "patient_id", "recording_id", "label", "crackle", "wheeze", "equipment",
    "input_sample_rate", "duration_sec", "n_samples_processed", "inband_rms",
)


class FeatureExtractionError(ValueError):
    """Un ciclo no puede producir el vector de caracteristicas completo."""


class StaleFeaturesError(RuntimeError):
    """La cache de caracteristicas no corresponde a la configuracion actual."""


# Parametros derivados


@dataclass(frozen=True)
class FrameParams:
    """Tamanos de trama en muestras a la frecuencia de trabajo."""

    sample_rate: int
    n_fft: int
    hop_length: int
    rms_frame: int
    rms_hop: int
    wavelet_min_samples: int

    @property
    def min_samples(self) -> int:
        """Longitud minima (muestras) para producir las 4 representaciones completas."""
        return max(self.n_fft, self.rms_frame, self.wavelet_min_samples)


def frame_params(fcfg: FeatureConfig, sample_rate: int) -> FrameParams:
    """Convierte las tramas de segundos a muestras para ``sample_rate``."""
    dec_len = pywt.Wavelet(fcfg.wavelet_name).dec_len
    return FrameParams(
        sample_rate=sample_rate,
        n_fft=seconds_to_samples(fcfg.stft_window_sec, sample_rate),
        hop_length=seconds_to_samples(fcfg.stft_hop_sec, sample_rate),
        rms_frame=seconds_to_samples(fcfg.rms_frame_sec, sample_rate),
        rms_hop=seconds_to_samples(fcfg.rms_hop_sec, sample_rate),
        # pywt.dwt_max_level(n, dec_len) >= L  <=>  n >= (dec_len - 1) * 2**L
        wavelet_min_samples=(dec_len - 1) * 2**fcfg.wavelet_level,
    )


def wavelet_subband_names(level: int) -> list[str]:
    """Nombres de sub-bandas en el orden de ``pywt.wavedec``: A{L}, D{L}, ..., D1."""
    return [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]


def wavelet_subband_ranges(level: int, sample_rate: int) -> dict[str, tuple[float, float]]:
    """Banda de frecuencia nominal (Hz) de cada sub-banda de la DWT."""
    ranges = {f"A{level}": (0.0, sample_rate / 2 ** (level + 1))}
    for j in range(level, 0, -1):
        ranges[f"D{j}"] = (sample_rate / 2 ** (j + 1), sample_rate / 2**j)
    return ranges


def feature_names(fcfg: FeatureConfig) -> dict[str, list[str]]:
    """Nombres de las columnas de cada representacion (orden fijo)."""
    names = {
        "mfcc": [f"mfcc_{stat}_{i:02d}" for stat in ("mean", "std") for i in range(fcfg.n_mfcc)],
        "mel": [f"mel_{stat}_{i:03d}" for stat in ("mean", "std") for i in range(fcfg.n_mels)],
        "wavelet": [f"dwt_{band}_{stat}" for band in wavelet_subband_names(fcfg.wavelet_level)
                    for stat in WAVELET_STATS],
        "rms": [f"rms_{stat}" for stat in RMS_STATS],
    }
    return {name: names[name] for name in fcfg.representations}


# Representaciones


@lru_cache(maxsize=8)
def _mel_basis(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float, htk: bool,
               norm: str) -> np.ndarray:
    """Banco de filtros Mel (cacheado, solo lectura)."""
    basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk,
                                norm=norm, dtype=np.float64)
    basis.setflags(write=False)
    return basis


def log_mel_spectrogram(audio: np.ndarray, fcfg: FeatureConfig, params: FrameParams) -> np.ndarray:
    """Espectrograma log-Mel (dB, ref = 1) de forma ``(n_mels, n_tramas)``."""
    spectrum = librosa.stft(audio, n_fft=params.n_fft, hop_length=params.hop_length, win_length=params.n_fft,
                            window=fcfg.stft_window, center=False)
    power = np.abs(spectrum) ** fcfg.mel_power
    basis = _mel_basis(params.sample_rate, params.n_fft, fcfg.n_mels, fcfg.mel_fmin_hz, fcfg.mel_fmax_hz,
                       fcfg.mel_htk, fcfg.mel_norm)
    return librosa.power_to_db(basis @ power, ref=1.0, amin=fcfg.mel_amin, top_db=None)


def mfcc_matrix(log_mel: np.ndarray, fcfg: FeatureConfig) -> np.ndarray:
    """MFCC ``(n_mfcc, n_tramas)``: DCT-II ortonormal del log-Mel, sin lifter."""
    return librosa.feature.mfcc(S=log_mel, n_mfcc=fcfg.n_mfcc, dct_type=2, norm="ortho", lifter=0)


def temporal_mean_std(matrix: np.ndarray) -> np.ndarray:
    """Concatena la media y la desviacion (ddof=0) de cada fila a lo largo del tiempo."""
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def wavelet_features(audio: np.ndarray, fcfg: FeatureConfig) -> np.ndarray:
    """Vector DWT de dimension fija (ver formulas en el docstring del modulo).

    Raises:
        FeatureExtractionError: si la senal no admite el nivel configurado
            (nunca se reduce el nivel ni se rellena en silencio).
    """
    wavelet = pywt.Wavelet(fcfg.wavelet_name)
    max_level = pywt.dwt_max_level(len(audio), wavelet.dec_len)
    if max_level < fcfg.wavelet_level:
        raise FeatureExtractionError(
            f"DWT {fcfg.wavelet_name} nivel {fcfg.wavelet_level} requiere >= "
            f"{(wavelet.dec_len - 1) * 2 ** fcfg.wavelet_level} muestras; el ciclo tiene {len(audio)} "
            f"(nivel maximo posible = {max_level})."
        )
    coefficients = pywt.wavedec(audio, wavelet, mode=fcfg.wavelet_mode, level=fcfg.wavelet_level)
    energies = np.array([float(np.sum(c**2)) for c in coefficients])
    total = float(energies.sum())
    values: list[float] = []
    for c, energy in zip(coefficients, energies):
        if energy > 0 and c.size > 1:
            q = c**2 / energy
            q = q[q > 0]
            entropy = float(-np.sum(q * np.log2(q)) / np.log2(c.size))
        else:
            entropy = 0.0
        values.extend([float(np.mean(c)), float(np.std(c)), energy / total if total > 0 else 0.0, entropy])
    return np.asarray(values)


def rms_contour(audio: np.ndarray, params: FrameParams) -> np.ndarray:
    """RMS por trama (sin ventana, sin relleno)."""
    return librosa.feature.rms(y=audio, frame_length=params.rms_frame, hop_length=params.rms_hop,
                               center=False, dtype=np.float64)[0]


def rms_features(audio: np.ndarray, params: FrameParams) -> np.ndarray:
    """Estadisticos del RMS por trama en el orden de :data:`RMS_STATS`."""
    r = rms_contour(audio, params)
    p25, median, p75 = np.percentile(r, [25, 50, 75])
    return np.array([r.mean(), r.std(), median, r.min(), r.max(), p25, p75])


def extract_representations(audio: np.ndarray, fcfg: FeatureConfig, sample_rate: int) -> dict[str, np.ndarray]:
    """Calcula las representaciones configuradas a partir de una senal ya preprocesada.

    Raises:
        FeatureExtractionError: senal mas corta que el minimo o resultado no finito.
    """
    params = frame_params(fcfg, sample_rate)
    if len(audio) < params.min_samples:
        raise FeatureExtractionError(
            f"Ciclo de {len(audio)} muestras ({len(audio) / sample_rate:.3f} s); se requieren >= "
            f"{params.min_samples} ({params.min_samples / sample_rate:.3f} s) a {sample_rate} Hz."
        )
    wanted = set(fcfg.representations)
    out: dict[str, np.ndarray] = {}
    if wanted & {"mfcc", "mel"}:
        log_mel = log_mel_spectrogram(audio, fcfg, params)
        if "mel" in wanted:
            out["mel"] = temporal_mean_std(log_mel)
        if "mfcc" in wanted:
            out["mfcc"] = temporal_mean_std(mfcc_matrix(log_mel, fcfg))
    if "wavelet" in wanted:
        out["wavelet"] = wavelet_features(audio, fcfg)
    if "rms" in wanted:
        out["rms"] = rms_features(audio, params)
    for name, vector in out.items():
        if not np.all(np.isfinite(vector)):
            raise FeatureExtractionError(f"La representacion '{name}' produjo NaN o Inf.")
    return {name: out[name] for name in fcfg.representations}


def extract_features_from_audio(
    audio: np.ndarray, sample_rate: int, cfg: Config
) -> tuple[dict[str, np.ndarray], Any]:
    """Punto de entrada unico para audio nuevo (interfaz, validacion externa, robustez).

    Aplica exactamente el preprocesamiento de entrenamiento y despues extrae las
    representaciones. Devuelve ``(caracteristicas, resultado_del_preprocesamiento)``.
    """
    processed = preprocess_audio(audio, sample_rate, cfg.preprocessing)
    return extract_representations(processed.audio, cfg.features, processed.sample_rate), processed


# Extraccion masiva (paralela por grabacion)


Extractor = Callable[[np.ndarray, int, Config], tuple[dict[str, np.ndarray], Any]]


def _process_recording(group: pd.DataFrame, dataset_dir: Path, cfg: Config,
                       extractor: Extractor | None = None) -> list[dict[str, Any]]:
    """Trabajador: lee una grabacion, preprocesa y extrae todos sus ciclos."""
    extract = extractor or extract_features_from_audio
    results: list[dict[str, Any]] = []
    try:
        segments = list(iter_cycle_segments(group, dataset_dir, show_progress=False))
    except (SegmentationError, OSError, RuntimeError) as exc:
        return [{"cycle_id": cid, "status": "error", "status_detail": f"segmentacion: {exc}"}
                for cid in group["cycle_id"]]
    for segment in segments:
        try:
            features, processed = extract(segment.audio, segment.sample_rate, cfg)
        except (PreprocessingError, FeatureExtractionError) as exc:
            results.append({"cycle_id": segment.cycle_id, "status": "error", "status_detail": str(exc)})
            continue
        results.append({
            "cycle_id": segment.cycle_id,
            "status": "ok",
            "status_detail": "",
            "n_samples_processed": int(processed.audio.shape[0]),
            "inband_rms": processed.inband_rms,
            "features": features,
        })
    return results


@dataclass
class FeatureSet:
    """Indice de ciclos y una matriz por representacion (mismas filas, mismo orden)."""

    index: pd.DataFrame
    matrices: dict[str, pd.DataFrame]
    failures: pd.DataFrame


def extract_dataset_features(
    cycles: pd.DataFrame, dataset_dir: Path, cfg: Config, n_jobs: int | None = None, show_progress: bool = True,
    extractor: Extractor | None = None, names: dict[str, list[str]] | None = None,
) -> FeatureSet:
    """Extrae las representaciones de todos los ciclos (una lectura por grabacion).

    ``extractor`` y ``names`` permiten reutilizar esta misma rutina con otros
    vectores (p. ej. los descriptores de la mejora propuesta); por defecto se
    extraen las 4 representaciones del experimento principal.

    Un ciclo que falla en cualquier representacion se excluye de todas (mismo conjunto
    de muestras para comparar representaciones) y queda registrado en ``failures``.
    Filas ordenadas por ``cycle_id`` (determinista, independiente del paralelismo).
    """
    jobs = cfg.features.n_jobs if n_jobs is None else n_jobs
    groups = [group for _, group in cycles.groupby("source_audio", sort=True)]
    runner = Parallel(n_jobs=jobs, return_as="generator_unordered")
    results: list[dict[str, Any]] = []
    for batch in tqdm(runner(delayed(_process_recording)(g, dataset_dir, cfg, extractor) for g in groups),
                      total=len(groups), desc="Extrayendo caracteristicas", disable=not show_progress):
        results.extend(batch)

    ok = sorted((r for r in results if r["status"] == "ok"), key=lambda r: r["cycle_id"])
    failures = pd.DataFrame(
        [{"cycle_id": r["cycle_id"], "status_detail": r["status_detail"]} for r in results if r["status"] != "ok"],
        columns=["cycle_id", "status_detail"],
    ).sort_values("cycle_id", ignore_index=True)

    meta = cycles.set_index("cycle_id")
    ok_ids = [r["cycle_id"] for r in ok]
    index = pd.DataFrame({
        "cycle_id": ok_ids,
        "patient_id": meta.loc[ok_ids, "patient_id"].astype(str).to_numpy(),
        "recording_id": meta.loc[ok_ids, "recording_id"].to_numpy(),
        "label": meta.loc[ok_ids, "label"].to_numpy(),
        "crackle": meta.loc[ok_ids, "crackle"].astype(int).to_numpy(),
        "wheeze": meta.loc[ok_ids, "wheeze"].astype(int).to_numpy(),
        "equipment": meta.loc[ok_ids, "equipment"].to_numpy(),
        "input_sample_rate": meta.loc[ok_ids, "sample_rate"].astype(int).to_numpy(),
        "duration_sec": meta.loc[ok_ids, "duration_sec"].astype(float).to_numpy(),
        "n_samples_processed": [r["n_samples_processed"] for r in ok],
        "inband_rms": [r["inband_rms"] for r in ok],
    })
    names = names or feature_names(cfg.features)
    matrices = {
        name: pd.DataFrame(np.vstack([r["features"][name] for r in ok]) if ok else np.empty((0, len(cols))),
                           columns=cols)
        for name, cols in names.items()
    }
    return FeatureSet(index=index, matrices=matrices, failures=failures)


# Cache en disco


def config_fingerprint(preprocessing: PreprocessingConfig, features: FeatureConfig) -> str:
    """Huella SHA-256 de los parametros que determinan las caracteristicas.

    ``n_jobs`` se excluye: no cambia el resultado, solo la velocidad.
    """
    feature_params = {k: v for k, v in features.as_dict().items() if k != "n_jobs"}
    payload = json.dumps({"preprocessing": preprocessing.as_dict(), "features": feature_params},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_feature_set(feature_set: FeatureSet, features_dir: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    """Guarda indice y matrices en Parquet, y los metadatos (con la huella) en JSON."""
    features_dir.mkdir(parents=True, exist_ok=True)
    paths = {"index": features_dir / "indice.parquet"}
    feature_set.index.to_parquet(paths["index"], index=False)
    for name, matrix in feature_set.matrices.items():
        paths[name] = features_dir / f"{name}.parquet"
        table = pd.concat([feature_set.index[["cycle_id"]].reset_index(drop=True), matrix.reset_index(drop=True)],
                          axis=1)
        table.to_parquet(paths[name], index=False)
    paths["metadata"] = features_dir / "metadatos_caracteristicas.json"
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
    return paths


def load_features(cfg: Config, representation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga ``(indice, X)`` de una representacion verificando que la cache este vigente.

    Raises:
        FileNotFoundError: si no se ha ejecutado la extraccion.
        StaleFeaturesError: si la cache se genero con otra configuracion o las filas
            no coinciden con el indice.
    """
    features_dir = cfg.paths.features_dir
    meta_path = features_dir / "metadatos_caracteristicas.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No existe {meta_path}. Ejecute: python scripts/extraer_caracteristicas.py")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = config_fingerprint(cfg.preprocessing, cfg.features)
    if metadata.get("config_fingerprint") != expected:
        raise StaleFeaturesError(
            "La cache de caracteristicas no corresponde a config.yaml actual. "
            "Vuelva a ejecutar: python scripts/extraer_caracteristicas.py"
        )
    index = pd.read_parquet(features_dir / "indice.parquet")
    matrix = pd.read_parquet(features_dir / f"{representation}.parquet")
    if not matrix["cycle_id"].equals(index["cycle_id"]):
        raise StaleFeaturesError(f"Las filas de {representation}.parquet no coinciden con indice.parquet.")
    expected_columns = feature_names(cfg.features)[representation]
    if list(matrix.columns[1:]) != expected_columns:
        raise StaleFeaturesError(f"Columnas inesperadas en {representation}.parquet.")
    return index, matrix.drop(columns="cycle_id")


# Validacion


def validate_matrix(
    name: str, matrix: pd.DataFrame, expected_columns: list[str], n_expected_rows: int, n_failed: int,
    constant_std_threshold: float,
) -> dict[str, Any]:
    """Controles de la Fase 4.5 para una representacion."""
    values = matrix.to_numpy(dtype=float)
    std = values.std(axis=0) if len(values) else np.zeros(values.shape[1])
    constant = [col for col, s in zip(matrix.columns, std) if s < constant_std_threshold]
    return {
        "representation": name,
        "n_samples": int(values.shape[0]),
        "n_samples_expected": n_expected_rows,
        "n_features": int(values.shape[1]),
        "n_features_expected": len(expected_columns),
        "columns_match": list(matrix.columns) == expected_columns,
        "n_nan": int(np.isnan(values).sum()),
        "n_inf": int(np.isinf(values).sum()),
        "n_constant_features": len(constant),
        "constant_features": "; ".join(constant),
        "n_failed_extractions": n_failed,
    }


def feature_statistics(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Estadisticos descriptivos por caracteristica (sin etiquetas)."""
    rows = []
    for name, matrix in matrices.items():
        for column in matrix.columns:
            v = matrix[column].to_numpy(dtype=float)
            rows.append({"representation": name, "feature": column, "mean": v.mean(), "std": v.std(),
                         "min": v.min(), "max": v.max(), "n_unique": int(np.unique(v).size)})
    return pd.DataFrame(rows)
