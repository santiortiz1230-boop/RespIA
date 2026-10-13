"""Segmentacion de ciclos respiratorios a partir de las anotaciones originales (Fase 2).

Decision de diseno: extraccion **bajo demanda** (sin copiar segmentos a disco).

* Cada ciclo se define de forma exacta y reproducible por ``source_audio`` y el
  intervalo semiabierto de muestras ``[round(inicio*sr), round(fin*sr))`` a la
  frecuencia de muestreo nativa. Aqui no se remuestrea ni se filtra: eso ocurre en
  la funcion unica de preprocesamiento (Fase 3).
* Acceso masivo (:func:`iter_cycle_segments`): lee cada WAV una sola vez y recorta
  todos sus ciclos; en memoria hay una sola grabacion a la vez.
* Acceso puntual (:func:`load_cycle_segment`): lee solo el tramo del ciclo.
* Los WAV originales se abren en modo lectura; nunca se escriben.

Los errores (intervalo vacio, fuera de rango, sample rate distinto al de los
metadatos) lanzan :class:`SegmentationError`: no hay fallos silenciosos.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.configuracion import AuditConfig, SegmentationConfig
from src.carga_datos import ERROR, ISSUE_COLUMNS, WARNING, make_issue, signal_stats

REQUIRED_CYCLE_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "recording_id",
    "patient_id",
    "source_audio",
    "start_sec",
    "end_sec",
    "duration_sec",
    "label",
    "sample_rate",
    "equipment",
    "is_valid",
)

STATUS_OK = "ok"
STATUS_ERROR = "error"


class SegmentationError(ValueError):
    """Ciclo imposible de extraer; ``category`` identifica la causa."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CycleSegment:
    """Un ciclo respiratorio extraido a la frecuencia nativa, sin procesar."""

    cycle_id: str
    recording_id: str
    patient_id: str
    label: str
    sample_rate: int
    start_sample: int
    end_sample: int
    audio: np.ndarray  # float32, escala [-1, 1); 1-D si el WAV es mono

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.sample_rate


# Conversion tiempo -> muestras y validaciones


def seconds_to_sample(time_sec: float, sample_rate: int) -> int:
    """Indice de muestra mas cercano a ``time_sec`` (redondeo con mitades hacia arriba)."""
    return int(math.floor(time_sec * sample_rate + 0.5))


def cycle_sample_bounds(start_sec: float, end_sec: float, sample_rate: int) -> tuple[int, int]:
    """Limites ``[inicio, fin)`` en muestras de un ciclo anotado en segundos."""
    return seconds_to_sample(start_sec, sample_rate), seconds_to_sample(end_sec, sample_rate)


def check_bounds(start_sample: int, end_sample: int, n_frames: int) -> None:
    """Verifica que el intervalo exista dentro del audio y no este vacio.

    Raises:
        SegmentationError: ``segmento_fuera_de_rango`` o ``segmento_vacio``.
    """
    if start_sample < 0 or end_sample > n_frames:
        raise SegmentationError(
            "segmento_fuera_de_rango", f"[{start_sample}, {end_sample}) fuera de un audio de {n_frames} muestras"
        )
    if end_sample <= start_sample:
        raise SegmentationError("segmento_vacio", f"intervalo vacio [{start_sample}, {end_sample})")


def check_sample_rate(file_rate: int, metadata_rate: int, cycle_id: str) -> None:
    """Verifica que el WAV tenga el sample rate registrado en los metadatos.

    Raises:
        SegmentationError: ``sample_rate_distinto`` (metadatos desactualizados).
    """
    if int(file_rate) != int(metadata_rate):
        raise SegmentationError(
            "sample_rate_distinto", f"{cycle_id}: archivo={file_rate} Hz, metadatos={metadata_rate} Hz"
        )


def validate_cycle_table(cycles: pd.DataFrame) -> None:
    """Comprueba que la tabla de ciclos tenga las columnas necesarias."""
    missing = [c for c in REQUIRED_CYCLE_COLUMNS if c not in cycles.columns]
    if missing:
        raise ValueError(f"Faltan columnas en la tabla de ciclos: {missing}")
    if not cycles["cycle_id"].is_unique:
        raise ValueError("cycle_id no es unico en la tabla de ciclos.")


# Lectura


def read_recording(path: Path) -> tuple[np.ndarray, int]:
    """Lee un WAV completo en float32 (solo lectura). Devuelve ``(audio, sample_rate)``."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sample_rate)


def _segment_from_audio(row: Any, audio: np.ndarray, sample_rate: int) -> CycleSegment:
    """Recorta un ciclo de una grabacion ya cargada (copia: no retiene la grabacion)."""
    check_sample_rate(sample_rate, row.sample_rate, row.cycle_id)
    start, end = cycle_sample_bounds(row.start_sec, row.end_sec, sample_rate)
    check_bounds(start, end, audio.shape[0])
    return CycleSegment(
        cycle_id=row.cycle_id,
        recording_id=row.recording_id,
        patient_id=str(row.patient_id),
        label=row.label,
        sample_rate=sample_rate,
        start_sample=start,
        end_sample=end,
        audio=audio[start:end].copy(),
    )


def load_cycle_segment(row: Any, dataset_dir: Path) -> CycleSegment:
    """Extrae un ciclo leyendo del disco solo su tramo (acceso puntual).

    Args:
        row: fila de ``cycle_metadata`` (``pd.Series`` o namedtuple).
        dataset_dir: carpeta raiz del dataset (``source_audio`` es relativo a ella).
    """
    path = Path(dataset_dir) / row.source_audio
    info = sf.info(str(path))
    check_sample_rate(info.samplerate, row.sample_rate, row.cycle_id)
    start, end = cycle_sample_bounds(row.start_sec, row.end_sec, int(info.samplerate))
    check_bounds(start, end, int(info.frames))
    audio, _ = sf.read(str(path), start=start, stop=end, dtype="float32", always_2d=False)
    if audio.shape[0] != end - start:
        raise SegmentationError("lectura_incompleta", f"{row.cycle_id}: se leyeron {audio.shape[0]} de {end - start}")
    return CycleSegment(
        cycle_id=row.cycle_id,
        recording_id=row.recording_id,
        patient_id=str(row.patient_id),
        label=row.label,
        sample_rate=int(info.samplerate),
        start_sample=start,
        end_sample=end,
        audio=audio,
    )


def iter_cycle_segments(
    cycles: pd.DataFrame, dataset_dir: Path, show_progress: bool = True
) -> Iterator[CycleSegment]:
    """Recorre los ciclos leyendo cada WAV una sola vez (acceso masivo eficiente).

    Los segmentos salen agrupados por grabacion; identifiquelos por ``cycle_id``.
    Cualquier error se propaga (no se omiten ciclos en silencio).
    """
    validate_cycle_table(cycles)
    groups = cycles.groupby("source_audio", sort=True)
    for source, group in tqdm(groups, total=groups.ngroups, desc="Segmentando", disable=not show_progress):
        audio, sample_rate = read_recording(Path(dataset_dir) / str(source))
        for row in group.itertuples(index=False):
            yield _segment_from_audio(row, audio, sample_rate)


# Manifiesto de segmentos con control de calidad


def longest_constant_run(audio: np.ndarray) -> int:
    """Longitud (muestras) de la racha mas larga de valores identicos consecutivos.

    Una racha larga indica senal plana: meseta de saturacion o corte de la senal.
    Para audio multicanal se evalua el promedio de canales.
    """
    x = audio if audio.ndim == 1 else audio.mean(axis=1)
    if x.size < 2:
        return int(x.size)
    boundaries = np.concatenate(([-1], np.flatnonzero(np.diff(x) != 0), [x.size - 1]))
    return int(np.max(np.diff(boundaries)))


def duration_outlier_flags(
    durations: pd.Series, iqr_k: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Marca duraciones atipicas con cercas de Tukey calculadas sobre ``log(duracion)``.

    La duracion es positiva y asimetrica a la derecha: en escala lineal la cerca
    inferior resulta negativa y nunca detectaria ciclos anormalmente cortos. En
    escala logaritmica las cercas son multiplicativas::

        Q1, Q3 = cuartiles de log(d);  IQR = Q3 - Q1
        cerca_k = [exp(Q1 - k*IQR), exp(Q3 + k*IQR)]

    Returns:
        DataFrame con una columna booleana ``duration_outlier_{nombre}`` por cada
        ``k`` y diccionario ``{nombre: (cerca_inferior_s, cerca_superior_s)}``.
    """
    log_d = np.log(durations.astype(float))
    q1, q3 = float(log_d.quantile(0.25)), float(log_d.quantile(0.75))
    iqr = q3 - q1
    flags = pd.DataFrame(index=durations.index)
    fences: dict[str, tuple[float, float]] = {}
    for name, k in iqr_k.items():
        low, high = math.exp(q1 - k * iqr), math.exp(q3 + k * iqr)
        fences[name] = (low, high)
        flags[f"duration_outlier_{name}"] = (durations < low) | (durations > high)
    return flags, fences


@dataclass
class SegmentationResult:
    """Manifiesto de segmentos, incidencias y cercas de duracion usadas."""

    manifest: pd.DataFrame
    issues: pd.DataFrame
    duration_fences: dict[str, tuple[float, float]]


def build_segment_manifest(
    cycles: pd.DataFrame,
    dataset_dir: Path,
    audit: AuditConfig,
    segmentation: SegmentationConfig,
    show_progress: bool = True,
) -> SegmentationResult:
    """Segmenta todos los ciclos validos y registra el control de calidad de cada uno.

    Por segmento se comprueba: audio legible, sample rate igual al de los metadatos,
    intervalo dentro del audio y no vacio, duracion real vs anotada (<= 1 muestra de
    diferencia por redondeo), NaN/Inf, silencio, saturacion y duracion atipica.
    """
    validate_cycle_table(cycles)
    dataset_dir = Path(dataset_dir)
    issues: list[dict[str, Any]] = [
        make_issue(WARNING, "ciclo_invalido_no_segmentado", row.recording_id, row.source_audio,
                   detail=f"{row.cycle_id}: intervalo invalido en la Fase 1")
        for row in cycles.loc[~cycles["is_valid"]].itertuples(index=False)
    ]
    rows: list[dict[str, Any]] = []
    valid = cycles.loc[cycles["is_valid"]]
    groups = valid.groupby("source_audio", sort=True)

    for source, group in tqdm(groups, total=groups.ngroups, desc="Segmentando", disable=not show_progress):
        try:
            audio, sample_rate = read_recording(dataset_dir / str(source))
        except Exception as exc:  # noqa: BLE001 - se reporta cualquier fallo de lectura
            issues.append(make_issue(ERROR, "audio_corrupto", str(group["recording_id"].iloc[0]), str(source),
                                     detail=repr(exc)))
            rows.extend(
                {**_base_record(row), "status": STATUS_ERROR, "status_detail": "audio_corrupto"}
                for row in group.itertuples(index=False)
            )
            continue

        for row in group.itertuples(index=False):
            base = _base_record(row)
            try:
                segment = _segment_from_audio(row, audio, sample_rate)
            except SegmentationError as exc:
                issues.append(make_issue(ERROR, exc.category, row.recording_id, row.source_audio, detail=str(exc)))
                rows.append({**base, "status": STATUS_ERROR, "status_detail": exc.category})
                continue

            stats = signal_stats(segment.audio, audit.clipping_level)
            duration_error = segment.duration_sec - float(row.duration_sec)
            flat_ms = 1000.0 * longest_constant_run(segment.audio) / sample_rate
            record = {
                **base,
                "start_sample": segment.start_sample,
                "end_sample": segment.end_sample,
                "n_samples": segment.n_samples,
                "segment_duration_sec": segment.duration_sec,
                "duration_error_samples": duration_error * sample_rate,
                **stats,
                "is_silent": bool(stats["peak_abs"] < audit.silence_peak_threshold),
                "is_saturated": bool(stats["clipping_fraction"] > audit.clipping_warning_fraction),
                "longest_flat_run_ms": flat_ms,
                "has_flatline": bool(flat_ms >= 1000.0 * segmentation.flatline_min_sec),
                "status": STATUS_OK,
                "status_detail": "",
            }
            if stats["n_nan"] or stats["n_inf"]:
                record.update(status=STATUS_ERROR, status_detail="segmento_nan_inf")
                issues.append(make_issue(ERROR, "segmento_nan_inf", row.recording_id, row.source_audio,
                                         detail=f"{row.cycle_id}: NaN={stats['n_nan']}, Inf={stats['n_inf']}"))
            if abs(record["duration_error_samples"]) > 1.0 + 1e-6:
                record.update(status=STATUS_ERROR, status_detail="duracion_inconsistente")
                issues.append(make_issue(ERROR, "duracion_inconsistente", row.recording_id, row.source_audio,
                                         detail=f"{row.cycle_id}: {record['duration_error_samples']:.2f} muestras"))
            if record["is_silent"]:
                issues.append(make_issue(WARNING, "segmento_silencioso", row.recording_id, row.source_audio,
                                         detail=f"{row.cycle_id}: pico={stats['peak_abs']:.2e}"))
            rows.append(record)
        del audio  # una grabacion en memoria a la vez

    manifest = pd.DataFrame(rows)
    fences: dict[str, tuple[float, float]] = {}
    ok = manifest["status"] == STATUS_OK if not manifest.empty else pd.Series(dtype=bool)
    if ok.any():
        flags, fences = duration_outlier_flags(
            manifest.loc[ok, "segment_duration_sec"], segmentation.duration_outlier_iqr_k
        )
        for column in flags.columns:
            manifest[column] = False
            manifest.loc[ok, column] = flags[column]
    return SegmentationResult(
        manifest=manifest, issues=pd.DataFrame(issues, columns=list(ISSUE_COLUMNS)), duration_fences=fences
    )


def _base_record(row: Any) -> dict[str, Any]:
    """Metadatos que el manifiesto conserva de cada ciclo."""
    return {
        "cycle_id": row.cycle_id,
        "recording_id": row.recording_id,
        "patient_id": str(row.patient_id),
        "source_audio": row.source_audio,
        "label": row.label,
        "equipment": row.equipment,
        "sample_rate": int(row.sample_rate),
        "start_sec": float(row.start_sec),
        "end_sec": float(row.end_sec),
        "annotated_duration_sec": float(row.duration_sec),
    }


def load_segment_manifest(path: Path) -> pd.DataFrame:
    """Carga ``manifiesto_segmentos.csv`` preservando los identificadores como texto."""
    return pd.read_csv(path, dtype={"patient_id": str, "recording_id": str, "cycle_id": str})
