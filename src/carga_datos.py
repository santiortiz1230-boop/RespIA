"""Lectura del dataset ICBHI 2017: inventario de archivos, nombres, anotaciones y audio.

Este modulo solo lee: nunca modifica ni renombra los archivos originales.
Se reutiliza en la auditoria (Fase 1), la segmentacion (Fase 2) y la validacion externa.

Formato de nombre (segun ``filename_format.txt`` del dataset)::

    {paciente}_{indice_grabacion}_{posicion_toracica}_{modo_adquisicion}_{equipo}
    p. ej. 101_1b1_Al_sc_Meditron

Formato de anotacion (una fila por ciclo respiratorio, separada por espacios/tabs)::

    inicio_seg  fin_seg  crackles(0/1)  wheezes(0/1)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.configuracion import AuditConfig, IcbhiConfig, LabelConfig
from src.utilidades import to_posix_relative

RECORDING_NAME_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)_(?P<recording_index>[^_]+)_(?P<chest_location>[^_]+)"
    r"_(?P<acquisition_mode>[^_]+)_(?P<equipment>[^_]+)$"
)

ANNOTATION_COLUMNS: tuple[str, ...] = ("annotation_line", "start_sec", "end_sec", "crackle", "wheeze")

# Las 10 primeras columnas son las exigidas por la metodologia; el resto es trazabilidad.
CYCLE_METADATA_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "patient_id",
    "source_audio",
    "start_sec",
    "end_sec",
    "duration_sec",
    "crackle",
    "wheeze",
    "label",
    "sample_rate",
    "recording_id",
    "cycle_index",
    "annotation_line",
    "chest_location",
    "acquisition_mode",
    "equipment",
    "audio_duration_sec",
    "n_channels",
    "is_valid",
    "invalid_reason",
    "exceeds_audio_by_sec",
    "overlaps_previous",
)

ISSUE_COLUMNS: tuple[str, ...] = ("severity", "category", "recording_id", "file", "line", "detail")
ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"


def make_issue(
    severity: str,
    category: str,
    recording_id: str = "",
    file: str = "",
    line: int | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "recording_id": recording_id,
        "file": file,
        "line": line,
        "detail": detail,
    }


# Nombres de archivo


@dataclass(frozen=True)
class RecordingName:
    """Campos codificados en el nombre de una grabacion ICBHI."""

    recording_id: str
    patient_id: str
    recording_index: str
    chest_location: str
    acquisition_mode: str
    equipment: str


def _stem(name: str | Path) -> str:
    """Nombre sin carpeta ni extension ``.wav``/``.txt``."""
    path = Path(name)
    return path.stem if path.suffix.lower() in {".wav", ".txt"} else path.name


def parse_recording_name(name: str | Path) -> RecordingName:
    """Interpreta el nombre de una grabacion ICBHI.

    Args:
        name: nombre, ruta o stem (p. ej. ``"101_1b1_Al_sc_Meditron.wav"``).

    Raises:
        ValueError: si el nombre no sigue el formato de 5 campos.
    """
    stem = _stem(name)
    match = RECORDING_NAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"Nombre de grabacion no interpretable: '{stem}'")
    return RecordingName(recording_id=stem, **match.groupdict())


def is_recording_name(name: str | Path) -> bool:
    return RECORDING_NAME_PATTERN.match(_stem(name)) is not None


def extract_patient_id(name: str | Path) -> str:
    """Extrae el identificador de paciente (primer campo del nombre) como texto."""
    return parse_recording_name(name).patient_id


def make_cycle_id(recording_id: str, cycle_index: int) -> str:
    """Identificador unico y determinista de ciclo: ``{grabacion}_c{indice:03d}``."""
    return f"{recording_id}_c{cycle_index:03d}"


# Inventario de archivos


@dataclass
class FileInventory:
    """Archivos encontrados en la carpeta del dataset (busqueda recursiva)."""

    root: Path
    audio: dict[str, list[Path]]
    annotations: dict[str, list[Path]]
    auxiliary_txt: list[Path]
    other_files: list[Path]

    @property
    def n_audio_files(self) -> int:
        """Numero total de archivos de audio (incluye duplicados)."""
        return sum(len(paths) for paths in self.audio.values())

    @property
    def n_annotation_files(self) -> int:
        """Numero total de TXT de anotacion (incluye duplicados)."""
        return sum(len(paths) for paths in self.annotations.values())

    @property
    def n_txt_files(self) -> int:
        """Numero total de TXT (anotaciones + auxiliares)."""
        return self.n_annotation_files + len(self.auxiliary_txt)

    def relative(self, path: Path) -> str:
        """Ruta POSIX relativa a la raiz del dataset."""
        return to_posix_relative(path, self.root)


def scan_dataset_dir(root: Path, audio_extension: str = ".wav", annotation_extension: str = ".txt") -> FileInventory:
    """Recorre recursivamente ``root`` y clasifica los archivos encontrados.

    Un TXT se considera anotacion si su nombre sigue el formato de grabacion o si
    existe un audio con el mismo nombre; en otro caso es un TXT auxiliar
    (p. ej. ``filename_format.txt``).

    Raises:
        FileNotFoundError: si la carpeta no existe o no contiene audio ni anotaciones.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {root}")

    audio: dict[str, list[Path]] = defaultdict(list)
    txt_files: list[Path] = []
    other: list[Path] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix == audio_extension:
            audio[path.stem].append(path)
        elif suffix == annotation_extension:
            txt_files.append(path)
        else:
            other.append(path)

    annotations: dict[str, list[Path]] = defaultdict(list)
    auxiliary: list[Path] = []
    for path in txt_files:
        if path.stem in audio or is_recording_name(path.stem):
            annotations[path.stem].append(path)
        else:
            auxiliary.append(path)

    if not audio and not annotations:
        raise FileNotFoundError(f"La carpeta {root} no contiene archivos {audio_extension} ni {annotation_extension}.")
    return FileInventory(root=root, audio=dict(audio), annotations=dict(annotations), auxiliary_txt=auxiliary, other_files=other)


def pair_recordings(inventory: FileInventory, icbhi: IcbhiConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Asocia cada WAV con su TXT por nombre y valida los campos del nombre.

    Returns:
        Tabla de grabaciones (una fila por nombre unico) y lista de incidencias.
    """
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    known = {
        "chest_location": set(icbhi.known_chest_locations),
        "acquisition_mode": set(icbhi.known_acquisition_modes),
        "equipment": set(icbhi.known_equipment),
    }

    for stem in sorted(set(inventory.audio) | set(inventory.annotations)):
        wavs = inventory.audio.get(stem, [])
        txts = inventory.annotations.get(stem, [])
        for kind, paths in (("wav", wavs), ("txt", txts)):
            if len(paths) > 1:
                issues.append(make_issue(
                    WARNING, f"{kind}_duplicado", stem, inventory.relative(paths[0]),
                    detail=f"{len(paths)} copias: {[inventory.relative(p) for p in paths]}; se usa la primera.",
                ))
        if not wavs:
            issues.append(make_issue(ERROR, "anotacion_sin_audio", stem, inventory.relative(txts[0])))
        if not txts:
            issues.append(make_issue(ERROR, "audio_sin_anotacion", stem, inventory.relative(wavs[0])))

        fields: dict[str, Any] = dict.fromkeys(
            ("patient_id", "recording_index", "chest_location", "acquisition_mode", "equipment"), None
        )
        try:
            parsed = parse_recording_name(stem)
        except ValueError as exc:
            issues.append(make_issue(ERROR, "nombre_no_interpretable", stem, detail=str(exc)))
        else:
            fields = {
                "patient_id": parsed.patient_id,
                "recording_index": parsed.recording_index,
                "chest_location": parsed.chest_location,
                "acquisition_mode": parsed.acquisition_mode,
                "equipment": parsed.equipment,
            }
            for field_name, allowed in known.items():
                if fields[field_name] not in allowed:
                    issues.append(make_issue(
                        WARNING, "codigo_desconocido", stem,
                        detail=f"{field_name}='{fields[field_name]}' no esta en {sorted(allowed)}",
                    ))

        rows.append({
            "recording_id": stem,
            **fields,
            "audio_path": inventory.relative(wavs[0]) if wavs else "",
            "annotation_path": inventory.relative(txts[0]) if txts else "",
            "has_audio": bool(wavs),
            "has_annotation": bool(txts),
        })
    return pd.DataFrame(rows), issues


# Anotaciones


def read_annotation_file(path: Path, recording_id: str = "") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Lee un TXT de anotaciones ICBHI sin descartar errores en silencio.

    Las lineas que no pueden representarse como ciclo (columnas != 4, valores no
    numericos o no finitos, banderas distintas de 0/1) no se incluyen en la tabla,
    pero cada una se reporta como incidencia ``ERROR`` con su numero de linea.

    Returns:
        DataFrame con columnas :data:`ANNOTATION_COLUMNS` y lista de incidencias.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            issues.append(make_issue(INFO, "linea_vacia", recording_id, path.name, line_no))
            continue
        parts = line.split()
        if len(parts) != 4:
            issues.append(make_issue(
                ERROR, "linea_malformada", recording_id, path.name, line_no,
                f"se esperaban 4 columnas y hay {len(parts)}: {line!r}",
            ))
            continue
        try:
            start, end, crackle, wheeze = (float(value) for value in parts)
        except ValueError:
            issues.append(make_issue(ERROR, "valor_no_numerico", recording_id, path.name, line_no, repr(line)))
            continue
        if not all(math.isfinite(v) for v in (start, end, crackle, wheeze)):
            issues.append(make_issue(ERROR, "valor_no_finito", recording_id, path.name, line_no, repr(line)))
            continue
        if crackle not in (0.0, 1.0) or wheeze not in (0.0, 1.0):
            issues.append(make_issue(
                ERROR, "bandera_invalida", recording_id, path.name, line_no,
                f"crackle/wheeze deben ser 0 o 1: {line!r}",
            ))
            continue
        rows.append({
            "annotation_line": line_no,
            "start_sec": start,
            "end_sec": end,
            "crackle": int(crackle),
            "wheeze": int(wheeze),
        })

    frame = pd.DataFrame(rows, columns=list(ANNOTATION_COLUMNS)).astype(
        {"annotation_line": "int64", "start_sec": "float64", "end_sec": "float64", "crackle": "int64", "wheeze": "int64"}
    )
    return frame, issues


def validate_cycle_intervals(cycles: pd.DataFrame, audio_duration_sec: float, tolerance_sec: float) -> pd.DataFrame:
    """Valida los intervalos de ciclo contra la duracion real del audio.

    Criterios (un ciclo es valido solo si cumple los tres):
        * ``start_sec >= 0``
        * ``end_sec > start_sec``
        * ``end_sec <= audio_duration_sec`` (+ ``tolerance_sec`` solo por coma flotante)

    El solapamiento con el ciclo anterior se informa (``overlaps_previous``) pero
    no invalida el ciclo, porque no forma parte de los criterios definidos.

    Returns:
        Copia de ``cycles`` con ``duration_sec``, ``is_valid``, ``invalid_reason``,
        ``exceeds_audio_by_sec`` y ``overlaps_previous``.
    """
    out = cycles.copy()
    start = out["start_sec"].to_numpy(dtype=float)
    end = out["end_sec"].to_numpy(dtype=float)
    negative_start = start < 0.0
    end_not_after_start = end <= start
    exceeds_audio = end > audio_duration_sec + tolerance_sec

    reasons = [
        ";".join(
            name
            for flag, name in (
                (neg, "inicio_negativo"),
                (order, "fin_no_mayor_que_inicio"),
                (exceed, "fin_excede_duracion_audio"),
            )
            if flag
        )
        for neg, order, exceed in zip(negative_start, end_not_after_start, exceeds_audio)
    ]
    previous_end = np.concatenate(([-np.inf], end[:-1]))

    out["duration_sec"] = end - start
    out["is_valid"] = ~(negative_start | end_not_after_start | exceeds_audio)
    out["invalid_reason"] = reasons
    out["exceeds_audio_by_sec"] = np.where(exceeds_audio, end - audio_duration_sec, 0.0)
    out["overlaps_previous"] = start < previous_end - tolerance_sec
    return out


# Audio


@dataclass(frozen=True)
class AudioInfo:
    """Metadatos de cabecera de un archivo de audio."""

    sample_rate: int
    n_channels: int
    n_frames: int
    duration_sec: float
    subtype: str
    container: str


def read_audio_info(path: Path) -> AudioInfo:
    """Lee solo la cabecera del audio (rapido, sin cargar muestras)."""
    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        n_channels=int(info.channels),
        n_frames=int(info.frames),
        duration_sec=float(info.frames) / float(info.samplerate),
        subtype=str(info.subtype),
        container=str(info.format),
    )


def compute_signal_stats(path: Path, clipping_level: float) -> dict[str, float | int]:
    """Lee el audio completo y calcula indicadores de integridad de senal.

    Returns:
        ``n_samples_read`` mas los indicadores de :func:`signal_stats`.
    """
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return {"n_samples_read": int(data.shape[0]), **signal_stats(data, clipping_level)}


def signal_stats(data: np.ndarray, clipping_level: float) -> dict[str, float | int]:
    """Indicadores de integridad de una senal (grabacion completa o segmento).

    Returns:
        ``n_nan``, ``n_inf``, ``peak_abs``, ``rms``, ``dc_offset`` y
        ``clipping_fraction`` (fraccion de muestras con ``|x| >= clipping_level``,
        escala completa = 1.0). Los estadisticos usan solo las muestras finitas.
    """
    finite = data[np.isfinite(data)].astype(np.float64)
    has_samples = finite.size > 0
    return {
        "n_nan": int(np.isnan(data).sum()),
        "n_inf": int(np.isinf(data).sum()),
        "peak_abs": float(np.max(np.abs(finite))) if has_samples else float("nan"),
        "rms": float(np.sqrt(np.mean(finite**2))) if has_samples else float("nan"),
        "dc_offset": float(np.mean(finite)) if has_samples else float("nan"),
        "clipping_fraction": float(np.mean(np.abs(finite) >= clipping_level)) if has_samples else float("nan"),
    }


# Construccion de metadatos por grabacion y por ciclo


@dataclass
class IcbhiMetadata:
    inventory: FileInventory
    recordings: pd.DataFrame
    cycles: pd.DataFrame
    issues: pd.DataFrame


def _audit_audio(
    wav_path: Path, recording_id: str, relative: str, audit: AuditConfig | None
) -> tuple[AudioInfo | None, dict[str, Any], list[dict[str, Any]]]:
    """Lee cabecera (y opcionalmente la senal completa) de una grabacion."""
    issues: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}
    try:
        info = read_audio_info(wav_path)
    except Exception as exc:  # noqa: BLE001 - se reporta cualquier fallo de lectura
        issues.append(make_issue(ERROR, "audio_ilegible", recording_id, relative, detail=repr(exc)))
        return None, extra, issues

    extra.update(
        sample_rate=info.sample_rate,
        n_channels=info.n_channels,
        n_frames=info.n_frames,
        duration_sec=info.duration_sec,
        subtype=info.subtype,
        container=info.container,
    )
    if info.n_frames == 0:
        issues.append(make_issue(ERROR, "audio_vacio", recording_id, relative))
    if info.n_channels != 1:
        issues.append(make_issue(INFO, "audio_multicanal", recording_id, relative, detail=f"{info.n_channels} canales"))

    if audit is not None and audit.signal_check:
        try:
            stats = compute_signal_stats(wav_path, audit.clipping_level)
        except Exception as exc:  # noqa: BLE001
            issues.append(make_issue(ERROR, "audio_corrupto", recording_id, relative, detail=repr(exc)))
            return info, extra, issues
        extra.update(stats)
        extra["is_silent"] = bool(stats["peak_abs"] < audit.silence_peak_threshold)
        if stats["n_samples_read"] != info.n_frames:
            issues.append(make_issue(
                ERROR, "audio_truncado", recording_id, relative,
                detail=f"cabecera={info.n_frames} muestras, leidas={stats['n_samples_read']}",
            ))
        if stats["n_nan"] or stats["n_inf"]:
            issues.append(make_issue(
                ERROR, "audio_nan_inf", recording_id, relative, detail=f"NaN={stats['n_nan']}, Inf={stats['n_inf']}"
            ))
        if extra["is_silent"]:
            issues.append(make_issue(WARNING, "audio_silencioso", recording_id, relative, detail=f"pico={stats['peak_abs']:.2e}"))
        if stats["clipping_fraction"] > audit.clipping_warning_fraction:
            issues.append(make_issue(
                WARNING, "audio_saturado", recording_id, relative,
                detail=f"{100 * stats['clipping_fraction']:.2f} % de muestras con |x|>={audit.clipping_level}",
            ))
    return info, extra, issues


def build_icbhi_metadata(
    dataset_dir: Path,
    icbhi: IcbhiConfig,
    labels: LabelConfig,
    audit: AuditConfig | None = None,
    show_progress: bool = True,
) -> IcbhiMetadata:
    """Construye la tabla de grabaciones, la tabla de ciclos y el registro de incidencias.

    ``source_audio`` se guarda relativo a ``dataset_dir`` (portable). Solo se generan
    ciclos para grabaciones con audio legible, anotacion y nombre interpretable;
    cualquier exclusion queda registrada en ``issues``.
    """
    inventory = scan_dataset_dir(dataset_dir, icbhi.audio_extension, icbhi.annotation_extension)
    recordings, issues = pair_recordings(inventory, icbhi)

    cycle_frames: list[pd.DataFrame] = []
    extras: list[dict[str, Any]] = []
    iterator = tqdm(
        recordings.itertuples(index=False), total=len(recordings), desc="Auditando grabaciones", disable=not show_progress
    )
    for rec in iterator:
        extra: dict[str, Any] = {"n_annotation_rows": 0, "n_cycles": 0, "n_valid_cycles": 0}
        info: AudioInfo | None = None
        if rec.has_audio:
            info, audio_extra, audio_issues = _audit_audio(
                inventory.root / rec.audio_path, rec.recording_id, rec.audio_path, audit
            )
            extra.update(audio_extra)
            issues.extend(audio_issues)

        if rec.has_annotation:
            annotations, annotation_issues = read_annotation_file(inventory.root / rec.annotation_path, rec.recording_id)
            issues.extend(annotation_issues)
            extra["n_annotation_rows"] = len(annotations)
            if annotations.empty:
                issues.append(make_issue(WARNING, "anotacion_sin_ciclos", rec.recording_id, rec.annotation_path))
            elif info is None or rec.patient_id is None:
                issues.append(make_issue(
                    ERROR, "ciclos_no_utilizables", rec.recording_id, rec.annotation_path,
                    detail=f"{len(annotations)} ciclos sin audio legible o sin patient_id interpretable",
                ))
            else:
                cycles = validate_cycle_intervals(annotations, info.duration_sec, icbhi.time_tolerance_sec)
                cycles["cycle_index"] = np.arange(1, len(cycles) + 1)
                cycles["cycle_id"] = [make_cycle_id(rec.recording_id, i) for i in cycles["cycle_index"]]
                cycles["recording_id"] = rec.recording_id
                cycles["patient_id"] = rec.patient_id
                cycles["source_audio"] = rec.audio_path
                cycles["label"] = [labels.label_for(c, w) for c, w in zip(cycles["crackle"], cycles["wheeze"])]
                cycles["sample_rate"] = info.sample_rate
                cycles["n_channels"] = info.n_channels
                cycles["audio_duration_sec"] = info.duration_sec
                cycles["chest_location"] = rec.chest_location
                cycles["acquisition_mode"] = rec.acquisition_mode
                cycles["equipment"] = rec.equipment
                cycle_frames.append(cycles)
                extra["n_cycles"] = len(cycles)
                extra["n_valid_cycles"] = int(cycles["is_valid"].sum())

                for cyc in cycles.loc[~cycles["is_valid"]].itertuples(index=False):
                    issues.append(make_issue(
                        ERROR, "intervalo_invalido", rec.recording_id, rec.annotation_path, int(cyc.annotation_line),
                        f"{cyc.invalid_reason}: inicio={cyc.start_sec}, fin={cyc.end_sec}, "
                        f"duracion_audio={info.duration_sec:.6f}",
                    ))
                for cyc in cycles.loc[cycles["overlaps_previous"]].itertuples(index=False):
                    issues.append(make_issue(
                        INFO, "ciclo_solapado", rec.recording_id, rec.annotation_path, int(cyc.annotation_line),
                        f"inicio={cyc.start_sec} anterior al fin del ciclo previo",
                    ))
        extras.append(extra)

    recordings = pd.concat([recordings, pd.DataFrame(extras)], axis=1)
    if cycle_frames:
        cycles_df = pd.concat(cycle_frames, ignore_index=True)[list(CYCLE_METADATA_COLUMNS)]
    else:
        cycles_df = pd.DataFrame(columns=list(CYCLE_METADATA_COLUMNS))
    issues_df = pd.DataFrame(issues, columns=list(ISSUE_COLUMNS))
    return IcbhiMetadata(inventory=inventory, recordings=recordings, cycles=cycles_df, issues=issues_df)


def load_cycle_metadata(path: Path) -> pd.DataFrame:
    """Carga ``metadatos_ciclos.csv`` preservando ``patient_id`` como texto."""
    return pd.read_csv(path, dtype={"patient_id": str, "recording_id": str, "cycle_id": str})
