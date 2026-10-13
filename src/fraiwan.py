"""Dataset de validacion externa de Fraiwan et al. (2021): lectura y auditoria.

Solo lectura y descripcion. Este dataset no se usa para entrenar, ajustar
hiperparametros ni elegir modelos.

Estructura real (verificada en la auditoria):

* ``Audio Files/`` con 336 WAV = 112 pacientes x 3 modos de filtro del estetoscopio
  3M Littmann 3200. El nombre de cada archivo trae todos los campos::

      {modo}P{paciente}_{diagnostico},{tipo de sonido},{ubicacion},{edad},{sexo}.wav
      BP101_Asthma,E W,P L M,12,F.wav

  ``modo``: B = campana (Bell), D = diafragma (Diaphragm), E = extendido (Extended).
* ``Data annotation.xlsx``: una fila por paciente (edad, sexo, ubicacion, tipo de
  sonido, diagnostico) y la leyenda de abreviaturas.
* La etiqueta es una por grabacion completa. No hay tiempos de ciclos respiratorios.

Leyenda del propio dataset para ``tipo de sonido``: I = inspiratorio, E = espiratorio,
W = sibilancias (wheezes), C = crepitantes (crackles), Crep = crepitaciones, N = normal.
``B`` y ``Bronchial`` no estan definidos en la leyenda.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import soundfile as sf

FILTER_MODES = {"B": "Bell", "D": "Diaphragm", "E": "Extended"}
# Banda nominal de cada modo segun el fabricante y el articulo del dataset (Hz).
FILTER_BANDS_HZ = {"B": (20.0, 200.0), "D": (100.0, 500.0), "E": (50.0, 500.0)}
_NAME = re.compile(r"^(?P<mode>[BDE])P(?P<patient>\d+)_(?P<rest>.+)$")
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class FraiwanFormatError(ValueError):
    """Un archivo no sigue la estructura esperada del dataset."""


def decode_sound_type(sound_type: str) -> dict[str, bool]:
    """Decodifica la etiqueta con la leyenda del dataset (no es una equivalencia con ICBHI).

    Devuelve si la etiqueta declara sibilancias, crepitantes, sonido normal u otro
    sonido sin definicion en la leyenda (``B``, ``Bronchial``).
    """
    tokens = sound_type.strip().split()
    lowered = [t.lower() for t in tokens]
    return {
        "has_wheeze": "W" in tokens,
        "has_crackle": "C" in tokens or "crep" in lowered,
        "is_normal": tokens == ["N"],
        "has_undefined": "B" in tokens or "bronchial" in lowered,
    }


def parse_fraiwan_filename(name: str) -> dict[str, Any]:
    """Separa los campos del nombre de un WAV de Fraiwan.

    Raises:
        FraiwanFormatError: el nombre no tiene el prefijo o los 5 campos esperados.
    """
    stem = Path(name).stem
    match = _NAME.match(stem)
    if not match:
        raise FraiwanFormatError(f"Nombre sin el prefijo {{B,D,E}}P<numero>_: {name}")
    fields = [field.strip() for field in match.group("rest").split(",")]
    if len(fields) != 5:
        raise FraiwanFormatError(f"Se esperaban 5 campos separados por coma y hay {len(fields)}: {name}")
    diagnosis, sound_type, location, age, sex = fields
    if not age.isdigit() or sex.upper() not in ("M", "F"):
        raise FraiwanFormatError(f"Edad o sexo no validos en: {name}")
    return {
        "recording_id": stem,
        "filter_code": match.group("mode"),
        "filter_mode": FILTER_MODES[match.group("mode")],
        "patient_id": f"F{int(match.group('patient')):03d}",
        "patient_number": int(match.group("patient")),
        "diagnosis": diagnosis,
        "sound_type": " ".join(sound_type.split()),
        "location": " ".join(location.split()),
        "age": int(age),
        "sex": sex.upper(),
        **decode_sound_type(sound_type),
    }


def read_annotation_xlsx(path: Path) -> pd.DataFrame:
    """Lee ``Data annotation.xlsx`` con la libreria estandar (una fila por paciente, en orden).

    Columnas: ``patient_number, age, sex, location, sound_type, diagnosis``.
    """
    with zipfile.ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.iter(f"{{{_XLSX_NS['m']}}}t"))
                  for item in shared_root.findall("m:si", _XLSX_NS)]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{{{_XLSX_NS['m']}}}row"):
        cells: dict[str, str | None] = {}
        for cell in row.findall("m:c", _XLSX_NS):
            value = cell.find("m:v", _XLSX_NS)
            text = None if value is None else (shared[int(value.text)] if cell.get("t") == "s" else value.text)
            cells[re.match(r"[A-Z]+", cell.get("r")).group(0)] = text
        rows.append(cells)
    if not rows or rows[0].get("A") != "Age":
        raise FraiwanFormatError("La hoja no empieza con la cabecera esperada (Age, Gender, ...).")
    records = []
    for cells in rows[1:]:
        if cells.get("A") is None or cells.get("D") is None:
            continue
        records.append({"age": int(float(cells["A"])), "sex": str(cells.get("B") or "").strip().upper(),
                        "location": " ".join(str(cells.get("C") or "").split()),
                        "sound_type": " ".join(str(cells["D"]).split()),
                        "diagnosis": str(cells.get("E") or "").strip()})
    table = pd.DataFrame(records)
    table.insert(0, "patient_number", np.arange(1, len(table) + 1))
    return table


def build_fraiwan_metadata(dataset_dir: Path) -> pd.DataFrame:
    """Una fila por WAV: campos del nombre y propiedades del audio (sin decodificar la senal)."""
    audio_dir = Path(dataset_dir) / "Audio Files"
    files = sorted(audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No hay WAV en {audio_dir}")
    rows = []
    for path in files:
        info = sf.info(str(path))
        rows.append({**parse_fraiwan_filename(path.name), "source_audio": f"Audio Files/{path.name}",
                     "sample_rate": int(info.samplerate), "n_channels": int(info.channels),
                     "subtype": info.subtype, "n_frames": int(info.frames),
                     "duration_sec": info.frames / info.samplerate})
    return pd.DataFrame(rows).sort_values(["patient_number", "filter_code"], ignore_index=True)


def signal_summary(path: Path, clipping_level: float = 0.999) -> dict[str, float]:
    """Indicadores de calidad de una grabacion: pico, RMS, fraccion saturada y valores no finitos."""
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    return {"peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "clipped_fraction": float(np.mean(np.abs(audio) >= clipping_level)),
            "n_nonfinite": int(np.sum(~np.isfinite(audio)))}


def band_power_fractions(audio: np.ndarray, sample_rate: int,
                         edges_hz: tuple[float, ...] = (0.0, 100.0, 500.0, 2000.0)) -> list[float]:
    """Fraccion de la potencia de la senal en cada banda ``[edges[i], edges[i+1])`` (hasta el ultimo borde)."""
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    total = float(spectrum[freqs < edges_hz[-1]].sum())
    return [float(spectrum[(freqs >= low) & (freqs < high)].sum() / total) if total > 0 else 0.0
            for low, high in zip(edges_hz[:-1], edges_hz[1:])]


def compare_with_annotation(recordings: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Compara los campos del nombre de archivo con la hoja de calculo, paciente por paciente."""
    merged = recordings.merge(annotation, on="patient_number", how="left", suffixes=("", "_xlsx"))
    rows = []
    for field in ("age", "sex", "location", "sound_type", "diagnosis"):
        left = merged[field].astype(str).str.strip().str.lower()
        right = merged[f"{field}_xlsx"].astype(str).str.strip().str.lower()
        differ = left != right
        rows.append({"campo": field, "n_archivos": len(merged), "n_distintos": int(differ.sum()),
                     "ejemplos": "; ".join(f"{r.recording_id}: '{getattr(r, field)}' vs '{getattr(r, field + '_xlsx')}'"
                                           for r in merged[differ].head(3).itertuples())})
    return pd.DataFrame(rows)
