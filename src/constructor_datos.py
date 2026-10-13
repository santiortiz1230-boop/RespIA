"""Construccion de matrices X / y para entrenamiento y prueba (Fases 5 en adelante).

Une la cache de caracteristicas (Fase 4) con el split bloqueado (Fase 5) y el modo
de etiquetas (``labels.label_mode``: 3class o 4class) sin volver a extraer nada.

* Las clases se codifican como enteros ``0..K-1`` en el orden de ``config.yaml``
  (p. ej. NORMAL=0, CREPITANTE=1, SIBILANCIA=2). Como los enteros ya estan
  ordenados, scikit-learn conserva ese orden en ``classes_`` y en ``predict_proba``.
* ``load_train_data`` es la unica via para entrenar y ajustar hiperparametros.
* ``load_test_data`` exige declarar el proposito y registra cada acceso en
  ``results/registro_accesos_prueba.csv``: el uso del test queda auditado.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.configuracion import Config
from src.descriptores import DESCRIPTORS, load_descriptors
from src.extraccion_caracteristicas import load_features
from src.particion import TEST, TRAIN, load_locked_split

TEST_ACCESS_LOG = "registro_accesos_prueba.csv"
TEST_ACCESS_COLUMNS = ("timestamp_utc", "representation", "label_mode", "n_samples", "purpose")


class DatasetAlignmentError(RuntimeError):
    """La cache de caracteristicas y el split no cubren los mismos ciclos."""


@dataclass(frozen=True)
class Dataset:
    """Matriz de una particion lista para scikit-learn."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray        # patient_id de cada fila (para CV agrupada)
    cycle_ids: np.ndarray
    classes: tuple[str, ...]  # classes[i] = nombre de la clase codificada como i
    representation: str
    split: str
    label_mode: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


def encode_labels(labels: pd.Series | np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    """Codifica etiquetas de texto como enteros segun el orden de ``classes``.

    Raises:
        ValueError: si aparece una etiqueta que no pertenece a ``classes``.
    """
    mapping = {name: i for i, name in enumerate(classes)}
    values = pd.Series(labels).astype(str)
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise ValueError(f"Etiquetas fuera de {classes}: {unknown}")
    return values.map(mapping).to_numpy(dtype=np.int64)


def decode_labels(codes: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    """Inverso de :func:`encode_labels`."""
    return np.asarray(classes, dtype=object)[np.asarray(codes, dtype=int)]


FUSION = "fusion"
FUSION_PLUS = "fusion_plus"


def load_representation(cfg: Config, representation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga ``(indice, X)`` de una representacion o de una combinacion (MEJORAS PROPUESTAS).

    * ``fusion``: concatena ``cfg.features.representations`` en ese orden.
    * ``descriptors``: descriptores especificos de sibilancias y crepitantes.
    * ``fusion_plus``: ``fusion`` seguida de ``descriptors``.

    Se exige que todas las partes tengan exactamente las mismas filas.
    """
    if representation == DESCRIPTORS:
        return load_descriptors(cfg)
    if representation not in (FUSION, FUSION_PLUS):
        return load_features(cfg, representation)
    parts = [load_features(cfg, name) for name in cfg.features.representations]
    if representation == FUSION_PLUS:
        parts.append(load_descriptors(cfg))
    index = parts[0][0]
    for other_index, _ in parts[1:]:
        if not other_index["cycle_id"].equals(index["cycle_id"]):
            raise DatasetAlignmentError("Las representaciones no tienen las mismas filas; no se pueden fusionar.")
    return index, pd.concat([X for _, X in parts], axis=1)


def _assemble(cfg: Config, representation: str, split: str, label_mode: str | None) -> Dataset:
    """Une caracteristicas y split, filtra por particion y modo de etiquetas."""
    mode = label_mode or cfg.labels.label_mode
    if mode not in cfg.labels.modes:
        raise ValueError(f"label_mode '{mode}' no existe en config.yaml ({sorted(cfg.labels.modes)}).")
    classes = cfg.labels.modes[mode]
    manifest = load_locked_split(cfg)
    index, X = load_representation(cfg, representation)
    if set(index["cycle_id"]) != set(manifest["cycle_id"]):
        raise DatasetAlignmentError(
            "La cache de caracteristicas y el split no cubren los mismos ciclos; regenere el split o la cache."
        )
    assignment = manifest.set_index("cycle_id")
    split_of = index["cycle_id"].map(assignment["split"]).to_numpy()
    if not np.array_equal(index["patient_id"].astype(str).to_numpy(),
                          index["cycle_id"].map(assignment["patient_id"]).to_numpy()):
        raise DatasetAlignmentError("patient_id distinto entre la cache y el split.")
    keep = (split_of == split) & index["label"].isin(classes).to_numpy()
    selected = index.loc[keep]
    return Dataset(
        X=X.loc[keep].reset_index(drop=True),
        y=encode_labels(selected["label"], classes),
        groups=selected["patient_id"].astype(str).to_numpy(),
        cycle_ids=selected["cycle_id"].to_numpy(),
        classes=tuple(classes),
        representation=representation,
        split=split,
        label_mode=mode,
    )


def load_train_data(cfg: Config, representation: str, label_mode: str | None = None) -> Dataset:
    """Particion de entrenamiento (unica permitida para ajustar y seleccionar modelos)."""
    return _assemble(cfg, representation, TRAIN, label_mode)


def load_test_data(cfg: Config, representation: str, purpose: str, label_mode: str | None = None) -> Dataset:
    """Particion de prueba bloqueada: exige un proposito y registra el acceso.

    Solo debe llamarse para la evaluacion final (Fase 7), la robustez del modelo
    congelado (Fase 9) y el analisis estadistico (Fase 11).
    """
    if not purpose or not purpose.strip():
        raise ValueError("Debe declarar el proposito del acceso al conjunto de prueba.")
    data = _assemble(cfg, representation, TEST, label_mode)
    log_test_access(cfg.paths.results_dir, representation, data.label_mode, data.n_samples, purpose.strip())
    return data


def log_test_access(results_dir: Path, representation: str, label_mode: str, n_samples: int, purpose: str) -> Path:
    """Agrega una fila al registro de accesos al test (crea el archivo si no existe)."""
    path = results_dir / TEST_ACCESS_LOG
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new_file:
            writer.writerow(TEST_ACCESS_COLUMNS)
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), representation, label_mode,
                         n_samples, purpose])
    return path


def ensure_test_access_log(results_dir: Path) -> Path:
    """Crea el registro de accesos vacio (solo cabecera) si aun no existe."""
    path = results_dir / TEST_ACCESS_LOG
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(TEST_ACCESS_COLUMNS) + "\n", encoding="utf-8")
    return path
