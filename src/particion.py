"""Particion train/test 70/30 agrupada por paciente, reproducible y bloqueada (Fase 5).

Criterio (fijado antes de entrenar cualquier modelo; solo usa la composicion de
etiquetas por paciente, nunca rendimiento):

1. Candidatos: ``GroupShuffleSplit(n_splits=n_candidates, test_size, random_state)``
   con ``groups = patient_id`` sobre todos los ciclos con caracteristicas (las 4
   etiquetas). Cada candidato envia ``test_size`` de los pacientes a test, pero la
   fraccion de ciclos en test varia mucho (los pacientes aportan de 4 a 507 ciclos).
2. Para cada candidato y etiqueta ``k``: ``t_k = ciclos_test_k / ciclos_k``. Si
   ``t_k = test_size`` para todas las etiquetas, el test tiene a la vez el 30 % de
   los ciclos y la misma distribucion de clases que el total.
3. Se elige el candidato con menor ``max_k |t_k - test_size|``. Desempates, en orden:
   ``|f - test_size|`` (``f`` = fraccion global de ciclos en test),
   ``sum_k |t_k - test_size|`` y el menor numero de candidato.

Se usan las 4 etiquetas porque la particion es por paciente y es la misma para el
experimento de 3 clases y el de 4 clases. Opcionalmente (``split.balance_by``) el
mismo criterio minimax incluye tambien los niveles de ``equipment``.

Bloqueo: el manifiesto se guarda con una huella SHA-256 en ``bloqueo_particion.json``.
Toda fase posterior verifica la huella; regenerar un split distinto exige
``force=True`` y archiva el anterior en ``results/split_history/``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.configuracion import Config, LabelConfig, SplitConfig

TRAIN, TEST = "train", "test"
MANIFEST_NAME = "manifiesto_particion.csv"
LOCK_NAME = "bloqueo_particion.json"
MANIFEST_COLUMNS: tuple[str, ...] = (
    "patient_id", "audio_file", "cycle_id", "label", "split", "recording_id", "crackle", "wheeze", "equipment",
)


class SplitLeakageError(RuntimeError):
    """Un paciente aparece en train y en test (fuga de datos)."""


class SplitLockError(RuntimeError):
    """El split bloqueado no coincide con el esperado o fue modificado."""


@dataclass
class SplitSelection:
    test_patients: frozenset[str]
    candidate_index: int
    metrics: dict[str, Any]
    ranking: pd.DataFrame


def patient_label_counts(index: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    """Ciclos por paciente (filas, orden alfabetico) y etiqueta (columnas en ``labels``)."""
    counts = pd.crosstab(index["patient_id"].astype(str), index["label"])
    unknown = set(counts.columns) - set(labels)
    if unknown:
        raise ValueError(f"Etiquetas no definidas en la configuracion: {sorted(unknown)}")
    return counts.reindex(columns=list(labels), fill_value=0).sort_index()


def candidate_test_masks(
    patient_per_cycle: np.ndarray, patients: np.ndarray, n_candidates: int, test_size: float, seed: int
) -> np.ndarray:
    """Matriz booleana ``(n_candidates, n_pacientes)``: True = paciente en test.

    Usa ``GroupShuffleSplit`` a nivel de ciclo con ``groups = patient_id``.
    """
    position = {p: i for i, p in enumerate(patients)}
    splitter = GroupShuffleSplit(n_splits=n_candidates, test_size=test_size, random_state=seed)
    masks = np.zeros((n_candidates, len(patients)), dtype=bool)
    for row, (_, test_idx) in enumerate(splitter.split(np.zeros(len(patient_per_cycle)), groups=patient_per_cycle)):
        masks[row, [position[p] for p in np.unique(patient_per_cycle[test_idx])]] = True
    return masks


def score_candidates(masks: np.ndarray, count_tables: dict[str, pd.DataFrame], test_size: float) -> pd.DataFrame:
    """Metricas del criterio para cada candidato (una fila por candidato).

    Args:
        masks: ``(n_candidates, n_pacientes)``, True = paciente en test.
        count_tables: ``{atributo: ciclos por paciente (filas alineadas con masks) y nivel}``.
            ``label`` es obligatorio; ``equipment`` es opcional. ``max_abs_dev`` es el
            maximo sobre todos los niveles de todos los atributos balanceados.
    """
    labels_table = count_tables["label"]
    total = float(labels_table.to_numpy().sum())
    test_total = masks.astype(float) @ labels_table.to_numpy(dtype=float).sum(axis=1)
    table = pd.DataFrame({"candidate": np.arange(len(masks))})
    deviations = []
    for attribute, counts in count_tables.items():
        per_level = counts.to_numpy(dtype=float)
        fraction = (masks.astype(float) @ per_level) / per_level.sum(axis=0)
        deviation = np.abs(fraction - test_size)
        deviations.append(deviation)
        table[f"max_abs_dev_{attribute}"] = deviation.max(axis=1)
        for j, level in enumerate(counts.columns):
            table[f"t_{level}" if attribute == "label" else f"t_{attribute}_{level}"] = fraction[:, j]
    all_dev = np.hstack(deviations)
    table.insert(1, "max_abs_dev", all_dev.max(axis=1))
    table.insert(2, "overall_abs_dev", np.abs(test_total / total - test_size))
    table.insert(3, "sum_abs_dev", all_dev.sum(axis=1))
    table.insert(4, "test_fraction", test_total / total)
    table.insert(5, "n_test_patients", masks.sum(axis=1))
    return table


def balance_tables(index: pd.DataFrame, labels: LabelConfig, balance_by: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Tablas paciente x nivel de cada atributo a balancear (filas en el mismo orden)."""
    tables = {"label": patient_label_counts(index, labels.all_labels)}
    for attribute in balance_by:
        if attribute != "label":
            tables[attribute] = (pd.crosstab(index["patient_id"].astype(str), index[attribute])
                                 .reindex(index=tables["label"].index, fill_value=0))
    return tables


def select_split(index: pd.DataFrame, labels: LabelConfig, split_cfg: SplitConfig, seed: int) -> SplitSelection:
    """Genera los candidatos y elige el mejor segun el criterio del modulo."""
    tables = balance_tables(index, labels, split_cfg.balance_by)
    patients = tables["label"].index.to_numpy()
    masks = candidate_test_masks(index["patient_id"].astype(str).to_numpy(), patients, split_cfg.n_candidates,
                                 split_cfg.test_size, seed)
    scores = score_candidates(masks, tables, split_cfg.test_size)
    ranking = scores.sort_values(["max_abs_dev", "overall_abs_dev", "sum_abs_dev", "candidate"], kind="mergesort")
    best = int(ranking.iloc[0]["candidate"])
    metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in ranking.iloc[0].to_dict().items()}
    metrics["candidate"] = best
    return SplitSelection(
        test_patients=frozenset(patients[masks[best]]),
        candidate_index=best,
        metrics=metrics,
        ranking=ranking.head(20).reset_index(drop=True),
    )


def build_split_manifest(index: pd.DataFrame, test_patients: frozenset[str], labels: LabelConfig) -> pd.DataFrame:
    """Una fila por ciclo con su particion; columnas ``in_<modo>`` para cada modo de etiquetas."""
    manifest = pd.DataFrame({
        "patient_id": index["patient_id"].astype(str),
        "audio_file": index["audio_file"],
        "cycle_id": index["cycle_id"],
        "label": index["label"],
        "split": np.where(index["patient_id"].astype(str).isin(test_patients), TEST, TRAIN),
        "recording_id": index["recording_id"],
        "crackle": index["crackle"].astype(int),
        "wheeze": index["wheeze"].astype(int),
        "equipment": index["equipment"],
    })
    for mode, classes in labels.modes.items():
        manifest[f"in_{mode}"] = manifest["label"].isin(classes)
    return manifest.sort_values("cycle_id", ignore_index=True)


def assert_no_patient_leakage(manifest: pd.DataFrame) -> None:
    """Comprueba que cada paciente pertenezca a una sola particion.

    Raises:
        SplitLeakageError: con la lista de pacientes compartidos.
    """
    train = set(manifest.loc[manifest["split"] == TRAIN, "patient_id"])
    test = set(manifest.loc[manifest["split"] == TEST, "patient_id"])
    shared = train & test
    if shared:
        raise SplitLeakageError(f"Pacientes en train y test a la vez: {sorted(shared)}")
    if not manifest["cycle_id"].is_unique:
        raise SplitLeakageError("Hay ciclos repetidos en el manifiesto.")
    if not set(manifest["split"]) <= {TRAIN, TEST}:
        raise SplitLeakageError(f"Valores de split no validos: {sorted(set(manifest['split']))}")


def manifest_bytes(manifest: pd.DataFrame) -> bytes:
    """Serializacion canonica del manifiesto (UTF-8, fin de linea LF en cualquier SO)."""
    return manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_locked_split(
    manifest: pd.DataFrame, lock_info: dict[str, Any], results_dir: Path, force: bool = False
) -> tuple[str, Path, Path]:
    """Guarda el manifiesto y su bloqueo; nunca reemplaza en silencio un split distinto.

    Returns:
        ``(estado, ruta_manifiesto, ruta_bloqueo)`` con estado ``creado``,
        ``sin_cambios`` o ``reemplazado``.

    Raises:
        SplitLockError: si ya existe un split bloqueado distinto y ``force`` es False.
    """
    assert_no_patient_leakage(manifest)
    data = manifest_bytes(manifest)
    digest = sha256_bytes(data)
    manifest_path, lock_path = results_dir / MANIFEST_NAME, results_dir / LOCK_NAME
    status = "creado"
    if lock_path.is_file() and manifest_path.is_file():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if previous.get("manifest_sha256") == digest and sha256_bytes(manifest_path.read_bytes()) == digest:
            return "sin_cambios", manifest_path, lock_path
        if not force:
            raise SplitLockError(
                "Ya existe un split BLOQUEADO distinto en "
                f"{manifest_path}. No se reemplaza para no invalidar el test. "
                "Si realmente debe regenerarse (antes de entrenar), use --force-new-split."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = results_dir / "split_history" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, archive / MANIFEST_NAME)
        shutil.copy2(lock_path, archive / LOCK_NAME)
        status = "reemplazado"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(data)
    lock = {**lock_info, "manifest_sha256": digest, "locked_at_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds")}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return status, manifest_path, lock_path


def load_locked_split(cfg: Config) -> pd.DataFrame:
    """Carga el manifiesto verificando su huella y la ausencia de fuga.

    Raises:
        FileNotFoundError: si no se ha creado el split (Fase 5).
        SplitLockError: si el manifiesto fue modificado despues de bloquearse.
    """
    manifest_path = cfg.paths.results_dir / MANIFEST_NAME
    lock_path = cfg.paths.results_dir / LOCK_NAME
    if not manifest_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("No existe el split bloqueado. Ejecute: python scripts/crear_particion.py")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_bytes(manifest_path.read_bytes()) != lock.get("manifest_sha256"):
        raise SplitLockError(f"{manifest_path} fue modificado despues de bloquearse (huella distinta).")
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "cycle_id": str, "recording_id": str})
    assert_no_patient_leakage(manifest)
    return manifest


def split_summary(manifest: pd.DataFrame, labels: LabelConfig) -> pd.DataFrame:
    """Ciclos y pacientes por particion y etiqueta, para cada modo de etiquetas."""
    rows = []
    for mode, classes in labels.modes.items():
        subset = manifest[manifest["label"].isin(classes)]
        for split in (TRAIN, TEST, "total"):
            part = subset if split == "total" else subset[subset["split"] == split]
            for label in (*classes, "TOTAL"):
                cyc = part if label == "TOTAL" else part[part["label"] == label]
                all_label = subset if label == "TOTAL" else subset[subset["label"] == label]
                rows.append({
                    "label_mode": mode,
                    "split": split,
                    "label": label,
                    "n_cycles": len(cyc),
                    "pct_of_split": 100.0 * len(cyc) / max(len(part), 1),
                    "pct_of_label_in_split": 100.0 * len(cyc) / max(len(all_label), 1),
                    "n_patients": cyc["patient_id"].nunique(),
                })
    return pd.DataFrame(rows)
