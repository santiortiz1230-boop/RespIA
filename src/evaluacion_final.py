"""Orquestacion de la Fase 7: evaluacion final unica de los modelos congelados.

Orden de operaciones:

1. Catalogo de candidatos (experimento principal y mejoras propuestas) a partir de
   sus metadatos; se exige que compartan split, cache de caracteristicas y modo de
   etiquetas con la configuracion actual.
2. Modelos designados: el mejor por F1 de CV de cada grupo debe coincidir con lo
   pre-registrado en ``config.yaml`` (``evaluation.designated_*``).
3. Sello de congelacion (``results/sello_modelos.json``): SHA-256 de cada
   modelo, escrito antes de abrir el test. La evaluacion exige que los archivos sigan
   identicos: un modelo modificado despues del sello no se puede evaluar.
4. Solo si ``confirm=True`` se carga el conjunto de prueba (cada acceso queda en
   ``results/registro_accesos_prueba.csv``) y se evalua cada modelo una vez, sin reajustar nada.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.configuracion import Config
from src.constructor_datos import TEST_ACCESS_LOG, Dataset, load_test_data, load_train_data
from src.auditoria import improvement_label
from src.descriptores import descriptor_fingerprint
from src.evaluacion import (
    evaluate_model,
    metrics_by_group,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_test_comparison,
)
from src.extraccion_caracteristicas import config_fingerprint
from src.particion import LOCK_NAME
from src.utilidades import environment_info, save_dataframe, save_json, to_posix_relative

MAIN = "original"
FREEZE_NAME = "sello_modelos.json"
PURPOSE = "Fase 7 - evaluacion final unica de modelos congelados"
PROJECT_TARGET_F1 = 0.80
LIBRARIES = ("numpy", "pandas", "scikit-learn", "joblib", "librosa", "PyWavelets", "scipy")


class EvaluationSetupError(RuntimeError):
    """Los candidatos o la pre-registracion no son coherentes: no se abre el test."""


@dataclass
class Candidate:
    """Modelo congelado listo para evaluar."""

    group: str            # "original" (experimento principal) o nombre de la mejora
    representation: str
    classifier: str
    meta: dict[str, Any]
    model_path: Path

    @property
    def key(self) -> str:
        """Identificador unico para nombres de archivo."""
        return f"{self.group}__{self.representation}__{self.classifier}"

    @property
    def cv_f1(self) -> float:
        return float(self.meta["cv_scores"]["cv_f1_weighted"])


def load_candidates(cfg: Config) -> list[Candidate]:
    """Lee todos los candidatos guardados y verifica que sean comparables entre si."""
    root = cfg.paths.model_dir / "candidates"
    lock = json.loads((cfg.paths.results_dir / LOCK_NAME).read_text(encoding="utf-8"))
    fingerprint = config_fingerprint(cfg.preprocessing, cfg.features)
    candidates: list[Candidate] = []
    for meta_path in sorted(root.glob("*.json")) + sorted(root.glob("*/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        group = MAIN if meta_path.parent == root else meta_path.parent.name
        problems = []
        if meta.get("split_manifest_sha256") != lock["manifest_sha256"]:
            problems.append("split distinto")
        if meta.get("features_fingerprint") != fingerprint:
            problems.append("cache de caracteristicas distinta")
        if meta.get("label_mode") != cfg.labels.label_mode:
            problems.append("modo de etiquetas distinto")
        model_path = meta_path.with_suffix(".joblib")
        if not model_path.is_file():
            problems.append("falta el archivo .joblib")
        if problems:
            raise EvaluationSetupError(f"{meta_path.name}: {', '.join(problems)}. Reentrene antes de evaluar.")
        candidates.append(Candidate(group, meta["representation"], meta["classifier"], meta, model_path))
    expected = {(r, c) for r in cfg.features.representations for c in cfg.models}
    found = {(c.representation, c.classifier) for c in candidates if c.group == MAIN}
    if found != expected:
        raise EvaluationSetupError(
            f"El experimento principal debe tener exactamente {len(expected)} combinaciones; "
            f"faltan {sorted(expected - found)}, sobran {sorted(found - expected)} (ejecute entrenar_modelos.py).")
    return candidates


def designated_models(cfg: Config, candidates: list[Candidate]) -> tuple[Candidate, Candidate | None]:
    """Mejor por CV de cada grupo; debe coincidir con lo pre-registrado en ``config.yaml``."""
    main = max((c for c in candidates if c.group == MAIN), key=lambda c: c.cv_f1)
    declared = cfg.designated_main
    if (main.representation, main.classifier) != (declared["representation"], declared["classifier"]):
        raise EvaluationSetupError(
            f"El mejor por CV del experimento principal es {main.representation}+{main.classifier}, pero "
            f"config.yaml designa {declared}. Corrija la pre-registracion ANTES de abrir el test.")
    improvements = [c for c in candidates if c.group != MAIN]
    if not improvements:
        return main, None
    prototype = max(improvements, key=lambda c: c.cv_f1)
    declared = cfg.designated_prototype
    if (prototype.group, prototype.representation, prototype.classifier) != (
            declared.get("variant"), declared["representation"], declared["classifier"]):
        raise EvaluationSetupError(
            f"La mejor mejora por CV es {prototype.key}, pero config.yaml designa {declared}. "
            "Corrija la pre-registracion ANTES de abrir el test.")
    return main, prototype


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_test_opened(cfg: Config) -> bool:
    """``True`` si el registro de accesos al test tiene alguna fila ademas de la cabecera."""
    path = cfg.paths.results_dir / TEST_ACCESS_LOG
    return path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) > 1


def freeze_candidates(cfg: Config) -> Path:
    """Escribe el sello de congelacion (solo mientras el test siga cerrado).

    Raises:
        EvaluationSetupError: si el test ya se abrio (el sello no se puede reescribir).
    """
    path = cfg.paths.results_dir / FREEZE_NAME
    if is_test_opened(cfg):
        raise EvaluationSetupError("El test ya se abrio: el sello de congelacion no se puede reescribir.")
    candidates = load_candidates(cfg)
    main, prototype = designated_models(cfg, candidates)
    lock = json.loads((cfg.paths.results_dir / LOCK_NAME).read_text(encoding="utf-8"))
    return save_json({
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Modelos congelados ANTES de abrir el conjunto de prueba. Seleccion por F1 ponderada de CV (train).",
        "split_manifest_sha256": lock["manifest_sha256"],
        "label_mode": cfg.labels.label_mode,
        "designated_main": main.key,
        "designated_prototype": prototype.key if prototype else None,
        "models": {c.key: {"file": to_posix_relative(c.model_path, cfg.paths.root),
                           "sha256": sha256_file(c.model_path), "cv_f1_weighted": c.cv_f1} for c in candidates},
    }, path)


def verify_freeze(cfg: Config, candidates: list[Candidate]) -> dict[str, Any]:
    """Comprueba que los modelos son exactamente los sellados antes de abrir el test."""
    path = cfg.paths.results_dir / FREEZE_NAME
    if not path.is_file():
        raise EvaluationSetupError(f"Falta {FREEZE_NAME}: ejecute primero evaluar_modelos.py SIN --confirm-open-test.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sealed = manifest["models"]
    current = {c.key: sha256_file(c.model_path) for c in candidates}
    if set(current) != set(sealed):
        raise EvaluationSetupError(
            f"Los candidatos no son los sellados (sobran/faltan: {sorted(set(current) ^ set(sealed))}).")
    changed = sorted(key for key, digest in current.items() if sealed[key]["sha256"] != digest)
    if changed:
        raise EvaluationSetupError(f"Modelos modificados despues del sello de congelacion: {changed}.")
    return manifest


def smoke_check(cfg: Config, n_rows: int = 64) -> int:
    """Carga cada modelo y predice unas filas de train (nunca de test) para detectar fallos antes de evaluar.

    Raises:
        EvaluationSetupError: columnas distintas, orden de clases incorrecto o probabilidades invalidas.
    """
    candidates = load_candidates(cfg)
    samples = {r: load_train_data(cfg, r) for r in sorted({c.representation for c in candidates})}
    for candidate in candidates:
        data = samples[candidate.representation]
        if list(data.X.columns) != candidate.meta["feature_names"]:
            raise EvaluationSetupError(f"{candidate.key}: las columnas no coinciden con las del entrenamiento.")
        model = joblib.load(candidate.model_path)
        if list(model.classes_) != list(range(len(data.classes))):
            raise EvaluationSetupError(f"{candidate.key}: orden de clases {list(model.classes_)} inesperado.")
        rows = data.X.iloc[:n_rows]
        proba = model.predict_proba(rows)
        if model.predict(rows).shape != (len(rows),) or proba.shape != (len(rows), len(data.classes)) or (
                not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)) or not np.isfinite(proba).all():
            raise EvaluationSetupError(f"{candidate.key}: predicciones o probabilidades invalidas.")
    return len(candidates)


def evaluation_plan(cfg: Config) -> dict[str, Any]:
    """Resumen de lo que se evaluaria (no abre el test)."""
    candidates = load_candidates(cfg)
    main, prototype = designated_models(cfg, candidates)
    return {
        "n_main": sum(c.group == MAIN for c in candidates),
        "n_improvements": sum(c.group != MAIN for c in candidates),
        "representations": sorted({c.representation for c in candidates}),
        "designated_main": f"{main.representation} + {main.classifier} (CV {main.cv_f1:.4f})",
        "designated_prototype": (f"{prototype.key} (CV {prototype.cv_f1:.4f})" if prototype else "ninguna"),
    }


def _result_row(candidate: Candidate, result: Any, data: Dataset, designated: bool) -> dict[str, Any]:
    meta = candidate.meta
    row: dict[str, Any] = {
        "variant": candidate.group,
        "representation": candidate.representation,
        "classifier": candidate.classifier,
        "best_parameters": json.dumps(meta["best_parameters"], sort_keys=True, default=str),
        "cv_f1_weighted": candidate.cv_f1,
        "cv_f1_weighted_std": float(meta["cv_scores"].get("cv_f1_weighted_std", float("nan"))),
        "train_f1_weighted": float(meta["train_scores"]["train_f1_weighted"]),
        "test_accuracy": result.metrics["accuracy"],
        "test_precision_weighted": result.metrics["precision_weighted"],
        "test_recall_weighted": result.metrics["recall_weighted"],
        "test_f1_weighted": result.metrics["f1_weighted"],
        "test_auc_roc_weighted": result.metrics["auc_ovr_weighted"],
        "test_f1_macro": result.metrics["f1_macro"],
        "training_time_seconds": float(meta.get("training_time_seconds", float("nan"))),
        "inference_time_seconds": result.inference_time_seconds,
        "n_features": int(meta["n_features"]),
        "n_test_samples": data.n_samples,
        "designated_by_cv": designated,
    }
    for record in result.report.itertuples(index=False):
        row[f"test_f1_{record[0]}"] = float(record.f1)
        row[f"test_recall_{record[0]}"] = float(record.recall)
        row[f"test_precision_{record[0]}"] = float(record.precision)
        row[f"test_auc_{record[0]}"] = result.metrics[f"auc_ovr_{record[0]}"]
    return row


def _freeze(candidate: Candidate, row: dict[str, Any], cfg: Config, stem: str, role: str,
            lock: dict[str, Any]) -> Path:
    """Copia el modelo designado a ``models/`` y escribe sus metadatos completos."""
    target = cfg.paths.model_dir / f"{stem}.joblib"
    shutil.copy2(candidate.model_path, target)
    meta = candidate.meta
    model = joblib.load(target)
    scaler_info: Any
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["scaler"]
        scaler_info = {"type": "StandardScaler (ajustado solo con train, dentro del pipeline)",
                       "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    else:
        scaler_info = "un StandardScaler por miembro del jurado, dentro de cada pipeline (ajustados solo con train)"
    save_json({
        "role": role,
        "selection_criterion": "mayor F1 ponderada en validacion cruzada de train (pre-registrado antes del test)",
        "experiment_version": cfg.experiment_version,
        "variant": candidate.group,
        "representation": candidate.representation,
        "classifier": candidate.classifier,
        "members": meta.get("members"),
        "classes": meta["classes"],
        "class_encoding": meta["class_encoding"],
        "label_mode": meta["label_mode"],
        "feature_names": meta["feature_names"],
        "n_features": meta["n_features"],
        "scaler": scaler_info,
        "hyperparameters": meta["best_parameters"],
        "probability_calibration": meta.get("probability_calibration"),
        "sample_rate": cfg.preprocessing.target_sr,
        "preprocessing": cfg.preprocessing.as_dict(),
        "features": cfg.features.as_dict(),
        "descriptors": cfg.descriptors.as_dict() if candidate.representation in ("descriptors", "fusion_plus") else None,
        "features_fingerprint": meta["features_fingerprint"],
        "descriptors_fingerprint": descriptor_fingerprint(cfg),
        "split_manifest_sha256": lock["manifest_sha256"],
        "cv": meta.get("cv"),
        "cv_scores": meta["cv_scores"],
        "train_scores": meta["train_scores"],
        "test_scores": {k: v for k, v in row.items() if k.startswith("test_")},
        "source_model_file": to_posix_relative(candidate.model_path, cfg.paths.root),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **environment_info(LIBRARIES),
    }, cfg.paths.model_dir / f"metadatos_{'modelo' if stem == 'mejor_modelo' else stem}.json")
    return target


def _plot_model_figures(key: str, label: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray,
                        classes: tuple[str, ...], f1: float, figures_dir: Path) -> None:
    """Matriz de confusion y curvas ROC de un modelo (titulos en varias lineas para que no se corten)."""
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plot_confusion_matrix(confusion, classes,
                          f"Matriz de confusión (test)\n{label}\nF1 ponderada = " + f"{f1:.3f}".replace(".", ","),
                          figures_dir / "confusion_matrices" / f"{key}.png")
    plot_roc_curves(y_true, proba, classes, f"Curvas ROC uno-contra-resto (test)\n{label}",
                    figures_dir / "roc_curves" / f"{key}.png")


def replot_figures(cfg: Config) -> int:
    """Regenera matrices de confusion y curvas ROC desde ``results/predictions`` (no carga el test)."""
    paths = cfg.paths
    classes = tuple(cfg.labels.modes[cfg.labels.label_mode])
    tables = [pd.read_csv(paths.results_dir / "resultados_12_modelos.csv")]
    improvements = paths.results_dir / "improvements" / "resultados_mejoras_prueba.csv"
    if improvements.is_file():
        tables.append(pd.read_csv(improvements))
    count = 0
    for row in pd.concat(tables, ignore_index=True).itertuples():
        key = f"{row.variant}__{row.representation}__{row.classifier}"
        predictions = pd.read_parquet(paths.results_dir / "predictions" / f"{key}.parquet")
        proba = predictions[[f"proba_{name}" for name in classes]].to_numpy()
        _plot_model_figures(key, improvement_label(row.representation, row.classifier, row.variant),
                            predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy(), proba, classes,
                            float(row.test_f1_weighted), paths.figures_dir)
        count += 1
    return count


def run_final_evaluation(cfg: Config, confirm: bool) -> dict[str, Any]:
    """Evalua todos los candidatos sobre el conjunto de prueba (una sola vez por ejecucion).

    Args:
        confirm: debe ser ``True`` para cargar el test; con ``False`` solo se valida el
            plan y se devuelve sin tocar el test.
    """
    plan = evaluation_plan(cfg)
    if not confirm:
        return {"opened_test": False, "plan": plan}

    paths = cfg.paths
    lock = json.loads((paths.results_dir / LOCK_NAME).read_text(encoding="utf-8"))
    candidates = load_candidates(cfg)
    main_best, prototype = designated_models(cfg, candidates)
    freeze = verify_freeze(cfg, candidates)
    index = pd.read_parquet(paths.features_dir / "indice.parquet").set_index("cycle_id")

    datasets: dict[str, Dataset] = {}
    for representation in sorted({c.representation for c in candidates}):
        datasets[representation] = load_test_data(cfg, representation, PURPOSE)
    reference = next(iter(datasets.values()))
    for data in datasets.values():
        if not (np.array_equal(data.cycle_ids, reference.cycle_ids) and np.array_equal(data.y, reference.y)):
            raise EvaluationSetupError("Las representaciones no comparten las mismas filas de test.")
    equipment = index.loc[reference.cycle_ids, "equipment"].to_numpy()
    train_majority = int(np.bincount(load_train_data(cfg, reference.representation).y).argmax())
    baseline_f1 = float(f1_score(reference.y, np.full(reference.n_samples, train_majority), average="weighted",
                                 zero_division=0))

    rows, by_equipment = [], []
    for candidate in candidates:
        data = datasets[candidate.representation]
        if list(data.X.columns) != candidate.meta["feature_names"]:
            raise EvaluationSetupError(f"{candidate.key}: las columnas no coinciden con las del entrenamiento.")
        model = joblib.load(candidate.model_path)
        result = evaluate_model(model, data)
        designated = candidate is main_best or candidate is prototype
        rows.append(_result_row(candidate, result, data, designated))
        label = improvement_label(candidate.representation, candidate.classifier, candidate.group)
        save_dataframe(result.report, paths.results_dir / "classification_reports" / f"{candidate.key}.csv")
        _plot_model_figures(candidate.key, label, data.y, result.y_pred, result.proba, data.classes,
                            result.metrics["f1_weighted"], paths.figures_dir)
        predictions = pd.DataFrame({"cycle_id": data.cycle_ids, "patient_id": data.groups, "equipment": equipment,
                                    "y_true": data.y, "y_pred": result.y_pred})
        for code, name in enumerate(data.classes):
            predictions[f"proba_{name}"] = result.proba[:, code]
        (paths.results_dir / "predictions").mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(paths.results_dir / "predictions" / f"{candidate.key}.parquet", index=False)
        by_equipment.append(metrics_by_group(data.y, result.y_pred, equipment).assign(
            variant=candidate.group, representation=candidate.representation, classifier=candidate.classifier))

    table = pd.DataFrame(rows)
    main_table = table[table["variant"] == MAIN].sort_values("test_f1_weighted", ascending=False, ignore_index=True)
    improvements = table[table["variant"] != MAIN].sort_values("test_f1_weighted", ascending=False,
                                                              ignore_index=True)
    outputs = {
        "results_12_models": save_dataframe(main_table, paths.results_dir / "resultados_12_modelos.csv"),
        "by_equipment": save_dataframe(pd.concat(by_equipment, ignore_index=True),
                                       paths.results_dir / "metricas_prueba_por_equipo.csv"),
        "comparison_f1": plot_test_comparison(
            main_table.rename(columns={"designated_by_cv": "designated"}), PROJECT_TARGET_F1, baseline_f1,
            "Experimento principal: F1 ponderada en test y en validación cruzada (12 combinaciones)",
            paths.figures_dir / "comparacion_f1.png"),
    }
    if not improvements.empty:
        outputs["results_improvements"] = save_dataframe(
            improvements, paths.results_dir / "improvements" / "resultados_mejoras_prueba.csv")
        combined = pd.concat([improvements, main_table[main_table["designated_by_cv"]]], ignore_index=True)
        outputs["comparison_f1_improvements"] = plot_test_comparison(
            combined.rename(columns={"designated_by_cv": "designated"}), PROJECT_TARGET_F1, baseline_f1,
            "Mejoras propuestas: F1 ponderada en test y en validación cruzada",
            paths.figures_dir / "comparacion_f1_mejoras.png")

    main_row = main_table[main_table["designated_by_cv"]].iloc[0].to_dict()
    outputs["best_model"] = _freeze(main_best, main_row, cfg, "mejor_modelo",
                                    "BEST_MODEL del experimento principal (hipotesis)", lock)
    summary: dict[str, Any] = {
        "fase": "7 - evaluacion final",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_criterion": cfg.best_model_selection,
        "target_f1_weighted": PROJECT_TARGET_F1,
        "n_test_samples": reference.n_samples,
        "n_test_patients": int(len(set(reference.groups))),
        "test_majority_baseline_f1_weighted": baseline_f1,
        "best_model": main_row,
        "target_reached_by_best_model": bool(main_row["test_f1_weighted"] >= PROJECT_TARGET_F1),
        "top_main_by_test": main_table.iloc[0][["representation", "classifier", "test_f1_weighted"]].to_dict(),
    }
    if prototype is not None:
        prototype_row = improvements[improvements["designated_by_cv"]].iloc[0].to_dict()
        outputs["prototype_model"] = _freeze(prototype, prototype_row, cfg, "modelo_prototipo",
                                             "Mejor MEJORA PROPUESTA por CV (prototipo / interfaz)", lock)
        summary["prototype_model"] = prototype_row
        summary["target_reached_by_prototype"] = bool(prototype_row["test_f1_weighted"] >= PROJECT_TARGET_F1)
        summary["top_improvement_by_test"] = improvements.iloc[0][
            ["variant", "representation", "classifier", "test_f1_weighted"]].to_dict()
    outputs["best_model_results"] = save_json(summary, paths.results_dir / "resultados_mejor_modelo.json")
    save_json({**summary, "plan": plan, "split_manifest_sha256": lock["manifest_sha256"],
               "models_frozen_at_utc": freeze["frozen_at_utc"],
               "outputs": {k: to_posix_relative(v, paths.root) for k, v in outputs.items()},
               **environment_info(LIBRARIES)}, paths.audit_dir / "ejecucion_evaluacion.json")
    return {"opened_test": True, "plan": plan, "summary": summary, "main": main_table,
            "improvements": improvements, "outputs": outputs, "baseline_f1": baseline_f1}
