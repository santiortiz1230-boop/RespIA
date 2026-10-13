"""Busqueda de hiperparametros con CV agrupada por paciente y ajuste final (Fase 6).

* Solo se usa la particion de entrenamiento (``load_train_data``); el test no se toca.
* CV: ``StratifiedGroupKFold`` (grupos = paciente). Se prueba primero el numero de
  folds de la metodologia (10) y, si no es viable segun un criterio fijado a priori
  (cada fold de validacion con todas las clases, >= N pacientes y >= M ciclos por
  clase), se usa el alternativo (5). El mismo plan de folds se usa en las 12
  combinaciones (mismas filas, mismas etiquetas, mismos grupos).
* ``GridSearchCV`` recibe el divisor agrupado y ``groups``; el ``StandardScaler`` del
  pipeline se reajusta dentro de cada fold. Metrica de seleccion: F1 ponderada.
* El modelo final se ajusta con los mejores hiperparametros sobre todo el train. En
  SVM, ``predict`` es la regla del propio SVC (la evaluada en la CV) y
  ``predict_proba`` viene de una calibracion de Platt con folds agrupados por paciente
  (``CalibratedProbaSVC``); por eso la F1 de CV del modelo final es la de la busqueda.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.configuracion import Config
from src.constructor_datos import Dataset
from src.modelos import build_final_pipeline, build_pipeline, param_grid, strip_prefix

METRIC_FUNCTIONS = {
    "f1_weighted": partial(f1_score, average="weighted", zero_division=0),
    "accuracy": accuracy_score,
    "precision_weighted": partial(precision_score, average="weighted", zero_division=0),
    "recall_weighted": partial(recall_score, average="weighted", zero_division=0),
}
METRICS: tuple[str, ...] = tuple(METRIC_FUNCTIONS)
SCORERS = {name: make_scorer(function) for name, function in METRIC_FUNCTIONS.items()}


class CVPlanError(RuntimeError):
    """Ningun numero de folds cumple el criterio de viabilidad."""


class FoldLeakageError(RuntimeError):
    """Un paciente aparece en la parte de ajuste y en la de validacion de un fold."""


# Plan de validacion cruzada


@dataclass
class CVPlan:
    """Divisor elegido, sus folds y la evidencia de viabilidad."""

    n_splits: int
    splitter: StratifiedGroupKFold
    folds: list[tuple[np.ndarray, np.ndarray]]
    composition: pd.DataFrame
    viability: pd.DataFrame
    reason: str


def fold_composition(y: np.ndarray, groups: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
                     classes: tuple[str, ...]) -> pd.DataFrame:
    """Ciclos y pacientes por clase en la validacion de cada fold (verifica fuga por paciente)."""
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        shared = set(groups[fit_idx]) & set(groups[val_idx])
        if shared:
            raise FoldLeakageError(f"Fold {fold}: pacientes en ajuste y validacion: {sorted(shared)}")
        row: dict[str, Any] = {"fold": fold, "n_fit": len(fit_idx), "n_val": len(val_idx),
                               "patients_val": len(set(groups[val_idx]))}
        for code, name in enumerate(classes):
            mask = y[val_idx] == code
            row[f"{name}_cycles"] = int(mask.sum())
            row[f"{name}_patients"] = len(set(groups[val_idx][mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def is_viable(composition: pd.DataFrame, classes: tuple[str, ...], min_patients: int, min_cycles: int) -> bool:
    """Todas las clases, en todos los folds, con >= ``min_patients`` pacientes y >= ``min_cycles`` ciclos."""
    patients = composition[[f"{c}_patients" for c in classes]].to_numpy()
    cycles = composition[[f"{c}_cycles" for c in classes]].to_numpy()
    return bool((patients >= min_patients).all() and (cycles >= min_cycles).all())


def plan_cv(data: Dataset, cfg: Config) -> CVPlan:
    """Elige el numero de folds (preferido o alternativo) con el criterio de viabilidad."""
    t = cfg.training
    tried: list[dict[str, Any]] = []
    for n_splits in dict.fromkeys((t.preferred_n_splits, t.fallback_n_splits)):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=t.shuffle,
                                        random_state=cfg.random_state if t.shuffle else None)
        folds = list(splitter.split(data.X, data.y, data.groups))
        composition = fold_composition(data.y, data.groups, folds, data.classes)
        viable = is_viable(composition, data.classes, t.min_patients_per_class_per_fold,
                           t.min_cycles_per_class_per_fold)
        patient_cols = [f"{c}_patients" for c in data.classes]
        cycle_cols = [f"{c}_cycles" for c in data.classes]
        worst = composition[patient_cols].min()
        tried.append({
            "n_splits": n_splits,
            "viable": viable,
            "min_patients_per_class_per_fold": int(composition[patient_cols].to_numpy().min()),
            "min_cycles_per_class_per_fold": int(composition[cycle_cols].to_numpy().min()),
            "worst_class": str(worst.idxmin()).replace("_patients", ""),
            "val_size_min": int(composition["n_val"].min()),
            "val_size_max": int(composition["n_val"].max()),
        })
        if viable:
            reason = (f"{n_splits} folds: cada clase tiene >= {tried[-1]['min_patients_per_class_per_fold']} "
                      f"pacientes y >= {tried[-1]['min_cycles_per_class_per_fold']} ciclos en todo fold")
            if n_splits != t.preferred_n_splits:
                first = tried[0]
                reason += (f"; {t.preferred_n_splits} folds NO viable: la clase {first['worst_class']} queda con "
                           f"{first['min_patients_per_class_per_fold']} paciente(s) en algun fold "
                           f"(minimo exigido {t.min_patients_per_class_per_fold})")
            return CVPlan(n_splits, splitter, folds, composition, pd.DataFrame(tried), reason)
    raise CVPlanError(f"Ningun plan de CV es viable: {tried}")


# Busqueda y ajuste final


@dataclass
class SearchResult:
    representation: str
    classifier: str
    best_params: dict[str, Any]
    best_index: int
    cv_scores: dict[str, float]
    fold_f1: list[float]
    cv_results: pd.DataFrame
    final_model: Pipeline
    train_scores: dict[str, float]
    search_time_seconds: float
    training_time_seconds: float
    n_grid_candidates: int
    n_splits: int
    extra: dict[str, Any] = field(default_factory=dict)


def resubstitution_scores(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Metricas del modelo final sobre su propio train (diagnostico de sobreajuste)."""
    predicted = model.predict(X)
    return {f"train_{name}": float(function(y, predicted)) for name, function in METRIC_FUNCTIONS.items()}


def search_and_fit(data: Dataset, classifier: str, cfg: Config, plan: CVPlan,
                   n_jobs: int | None = None, overrides: dict[str, Any] | None = None) -> SearchResult:
    """GridSearchCV agrupado por paciente + ajuste final con los mejores hiperparametros.

    La mejor combinacion es la de mayor F1 ponderada media en validacion; en caso de
    empate, la primera en el orden del grid (misma regla que ``GridSearchCV``).
    """
    pipeline = build_pipeline(classifier, cfg, overrides=overrides)
    grid = param_grid(classifier, cfg)
    search = GridSearchCV(pipeline, grid, scoring=SCORERS, refit=False, cv=plan.splitter,
                          n_jobs=cfg.training.n_jobs if n_jobs is None else n_jobs,
                          return_train_score=True, error_score="raise")
    started = time.perf_counter()
    search.fit(data.X, data.y, groups=data.groups)
    search_time = time.perf_counter() - started

    results = pd.DataFrame(search.cv_results_)
    if int(search.n_splits_) != plan.n_splits:
        raise RuntimeError("GridSearchCV no uso el numero de folds del plan.")
    ranks = results[f"rank_test_{cfg.training.scoring}"].to_numpy()
    best = int(np.argmin(ranks))  # primera combinacion con el mejor rango (regla de sklearn)
    best_params = dict(results.loc[best, "params"])
    cv_scores: dict[str, float] = {}
    for metric in METRICS:
        cv_scores[f"cv_search_{metric}"] = float(results.loc[best, f"mean_test_{metric}"])
        cv_scores[f"cv_search_{metric}_std"] = float(results.loc[best, f"std_test_{metric}"])
        cv_scores[f"cv_fit_{metric}"] = float(results.loc[best, f"mean_train_{metric}"])
    search_fold_f1 = [float(results.loc[best, f"split{i}_test_f1_weighted"]) for i in range(plan.n_splits)]

    # La regla de clasificacion del modelo final es exactamente la evaluada en la
    # busqueda (en SVM la calibracion solo afecta a predict_proba).
    for metric in METRICS:
        cv_scores[f"cv_{metric}"] = cv_scores[f"cv_search_{metric}"]
        cv_scores[f"cv_{metric}_std"] = cv_scores[f"cv_search_{metric}_std"]
    fold_f1 = search_fold_f1

    final = build_final_pipeline(classifier, cfg, best_params, calibration_folds=plan.folds, overrides=overrides)
    started = time.perf_counter()
    final.fit(data.X, data.y)
    training_time = time.perf_counter() - started
    extra: dict[str, Any] = {}
    if cfg.models[classifier].calibration is not None:
        proba_argmax = final.predict_proba(data.X).argmax(axis=1)
        extra["train_predict_vs_proba_argmax_disagreement"] = float(np.mean(final.predict(data.X) != proba_argmax))

    keep = ["params"] + [c for c in results.columns if c.startswith(("mean_", "std_", "rank_", "split"))]
    return SearchResult(
        representation=data.representation,
        classifier=classifier,
        best_params=strip_prefix(best_params),
        best_index=best,
        cv_scores=cv_scores,
        fold_f1=fold_f1,
        cv_results=results[keep],
        final_model=final,
        train_scores=resubstitution_scores(final, data.X, data.y),
        search_time_seconds=search_time,
        training_time_seconds=training_time,
        n_grid_candidates=len(results),
        n_splits=plan.n_splits,
        extra=extra,
    )


def dummy_baselines(data: Dataset, plan: CVPlan, seed: int) -> pd.DataFrame:
    """Lineas base triviales con los mismos folds (referencia para interpretar la F1).

    * ``most_frequent``: predice siempre la clase mayoritaria del fold de ajuste.
    * ``stratified``: predice al azar respetando las proporciones de clase del ajuste.
    """
    rows = []
    for strategy in ("most_frequent", "stratified"):
        scores = {name: [] for name in METRICS}
        for fit_idx, val_idx in plan.folds:
            model = DummyClassifier(strategy=strategy, random_state=seed).fit(data.X.iloc[fit_idx], data.y[fit_idx])
            predicted = model.predict(data.X.iloc[val_idx])
            for name, function in METRIC_FUNCTIONS.items():
                scores[name].append(float(function(data.y[val_idx], predicted)))
        row: dict[str, Any] = {"baseline": strategy}
        for name, values in scores.items():
            row[f"cv_{name}"] = float(np.mean(values))
            row[f"cv_{name}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(result: SearchResult, data: Dataset, model_path: Path, root: Path) -> dict[str, Any]:
    """Fila de ``resultados_cv_12_modelos.csv``."""
    return {
        "representation": result.representation,
        "classifier": result.classifier,
        "best_parameters": json.dumps(result.best_params, sort_keys=True, default=str),
        "cv_f1_weighted": result.cv_scores["cv_f1_weighted"],
        "cv_f1_weighted_std": result.cv_scores["cv_f1_weighted_std"],
        "cv_accuracy": result.cv_scores["cv_accuracy"],
        "cv_precision_weighted": result.cv_scores["cv_precision_weighted"],
        "cv_recall_weighted": result.cv_scores["cv_recall_weighted"],
        "cv_fit_f1_weighted": result.cv_scores["cv_fit_f1_weighted"],
        "train_f1_weighted": result.train_scores["train_f1_weighted"],
        "gap_train_minus_cv_f1": result.train_scores["train_f1_weighted"] - result.cv_scores["cv_f1_weighted"],
        "fold_f1_weighted": json.dumps([round(v, 4) for v in result.fold_f1]),
        "n_features": int(data.X.shape[1]),
        "n_train_samples": data.n_samples,
        "n_splits": result.n_splits,
        "n_grid_candidates": result.n_grid_candidates,
        "search_time_seconds": result.search_time_seconds,
        "training_time_seconds": result.training_time_seconds,
        "model_file": model_path.relative_to(root).as_posix(),
    }


def save_candidate(result: SearchResult, data: Dataset, metadata: dict[str, Any], model_dir: Path,
                   variant: str | None = None) -> Path:
    """Guarda el pipeline final (joblib) y sus metadatos (JSON).

    Experimento principal en ``models/candidates``; mejoras en ``models/candidates/<variant>``.
    """
    folder = model_dir / "candidates" / variant if variant else model_dir / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{result.representation}__{result.classifier}"
    model_path = folder / f"{stem}.joblib"
    joblib.dump(result.final_model, model_path, compress=3)
    info = {
        **metadata,
        "representation": result.representation,
        "classifier": result.classifier,
        "label_mode": data.label_mode,
        "classes": list(data.classes),
        "class_encoding": {name: i for i, name in enumerate(data.classes)},
        "feature_names": list(data.X.columns),
        "n_features": int(data.X.shape[1]),
        "n_train_samples": data.n_samples,
        "n_train_patients": int(len(set(data.groups))),
        "pipeline_steps": [name for name, _ in result.final_model.steps],
        "best_parameters": result.best_params,
        "final_only_parameters": metadata.get("final_only_parameters", {}),
        "cv_scores": result.cv_scores,
        "cv_fold_f1_weighted": result.fold_f1,
        "train_scores": result.train_scores,
        **result.extra,
        "search_time_seconds": result.search_time_seconds,
        "training_time_seconds": result.training_time_seconds,
        "n_grid_candidates": result.n_grid_candidates,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / f"{stem}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    return model_path
