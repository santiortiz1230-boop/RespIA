"""Definicion de los 3 clasificadores como Pipeline(StandardScaler -> modelo) (Fase 6).

El ``StandardScaler`` forma parte de cada pipeline (tambien de Random Forest, por
coherencia metodologica): dentro de ``GridSearchCV`` se ajusta de nuevo en cada
fold solo con las filas de entrenamiento del fold, y en el modelo final solo con
el 70 % de train. Nunca ve datos de validacion ni de prueba.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.configuracion import Config

CLASSIFIER_NAMES: tuple[str, ...] = ("random_forest", "svm_rbf", "knn")
DISPLAY_NAMES: dict[str, str] = {"random_forest": "Random Forest", "svm_rbf": "SVM-RBF", "knn": "KNN"}
REPRESENTATION_NAMES: dict[str, str] = {"mfcc": "MFCC", "mel": "Mel", "wavelet": "Wavelet", "rms": "RMS"}
STEP = "classifier"


def base_estimator(name: str, cfg: Config, overrides: dict[str, Any] | None = None) -> BaseEstimator:
    """Clasificador con sus parametros fijos (mas ``overrides`` de una mejora) y la semilla del proyecto."""
    fixed = {**cfg.models[name].fixed, **(overrides or {})}
    if name == "random_forest":
        return RandomForestClassifier(random_state=cfg.random_state, n_jobs=1, **fixed)
    if name == "svm_rbf":
        return SVC(random_state=cfg.random_state, **fixed)
    if name == "knn":
        return KNeighborsClassifier(**fixed)
    raise ValueError(f"Clasificador desconocido: {name}")


def build_pipeline(name: str, cfg: Config, scaler: BaseEstimator | None = None,
                   overrides: dict[str, Any] | None = None) -> Pipeline:
    """``Pipeline([("scaler", StandardScaler()), ("classifier", modelo)])``.

    Args:
        scaler: solo para pruebas (p. ej. un escalador instrumentado); por defecto StandardScaler.
        overrides: parametros fijos adicionales de una MEJORA PROPUESTA (p. ej. class_weight).
    """
    return Pipeline([("scaler", scaler if scaler is not None else StandardScaler()),
                     (STEP, base_estimator(name, cfg, overrides))])


def param_grid(name: str, cfg: Config) -> dict[str, list[Any]]:
    """Grid de GridSearchCV con el prefijo del paso del pipeline."""
    return {f"{STEP}__{key}": list(values) for key, values in cfg.models[name].grid.items()}


def final_overrides(name: str, cfg: Config) -> dict[str, Any]:
    """Parametros que solo se activan en el ajuste final (p. ej. ``probability=True`` en SVM)."""
    return {f"{STEP}__{key}": value for key, value in cfg.models[name].final_params.items()}


class CalibratedProbaSVC(ClassifierMixin, BaseEstimator):
    """SVC que clasifica con su propia regla de decision y da probabilidades calibradas.

    * ``predict`` / ``decision_function``: los del SVC ajustado con todo el train, es
      decir, exactamente la regla evaluada en la validacion cruzada.
    * ``predict_proba``: calibracion de Platt (``CalibratedClassifierCV``,
      ``ensemble=False``) con predicciones fuera de fold usando folds agrupados por
      paciente. Solo se usa para ROC/AUC y para mostrar probabilidades en la interfaz.

    Nota: como en cualquier SVM con probabilidades de Platt, la clase de mayor
    probabilidad puede no coincidir con ``predict`` en casos limite; la tasa de
    desacuerdo se registra en los metadatos del modelo.
    """

    def __init__(self, estimator: BaseEstimator | None = None, method: str = "sigmoid",
                 cv: Sequence[tuple[np.ndarray, np.ndarray]] | None = None) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, X: Any, y: Any) -> CalibratedProbaSVC:  # noqa: N803
        """Ajusta la calibracion; su estimador interno es el SVC ajustado con todo X."""
        self.calibrator_ = CalibratedClassifierCV(clone(self.estimator), method=self.method, ensemble=False,
                                                  cv=self.cv)
        self.calibrator_.fit(X, y)
        self.svc_ = self.calibrator_.calibrated_classifiers_[0].estimator
        self.classes_ = self.calibrator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Clase segun la regla del SVC (la misma de la validacion cruzada)."""
        return self.svc_.predict(X)

    def decision_function(self, X: Any) -> np.ndarray:  # noqa: N803
        return self.svc_.decision_function(X)

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        """Probabilidades calibradas (Platt, folds agrupados por paciente)."""
        return self.calibrator_.predict_proba(X)


def build_final_pipeline(
    name: str, cfg: Config, params: dict[str, Any],
    calibration_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    """Pipeline final con los hiperparametros elegidos.

    Si el clasificador tiene ``calibration`` (SVM), el paso ``classifier`` es un
    :class:`CalibratedProbaSVC`: clasifica igual que el SVC evaluado en la CV y
    obtiene probabilidades calibradas con folds agrupados por paciente.

    Args:
        params: hiperparametros con prefijo ``classifier__`` (salida de GridSearchCV).
        calibration_folds: pares (ajuste, validacion) en posiciones de las filas que se
            usaran en ``fit`` (obligatorio si hay calibracion).
    """
    pipeline = clone(build_pipeline(name, cfg, overrides=overrides)).set_params(**params,
                                                                              **final_overrides(name, cfg))
    method = cfg.models[name].calibration
    if method is None:
        return pipeline
    if calibration_folds is None:
        raise ValueError(f"{name} requiere folds agrupados para calibrar las probabilidades.")
    calibrated = CalibratedProbaSVC(
        estimator=pipeline.named_steps[STEP], method=method,
        cv=[(np.asarray(fit_idx), np.asarray(val_idx)) for fit_idx, val_idx in calibration_folds],
    )
    return Pipeline([("scaler", pipeline.named_steps["scaler"]), (STEP, calibrated)])


def classifier_core(model: Pipeline) -> BaseEstimator:
    """Clasificador base del pipeline (el SVC ajustado, si esta envuelto en la calibracion)."""
    step = model.named_steps[STEP]
    return step.svc_ if isinstance(step, CalibratedProbaSVC) else step


def strip_prefix(params: dict[str, Any]) -> dict[str, Any]:
    """Quita el prefijo ``classifier__`` para reportar hiperparametros."""
    return {key.split("__", 1)[1] if "__" in key else key: value for key, value in params.items()}
