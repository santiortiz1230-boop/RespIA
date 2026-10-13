"""Evaluacion final de los modelos congelados (Fase 7).

Los modelos no se reajustan aqui: solo se cargan y se aplican a datos que no vieron.

Metricas (por modelo):

* accuracy; precision, recall y F1 **ponderadas** (por soporte de cada clase); F1 macro.
* **AUC-ROC multiclase**: esquema One-vs-Rest sobre ``predict_proba``, promediado con
  ponderacion por soporte (``roc_auc_score(multi_class="ovr", average="weighted")``),
  y ademas el AUC One-vs-Rest de cada clase por separado.
* matriz de confusion (filas = clase real, columnas = clase predicha, en el orden de
  ``classes``) y reporte por clase.
* tiempo de inferencia del modelo sobre todo el conjunto (sin extraccion de caracteristicas).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.auditoria import (
    COLOR_INCLUDED,
    COLOR_SERIES_2,
    GRID_COLOR,
    INK_PRIMARY,
    INK_SECONDARY,
    _fmt_int,
    _style_axes,
    improvement_label,
)
from src.constructor_datos import Dataset

CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CLASS_STYLES = ("-", "--", "-.", ":")
DECIMAL_COMMA = FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ","))


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    report: pd.DataFrame
    confusion: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    inference_time_seconds: float


def auc_scores(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    """AUC-ROC One-vs-Rest ponderada por soporte y AUC One-vs-Rest de cada clase.

    Si alguna clase no aparece en ``y_true`` su AUC no esta definida: se devuelve
    NaN para ella y para el promedio (nunca se inventa un valor).
    """
    scores: dict[str, float] = {}
    present = np.unique(y_true)
    for code, name in enumerate(classes):
        binary = (y_true == code).astype(int)
        scores[f"auc_ovr_{name}"] = (float(roc_auc_score(binary, proba[:, code]))
                                     if 0 < binary.sum() < len(binary) else float("nan"))
    if len(present) == len(classes):
        scores["auc_ovr_weighted"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted",
                                                         labels=list(range(len(classes)))))
    else:
        scores["auc_ovr_weighted"] = float("nan")
    return scores


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metricas globales de clasificacion (ponderadas por soporte, salvo ``f1_macro``)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> pd.DataFrame:
    """Precision, recall, F1 y soporte de cada clase, en el orden de ``classes``."""
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": list(classes), "precision": precision, "recall": recall, "f1": f1,
                         "support": support})


def evaluate_model(model: Any, data: Dataset) -> EvaluationResult:
    """Aplica un modelo ya ajustado a ``data`` y calcula todas las metricas.

    Raises:
        ValueError: si el orden de clases del modelo no es 0..K-1 (el de ``data.classes``).
    """
    expected = list(range(len(data.classes)))
    if list(model.classes_) != expected:
        raise ValueError(f"Orden de clases del modelo {list(model.classes_)} != {expected}")
    started = time.perf_counter()
    y_pred = model.predict(data.X)
    elapsed = time.perf_counter() - started
    proba = model.predict_proba(data.X)
    metrics = {**classification_metrics(data.y, y_pred), **auc_scores(data.y, proba, data.classes)}
    return EvaluationResult(
        metrics=metrics,
        report=per_class_report(data.y, y_pred, data.classes),
        confusion=confusion_matrix(data.y, y_pred, labels=expected),
        y_pred=y_pred,
        proba=proba,
        inference_time_seconds=elapsed,
    )


def metrics_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """F1 ponderada y accuracy dentro de cada grupo (p. ej. equipo de grabacion).

    Incluye la F1 de predecir siempre la clase mayoritaria del grupo, como referencia.
    """
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        majority = np.bincount(y_true[mask]).argmax()
        rows.append({
            "group": group,
            "n_cycles": int(mask.sum()),
            "f1_weighted": float(f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "f1_weighted_majority_baseline": float(f1_score(y_true[mask], np.full(mask.sum(), majority),
                                                            average="weighted", zero_division=0)),
        })
    return pd.DataFrame(rows)


def plot_confusion_matrix(confusion: np.ndarray, classes: tuple[str, ...], title: str, path: Path) -> Path:
    """Matriz de confusion con conteos y porcentaje por fila (recall)."""
    row_totals = confusion.sum(axis=1, keepdims=True)
    share = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=float), where=row_totals > 0)
    fig = Figure(figsize=(5.6, 4.9), layout="constrained")
    ax = fig.add_subplot()
    image = ax.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{_fmt_int(int(confusion[i, j]))}\n{100 * share[i, j]:.1f} %".replace(".", ","),
                    ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.55 else INK_PRIMARY)
    ax.set_xticks(range(len(classes)), classes, fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=9)
    ax.set_xlabel("Clase predicha", color=INK_SECONDARY)
    ax.set_ylabel("Clase real", color=INK_SECONDARY)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Fracción de la clase real", format=DECIMAL_COMMA)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    return path


def plot_roc_curves(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...], title: str, path: Path) -> Path:
    """Curvas ROC One-vs-Rest de cada clase con su AUC."""
    fig = Figure(figsize=(5.6, 4.9), layout="constrained")
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1.2)
    for code, name in enumerate(classes):
        binary = (y_true == code).astype(int)
        if not 0 < binary.sum() < len(binary):
            continue
        fpr, tpr, _ = roc_curve(binary, proba[:, code])
        auc = roc_auc_score(binary, proba[:, code])
        ax.plot(fpr, tpr, color=CLASS_COLORS[code % 4], linestyle=CLASS_STYLES[code % 4], linewidth=2,
                label=f"{name} vs resto (AUC = {auc:.3f})".replace(".", ","))
    ax.set_xlabel("Tasa de falsos positivos", color=INK_SECONDARY)
    ax.set_ylabel("Tasa de verdaderos positivos", color=INK_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA)
    ax.yaxis.set_major_formatter(DECIMAL_COMMA)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    return path


def plot_test_comparison(table: pd.DataFrame, target: float, baseline: float | None, title: str, path: Path) -> Path:
    """F1 ponderada en test (marcador lleno) y en CV (marcador hueco) de cada configuracion.

    ``table`` debe traer ``representation, classifier, variant, cv_f1_weighted,
    test_f1_weighted`` y, opcionalmente, ``designated`` (bool).
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415

    ranked = table.sort_values("test_f1_weighted", ascending=True).reset_index(drop=True)
    fig = Figure(figsize=(8.6, 0.36 * len(ranked) + 2.0), layout="constrained")
    ax = fig.add_subplot()
    for y, row in ranked.iterrows():
        color = COLOR_SERIES_2 if bool(row.get("designated", False)) else COLOR_INCLUDED
        ax.plot([row["cv_f1_weighted"], row["test_f1_weighted"]], [y, y], color=GRID_COLOR, linewidth=2, zorder=1)
        ax.scatter([row["cv_f1_weighted"]], [y], s=42, facecolor="white", edgecolor=INK_SECONDARY, linewidth=1.2,
                   zorder=2)
        ax.scatter([row["test_f1_weighted"]], [y], s=52, color=color, zorder=3)
        ax.annotate(f"{row['test_f1_weighted']:.3f}".replace(".", ","),
                    (max(row["test_f1_weighted"], row["cv_f1_weighted"]), y), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax.set_yticks(range(len(ranked)), [improvement_label(r.representation, r.classifier, r.variant)
                                       for r in ranked.itertuples()], fontsize=9)
    ax.axvline(target, color=INK_PRIMARY, linestyle="--", linewidth=1)
    if baseline is not None:
        ax.axvline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    low = min(float(ranked[["cv_f1_weighted", "test_f1_weighted"]].min().min()), baseline or 1.0) - 0.03
    ax.set_xlim(max(0.0, low), max(target, float(ranked[["cv_f1_weighted", "test_f1_weighted"]].max().max())) + 0.05)
    ax.set_xlabel("F1 ponderada", color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_INCLUDED, markersize=7, label="Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_SERIES_2, markersize=7,
               label="Test, modelo designado por CV"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK_SECONDARY,
               markersize=7, label="Validación cruzada (train)"),
        Line2D([0], [0], color=INK_PRIMARY, linestyle="--", label=f"Meta ({target:.2f})".replace(".", ",")),
    ]
    if baseline is not None:
        handles.append(Line2D([0], [0], color=INK_SECONDARY, linestyle=":",
                              label=f"Clase mayoritaria en test ({baseline:.3f})".replace(".", ",")))
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path
