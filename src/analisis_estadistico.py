"""Analisis estadistico por bootstrap agrupado por paciente (Fase 11).

Por que agrupar: los ciclos de un mismo paciente se parecen entre si, asi que no son
observaciones independientes. Si se remuestrearan ciclos sueltos, el intervalo saldria
demasiado estrecho. Aqui la unidad que se sortea es el paciente: en cada iteracion se
eligen ``P`` pacientes con reemplazo (de los ``P`` del conjunto) y se usan todos los
ciclos de cada uno, tantas veces como haya salido.

Calculo: las metricas de clasificacion dependen solo de la matriz de confusion, y la
matriz de una replica es la suma de las matrices de los pacientes sorteados. Se guarda
una matriz de confusion por paciente y cada replica es un producto
``multiplicidades (B x P) @ matrices (P x K*K)``; es exacto y muy rapido.

Las comparaciones entre modelos son pareadas: todos usan las mismas multiplicidades,
de modo que la diferencia de F1 de cada replica se calcula sobre los mismos pacientes.

Este modulo trabaja con las predicciones por ciclo guardadas en la Fase 7: no vuelve
a leer las caracteristicas del conjunto de prueba ni aplica modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

from src.auditoria import COLOR_INCLUDED, COLOR_SERIES_2, GRID_COLOR, INK_PRIMARY, INK_SECONDARY, _style_axes
from src.robustez import TWO_DECIMALS


def patient_confusions(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                       n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de confusion de cada paciente: ``(pacientes ordenados, array P x K x K)``."""
    patients, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    confusions = np.zeros((len(patients), n_classes, n_classes), dtype=np.int64)
    np.add.at(confusions, (codes, np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return patients, confusions


def bootstrap_multiplicities(n_groups: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Veces que sale cada grupo en cada replica: array ``B x P`` cuyas filas suman ``P``."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_bootstrap)


def replicate_confusions(multiplicities: np.ndarray, confusions: np.ndarray) -> np.ndarray:
    """Matriz de confusion de cada replica (``B x K x K``) a partir de las matrices por paciente."""
    n_groups, k, _ = confusions.shape
    flat = multiplicities.astype(np.float64) @ confusions.reshape(n_groups, k * k).astype(np.float64)
    return flat.reshape(-1, k, k)


def weighted_f1(confusion: np.ndarray) -> np.ndarray:
    """F1 ponderada por soporte de una o varias matrices de confusion (ultimas dos dimensiones K x K).

    Coincide con ``sklearn.metrics.f1_score(average="weighted", zero_division=0)``.
    """
    c = np.asarray(confusion, dtype=np.float64)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    support = c.sum(axis=-1)
    predicted = c.sum(axis=-2)
    denominator = support + predicted
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    total = support.sum(axis=-1)
    return np.divide((support * f1).sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)


def macro_f1(confusion: np.ndarray) -> np.ndarray:
    """F1 macro sobre las K clases fijas (una clase sin soporte ni predicciones aporta 0)."""
    c = np.asarray(confusion, dtype=np.float64)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denominator = c.sum(axis=-1) + c.sum(axis=-2)
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean(axis=-1)


def accuracy(confusion: np.ndarray) -> np.ndarray:
    """Accuracy de una o varias matrices de confusion."""
    c = np.asarray(confusion, dtype=np.float64)
    total = c.sum(axis=(-2, -1))
    return np.divide(np.trace(c, axis1=-2, axis2=-1), total, out=np.zeros_like(total), where=total > 0)


def weighted_ovr_auc(y_true: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """AUC-ROC One-vs-Rest ponderada por soporte; admite pesos por muestra (multiplicidad del bootstrap).

    Devuelve NaN si alguna clase queda sin muestras (o es la unica) con los pesos dados.
    """
    weights = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    keep = weights > 0
    y, p, w = np.asarray(y_true)[keep], np.asarray(proba)[keep], weights[keep]
    total, aucs, supports = w.sum(), [], []
    for code in range(p.shape[1]):
        positive = y == code
        support = w[positive].sum()
        if support <= 0 or support >= total:
            return float("nan")
        aucs.append(roc_auc_score(positive.astype(int), p[:, code], sample_weight=w))
        supports.append(support)
    return float(np.average(aucs, weights=supports))


@dataclass
class Interval:
    """Estimacion puntual e intervalo por percentiles de una distribucion bootstrap."""

    point: float
    mean: float
    standard_error: float
    low: float
    high: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Campos con prefijo opcional, listos para una fila de tabla."""
        return {f"{prefix}point": self.point, f"{prefix}boot_mean": self.mean,
                f"{prefix}boot_se": self.standard_error, f"{prefix}ci_low": self.low, f"{prefix}ci_high": self.high}


def percentile_interval(values: np.ndarray, point: float, confidence: float) -> Interval:
    """Intervalo de confianza por percentiles (ignora replicas no definidas)."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(clean, [alpha, 1.0 - alpha])
    return Interval(point=float(point), mean=float(clean.mean()), standard_error=float(clean.std(ddof=1)),
                    low=float(low), high=float(high))


def two_sided_p_value(differences: np.ndarray) -> float:
    """Valor p bilateral del bootstrap para "la diferencia es 0": 2 x la menor de las dos colas (con +1)."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    below = (np.sum(d <= 0) + 1) / (len(d) + 1)
    above = (np.sum(d >= 0) + 1) / (len(d) + 1)
    return float(min(1.0, 2.0 * min(below, above)))


def paired_comparison(values_a: np.ndarray, values_b: np.ndarray, point_a: float, point_b: float,
                      confidence: float) -> dict[str, float]:
    """Diferencia pareada ``A - B`` replica a replica: intervalo, fraccion de replicas con A > B y valor p."""
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    interval = percentile_interval(differences, point_a - point_b, confidence)
    return {"diff_point": interval.point, "diff_ci_low": interval.low, "diff_ci_high": interval.high,
            "diff_boot_se": interval.standard_error,
            "frac_replicas_a_mayor": float(np.mean(differences > 0)),
            "p_value": two_sided_p_value(differences),
            "ci_excluye_cero": bool(interval.low > 0 or interval.high < 0)}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Correccion de Holm para comparaciones multiples (devuelve los p ajustados en el orden original)."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[position])
        adjusted[position] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_auc(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray, multiplicities: np.ndarray,
                  patients: np.ndarray) -> np.ndarray:
    """AUC ponderada de cada replica usando la multiplicidad del paciente como peso de sus ciclos."""
    position = {patient: i for i, patient in enumerate(patients)}
    column = np.array([position[str(g)] for g in groups])
    return np.array([weighted_ovr_auc(y_true, proba, row[column]) for row in multiplicities])


def plot_intervals(table: pd.DataFrame, baseline: float, target: float, confidence: float, path: Path) -> Path:
    """F1 ponderada de cada modelo con su intervalo de confianza (una fila por modelo).

    ``table``: columnas ``label, point, ci_low, ci_high, highlight``; se dibuja en el orden dado (de abajo arriba).
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415

    fig = Figure(figsize=(8.6, 0.38 * len(table) + 2.1), layout="constrained")
    ax = fig.add_subplot()
    for y, row in enumerate(table.itertuples()):
        color = COLOR_SERIES_2 if row.highlight else COLOR_INCLUDED
        ax.plot([row.ci_low, row.ci_high], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([row.point], [y], s=46, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"{row.point:.3f}  [{row.ci_low:.3f}; {row.ci_high:.3f}]".replace(".", ","),
                    (row.ci_high, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_yticks(range(len(table)), table["label"], fontsize=9)
    ax.axvline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    ax.axvline(target, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.set_xlim(min(0.30, float(table["ci_low"].min()) - 0.03), max(target + 0.14, float(table["ci_high"].max()) + 0.2))
    ax.set_xlabel("F1 ponderada en test", color=INK_SECONDARY, fontsize=9.5)
    ax.xaxis.set_major_formatter(TWO_DECIMALS)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    handles = [
        Line2D([0], [0], color=COLOR_INCLUDED, marker="o", linewidth=2.2,
               label=f"Estimación e intervalo del {100 * confidence:.0f} %"),
        Line2D([0], [0], color=COLOR_SERIES_2, marker="o", linewidth=2.2, label="Modelos designados por CV"),
        Line2D([0], [0], color=INK_SECONDARY, linestyle=":", label=f"Clase mayoritaria ({baseline:.3f})".replace(".", ",")),
        Line2D([0], [0], color=INK_PRIMARY, linestyle="--", label=f"Meta ({target:.2f})".replace(".", ",")),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8.5)
    fig.suptitle("F1 ponderada con intervalo de confianza (bootstrap agrupado por paciente)", x=0.01, ha="left",
                 fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_differences(table: pd.DataFrame, title: str, confidence: float, path: Path) -> Path:
    """Diferencias pareadas de F1 con su intervalo; la linea vertical marca "sin diferencia".

    ``table``: columnas ``label, diff_point, diff_ci_low, diff_ci_high``.
    """
    fig = Figure(figsize=(8.6, 0.38 * len(table) + 1.9), layout="constrained")
    ax = fig.add_subplot()
    for y, row in enumerate(table.itertuples()):
        excludes = row.diff_ci_low > 0 or row.diff_ci_high < 0
        color = COLOR_SERIES_2 if excludes else COLOR_INCLUDED
        ax.plot([row.diff_ci_low, row.diff_ci_high], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter([row.diff_point], [y], s=46, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"{row.diff_point:+.3f}  [{row.diff_ci_low:+.3f}; {row.diff_ci_high:+.3f}]".replace(".", ","),
                    (row.diff_ci_high, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1)
    ax.set_yticks(range(len(table)), table["label"], fontsize=9)
    span = max(abs(float(table["diff_ci_low"].min())), abs(float(table["diff_ci_high"].max())))
    ax.set_xlim(-span - 0.02, span + 0.13)
    ax.set_xlabel(f"Diferencia de F1 ponderada, con intervalo del {100 * confidence:.0f} % "
                  "(en naranja si el intervalo no incluye el cero)", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.set_major_formatter(TWO_DECIMALS)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def summary_row(label: str, interval: Interval, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fila de tabla con la etiqueta, el intervalo y campos adicionales."""
    return {"label": label, **interval.as_dict(), **(extra or {})}
