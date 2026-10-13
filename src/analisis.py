"""Analisis del mejor resultado (Fase 8).

Trabaja con lo que la Fase 7 dejo guardado (tablas, metadatos y predicciones por
ciclo): no vuelve a cargar las caracteristicas del conjunto de prueba ni reajusta
los modelos congelados. Lo unico que se calcula de nuevo usa solo train:
separabilidad de las caracteristicas e importancia por permutacion dentro de la
validacion cruzada agrupada por paciente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.auditoria import COLOR_INCLUDED, COLOR_SERIES_2, GRID_COLOR, INK_PRIMARY, INK_SECONDARY, _style_axes
from src.configuracion import Config
from src.constructor_datos import Dataset
from src.evaluacion import CLASS_COLORS, DECIMAL_COMMA
from src.modelos import build_pipeline
from src.entrenamiento import CVPlan

MAIN = "original"


def prediction_key(variant: str, representation: str, classifier: str) -> str:
    """Nombre de archivo de las predicciones de un modelo (el mismo de la Fase 7)."""
    return f"{variant}__{representation}__{classifier}"


def load_predictions(cfg: Config, key: str) -> pd.DataFrame:
    """Predicciones por ciclo guardadas en la Fase 7 (no relee el test)."""
    return pd.read_parquet(cfg.paths.results_dir / "predictions" / f"{key}.parquet")


# Matriz de confusion y errores


def confusion_long(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> pd.DataFrame:
    """Matriz de confusion en formato largo, con porcentaje por clase real y por clase predicha."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rows = []
    for i, real in enumerate(classes):
        for j, predicted in enumerate(classes):
            rows.append({
                "clase_real": real, "clase_predicha": predicted, "n": int(matrix[i, j]),
                "pct_de_la_clase_real": 100.0 * matrix[i, j] / max(1, matrix[i].sum()),
                "pct_de_la_clase_predicha": 100.0 * matrix[i, j] / max(1, matrix[:, j].sum()),
            })
    return pd.DataFrame(rows)


def confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> pd.DataFrame:
    """Errores (celdas fuera de la diagonal) ordenados de mayor a menor numero de ciclos."""
    table = confusion_long(y_true, y_pred, classes)
    errors = table[table["clase_real"] != table["clase_predicha"]].copy()
    errors["pct_de_los_errores"] = 100.0 * errors["n"] / max(1, int(errors["n"].sum()))
    return errors.drop(columns="pct_de_la_clase_predicha").sort_values("n", ascending=False, ignore_index=True)


def prediction_balance(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> pd.DataFrame:
    """Cuantos ciclos hay de cada clase y cuantos predice el modelo (detecta clases que casi no predice)."""
    real = np.bincount(y_true, minlength=len(classes))
    predicted = np.bincount(y_pred, minlength=len(classes))
    return pd.DataFrame({"clase": list(classes), "n_real": real, "n_predicha": predicted,
                         "predicha_sobre_real": predicted / np.maximum(real, 1)})


# Train / CV / test


def generalization_table(results: pd.DataFrame, metas: dict[tuple[str, str, str], dict[str, Any]]) -> pd.DataFrame:
    """F1 ponderada en train, CV y test de cada modelo, con las brechas entre ellas.

    Args:
        results: tabla de la Fase 7 (``variant, representation, classifier, cv_f1_weighted, test_f1_weighted``).
        metas: metadatos de cada candidato, indexados por ``(variant, representation, classifier)``.
    """
    rows = []
    for row in results.itertuples():
        meta = metas[(row.variant, row.representation, row.classifier)]
        folds = [float(v) for v in meta["cv_fold_f1_weighted"]]
        cv, std, test = float(row.cv_f1_weighted), float(np.std(folds)), float(row.test_f1_weighted)
        train = float(meta["train_scores"]["train_f1_weighted"])
        rows.append({
            "variant": row.variant, "representation": row.representation, "classifier": row.classifier,
            "train_f1": train,
            "cv_fit_f1": float(meta["cv_scores"].get("cv_fit_f1_weighted", float("nan"))),
            "cv_f1": cv, "cv_f1_std": std, "cv_fold_min": min(folds), "cv_fold_max": max(folds),
            "test_f1": test,
            "gap_train_cv": train - cv,
            "gap_cv_test": cv - test,
            "test_dentro_del_rango_de_folds": bool(min(folds) <= test <= max(folds)),
            "test_menos_cv_en_desviaciones": (test - cv) / std if std > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# Desglose por grupos (equipo, paciente, duracion, zona del torax)


def group_report(predictions: pd.DataFrame, column: str, classes: tuple[str, ...]) -> pd.DataFrame:
    """Metricas dentro de cada valor de ``column`` (p. ej. equipo o paciente).

    Incluye la composicion por clase real, el recall de cada clase (NaN si la clase no
    aparece en el grupo) y la F1 de predecir siempre la clase mayoritaria del grupo.
    """
    labels = list(range(len(classes)))
    rows = []
    for value, group in predictions.groupby(column, sort=True):
        y_true, y_pred = group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        counts = np.bincount(y_true, minlength=len(classes))
        recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        row: dict[str, Any] = {
            column: value, "n_ciclos": len(group), "n_pacientes": int(group["patient_id"].nunique()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted_clase_mayoritaria": float(f1_score(y_true, np.full(len(group), counts.argmax()),
                                                             average="weighted", zero_division=0)),
        }
        for code, name in enumerate(classes):
            row[f"n_{name}"] = int(counts[code])
            row[f"recall_{name}"] = float(recalls[code]) if counts[code] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def duration_bins(durations: pd.Series, n_bins: int = 4) -> pd.Series:
    """Etiqueta cada ciclo con su cuartil de duracion (``"0,20-1,93 s"``...)."""
    edges = np.quantile(durations, np.linspace(0, 1, n_bins + 1))
    labels = [f"{low:.2f}-{high:.2f} s".replace(".", ",") for low, high in zip(edges[:-1], edges[1:])]
    return pd.cut(durations, bins=edges, labels=labels, include_lowest=True).astype(str)


def cycle_difficulty(predictions: dict[str, pd.DataFrame], classes: tuple[str, ...]) -> pd.DataFrame:
    """Para cada clase real: fraccion de ciclos que aciertan todos, ninguno o algunos de los modelos dados."""
    frames = list(predictions.values())
    reference = frames[0]
    for frame in frames[1:]:
        if not frame["cycle_id"].equals(reference["cycle_id"]):
            raise ValueError("Las predicciones no tienen las mismas filas.")
    hits = np.column_stack([(frame["y_pred"] == frame["y_true"]).to_numpy() for frame in frames]).sum(axis=1)
    y_true = reference["y_true"].to_numpy()
    rows = []
    for code, name in [(-1, "TODAS"), *enumerate(classes)]:
        mask = np.ones(len(y_true), dtype=bool) if code < 0 else y_true == code
        rows.append({
            "clase_real": name, "n_ciclos": int(mask.sum()), "n_modelos": len(frames),
            "pct_acierta_ningun_modelo": 100.0 * float(np.mean(hits[mask] == 0)),
            "pct_aciertan_todos": 100.0 * float(np.mean(hits[mask] == len(frames))),
            "modelos_que_aciertan_media": float(hits[mask].mean()),
        })
    return pd.DataFrame(rows)


# Solo train: que informacion llevan las caracteristicas


def feature_separability(data: Dataset) -> pd.DataFrame:
    """Cuanto separa cada caracteristica a las clases en train.

    ``eta2`` = varianza entre clases / varianza total (0 = no distingue nada, 1 = separa
    perfectamente). Las medias por clase se dan en desviaciones estandar de la caracteristica.
    """
    if data.split != "train":
        raise ValueError("La separabilidad se calcula solo con la particion de entrenamiento.")
    rows = []
    for column in data.X.columns:
        values = data.X[column].to_numpy(dtype=float)
        mean, std = values.mean(), values.std()
        z = (values - mean) / std if std > 0 else np.zeros_like(values)
        class_means = [float(z[data.y == code].mean()) for code in range(len(data.classes))]
        between = sum(float(np.sum(data.y == code)) * m**2 for code, m in enumerate(class_means))
        row: dict[str, Any] = {"feature": column, "eta2": between / len(z) if std > 0 else 0.0}
        for name, m in zip(data.classes, class_means):
            row[f"media_z_{name}"] = m
        rows.append(row)
    return pd.DataFrame(rows).sort_values("eta2", ascending=False, ignore_index=True)


def cv_permutation_importance(data: Dataset, classifier: str, params: dict[str, Any], cfg: Config, plan: CVPlan,
                              n_repeats: int = 20, overrides: dict[str, Any] | None = None) -> pd.DataFrame:
    """Caida de la F1 ponderada al desordenar cada caracteristica en la parte de validacion de cada fold.

    Usa solo train y los mismos folds agrupados por paciente; ``params`` son los
    hiperparametros elegidos (sin prefijo). Devuelve media y desviacion entre folds.
    """
    if data.split != "train":
        raise ValueError("La importancia por permutacion se calcula solo con la particion de entrenamiento.")
    rng = np.random.default_rng(cfg.random_state)
    prefixed = {f"classifier__{key}": value for key, value in params.items()}
    drops = np.zeros((len(plan.folds), data.X.shape[1]))
    base_scores = []
    for fold, (fit_idx, val_idx) in enumerate(plan.folds):
        model = build_pipeline(classifier, cfg, overrides=overrides).set_params(**prefixed)
        model.fit(data.X.iloc[fit_idx], data.y[fit_idx])
        validation = data.X.iloc[val_idx]
        base = f1_score(data.y[val_idx], model.predict(validation), average="weighted", zero_division=0)
        base_scores.append(base)
        for j, column in enumerate(data.X.columns):
            scores = []
            for _ in range(n_repeats):
                shuffled = validation.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                scores.append(f1_score(data.y[val_idx], model.predict(shuffled), average="weighted",
                                       zero_division=0))
            drops[fold, j] = base - float(np.mean(scores))
    return pd.DataFrame({
        "feature": list(data.X.columns),
        "caida_f1_media": drops.mean(axis=0),
        "caida_f1_std_entre_folds": drops.std(axis=0),
        "f1_cv_sin_permutar": float(np.mean(base_scores)),
    }).sort_values("caida_f1_media", ascending=False, ignore_index=True)


# Figuras


def plot_generalization(table: pd.DataFrame, labels: list[str], highlight: int, baseline: float, path: Path) -> Path:
    """F1 ponderada en train, validacion cruzada y test de cada combinacion (una fila por modelo)."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    order = np.argsort(table["test_f1"].to_numpy())
    fig = Figure(figsize=(8.6, 0.38 * len(table) + 2.1), layout="constrained")
    ax = fig.add_subplot()
    for y, i in enumerate(order):
        row = table.iloc[i]
        ax.plot([row["test_f1"], row["train_f1"]], [y, y], color=GRID_COLOR, linewidth=2, zorder=1)
        ax.scatter([row["train_f1"]], [y], s=44, marker="s", facecolor="white", edgecolor=INK_SECONDARY,
                   linewidth=1.2, zorder=2)
        ax.scatter([row["cv_f1"]], [y], s=44, facecolor="white", edgecolor=INK_SECONDARY, linewidth=1.2, zorder=3)
        ax.scatter([row["test_f1"]], [y], s=54, color=COLOR_SERIES_2 if i == highlight else COLOR_INCLUDED, zorder=4)
    ax.set_yticks(range(len(order)), [labels[i] for i in order], fontsize=9)
    ax.axvline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    ax.set_xlim(min(0.38, baseline - 0.03), 1.03)
    ax.set_xlabel("F1 ponderada", color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=INK_SECONDARY,
               markersize=7, label="Train (los mismos datos con que se ajustó)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK_SECONDARY,
               markersize=7, label="Validación cruzada"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_INCLUDED, markersize=7, label="Test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_SERIES_2, markersize=7,
               label="Test, modelo elegido"),
        Line2D([0], [0], color=INK_SECONDARY, linestyle=":",
               label=f"Clase mayoritaria en test ({baseline:.3f})".replace(".", ",")),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8.5)
    fig.suptitle("F1 ponderada en train, validación cruzada y test (12 combinaciones)", x=0.01, ha="left",
                 fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_class_recall(recalls: dict[str, list[float]], classes: tuple[str, ...], path: Path) -> Path:
    """Recall por clase en test de varios modelos (barras agrupadas por clase)."""
    fig = Figure(figsize=(8.0, 4.4), layout="constrained")
    ax = fig.add_subplot()
    width = 0.8 / len(recalls)
    for k, (name, values) in enumerate(recalls.items()):
        x = np.arange(len(classes)) + (k - (len(recalls) - 1) / 2) * width
        ax.bar(x, values, width=width * 0.92, color=CLASS_COLORS[k % len(CLASS_COLORS)], label=name)
        for xi, value in zip(x, values):
            ax.annotate(f"{100 * value:.0f} %", (xi, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(range(len(classes)), classes, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall en test (fracción de ciclos de la clase que se aciertan)", color=INK_SECONDARY,
                  fontsize=9)
    ax.yaxis.set_major_formatter(DECIMAL_COMMA)
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Recall por clase en el conjunto de prueba", loc="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_patient_accuracy(by_patient: pd.DataFrame, overall: float, path: Path) -> Path:
    """Accuracy del modelo en cada paciente de test (ordenados), con el numero de ciclos de cada uno."""
    ranked = by_patient.sort_values("accuracy", ignore_index=True)
    fig = Figure(figsize=(7.4, 0.21 * len(ranked) + 1.5), layout="constrained")
    ax = fig.add_subplot()
    ax.barh(range(len(ranked)), ranked["accuracy"], height=0.7, color=COLOR_INCLUDED)
    ax.axvline(overall, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.annotate(f"accuracy global = {overall:.3f}".replace(".", ","), (overall, 0.5), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK_PRIMARY,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax.set_yticks(range(len(ranked)), [f"{p} · {n} ciclos" for p, n in zip(ranked["patient_id"],
                                                                           ranked["n_ciclos"])], fontsize=7.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(ranked) - 0.3)
    ax.set_xlabel("Accuracy en los ciclos del paciente", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.set_major_formatter(DECIMAL_COMMA)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_title("Accuracy por paciente en el conjunto de prueba", loc="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path
