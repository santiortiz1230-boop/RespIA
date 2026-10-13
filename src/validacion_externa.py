"""Validacion externa con el dataset de Fraiwan et al. (2021) (Fase 10).

Fraiwan da una etiqueta por grabacion completa y no trae tiempos de ciclo, mientras
que los modelos clasifican un ciclo respiratorio. La estrategia (aprobada y fijada en
``config.yaml`` antes de evaluar) es por grabacion::

    grabacion -> ventanas consecutivas de ``window_sec`` sin solape (se descarta el
    resto final) -> preprocesamiento + caracteristicas + modelo congelado en cada
    ventana -> clase de la grabacion = la mas votada (empate: mayor probabilidad media)

Reglas que este modulo respeta:

* Fraiwan no se usa para ajustar nada: ni escalador, ni clasificador, ni umbrales.
* La equivalencia de etiquetas y las exclusiones se leen de ``config.yaml``; una
  etiqueta que no este ni en ``label_map`` ni en ``excluded`` detiene la ejecucion.
* Los tres modos de filtro son la misma grabacion: se evaluan por separado, sin mezclar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, f1_score

from src.auditoria import COLOR_INCLUDED, COLOR_SERIES_2, GRID_COLOR, INK_PRIMARY, INK_SECONDARY, _style_axes
from src.configuracion import Config
from src.evaluacion import auc_scores, classification_metrics, per_class_report, plot_confusion_matrix
from src.fraiwan import FILTER_MODES, build_fraiwan_metadata
from src.robustez import (
    CLEAN,
    MODEL_FILES,
    MODEL_LABELS,
    TWO_DECIMALS,
    Condition,
    _sha256,
    extract_condition,
    feature_columns,
    load_frozen_models,
)
from src.utilidades import environment_info, save_dataframe, save_json, to_posix_relative

LIBRARIES = ("numpy", "pandas", "scipy", "librosa", "scikit-learn", "joblib", "soundfile")
FIGURE_NAMES = {"best_model": "mejor_modelo", "prototype": "prototipo"}


class LabelMappingError(ValueError):
    """Hay etiquetas de Fraiwan sin equivalencia ni exclusion declarada en ``config.yaml``."""


# Etiquetas, ventanas y voto


def map_labels(recordings: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Anade ``label`` (clase del proyecto o vacio) y ``exclusion_reason`` a la tabla de grabaciones.

    Raises:
        LabelMappingError: alguna etiqueta no esta ni en ``label_map`` ni en ``excluded``.
    """
    external = cfg.external
    unknown = sorted(set(recordings["sound_type"]) - set(external.label_map) - set(external.excluded))
    if unknown:
        raise LabelMappingError(f"Etiquetas de Fraiwan sin equivalencia declarada: {unknown}. "
                                "No se inventa una equivalencia: declare cada una en config.yaml.")
    table = recordings.copy()
    table["label"] = table["sound_type"].map(external.label_map).fillna("")
    table["exclusion_reason"] = table["sound_type"].map(external.excluded).fillna("")
    return table


def split_windows(audio: np.ndarray, sample_rate: int, window_sec: float) -> list[np.ndarray]:
    """Ventanas consecutivas sin solape de ``window_sec``; el resto final incompleto se descarta."""
    size = int(round(window_sec * sample_rate))
    if size <= 0:
        raise ValueError("La ventana debe tener al menos una muestra.")
    return [audio[start:start + size] for start in range(0, len(audio) - size + 1, size)]


def aggregate_recording(predictions: np.ndarray, probabilities: np.ndarray, n_classes: int) -> tuple[int, np.ndarray]:
    """Clase mas votada entre las ventanas de una grabacion y probabilidad media por clase.

    En caso de empate gana, entre las clases empatadas, la de mayor probabilidad media.
    """
    counts = np.bincount(np.asarray(predictions, dtype=int), minlength=n_classes)
    mean_proba = np.asarray(probabilities, dtype=float).mean(axis=0)
    tied = np.flatnonzero(counts == counts.max())
    winner = int(tied[np.argmax(mean_proba[tied])])
    return winner, mean_proba


# Extraccion (un trabajador por grabacion)


def _process_recording(row: dict[str, Any], dataset_dir: Path, cfg: Config) -> list[dict[str, Any]]:
    """Lee una grabacion, la corta en ventanas y extrae el vector completo de cada una."""
    audio, sample_rate = sf.read(str(dataset_dir / row["source_audio"]), dtype="float32", always_2d=False)
    audio = audio if audio.ndim == 1 else audio.mean(axis=1)
    records = []
    for i, window in enumerate(split_windows(audio, int(sample_rate), cfg.external.window_sec)):
        vector, _ = extract_condition(window, int(sample_rate), f"{row['recording_id']}_w{i:02d}",
                                      Condition(CLEAN, None, 0), cfg)
        records.append({"recording_id": row["recording_id"], "window": i, "vector": vector})
    return records


def extract_windows(recordings: pd.DataFrame, cfg: Config, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indice de ventanas (``recording_id, window``) y su matriz de caracteristicas, en el mismo orden."""
    batches = Parallel(n_jobs=n_jobs)(delayed(_process_recording)(row, cfg.paths.raw_fraiwan_dir, cfg)
                                      for row in recordings.to_dict(orient="records"))
    records = sorted((r for batch in batches for r in batch), key=lambda r: (r["recording_id"], r["window"]))
    index = pd.DataFrame({"recording_id": [r["recording_id"] for r in records],
                          "window": [r["window"] for r in records]})
    matrix = pd.DataFrame(np.vstack([r["vector"] for r in records]), columns=feature_columns(cfg))
    if not np.all(np.isfinite(matrix.to_numpy())):
        raise RuntimeError("Hay caracteristicas no finitas en las ventanas de Fraiwan.")
    return index, matrix


# Metricas


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> dict[str, Any]:
    """Las mismas metricas de la evaluacion de ICBHI, mas el recall y el numero predicho de cada clase."""
    row: dict[str, Any] = {"n": int(len(y_true)), **classification_metrics(y_true, y_pred),
                           **auc_scores(y_true, proba, classes)}
    report = per_class_report(y_true, y_pred, classes)
    predicted = np.bincount(y_pred, minlength=len(classes))
    for code, record in enumerate(report.itertuples(index=False)):
        row[f"recall_{record[0]}"] = float(record.recall)
        row[f"precision_{record[0]}"] = float(record.precision)
        row[f"n_{record[0]}"] = int(record.support)
        row[f"n_pred_{record[0]}"] = int(predicted[code])
    return row


def generalization_loss(f1_internal: float, f1_external: float) -> tuple[float, float]:
    """Perdida absoluta (``F1_ICBHI - F1_externa``) y relativa en % respecto a la F1 de ICBHI."""
    absolute = f1_internal - f1_external
    return absolute, 100.0 * absolute / f1_internal if f1_internal > 0 else float("nan")


# Figura


def plot_external_comparison(table: pd.DataFrame, path: Path) -> Path:
    """F1 ponderada de cada modelo en el test de ICBHI y en Fraiwan (un grupo por modo de filtro).

    ``table``: columnas ``grupo, model, f1_weighted, baseline_f1``; los grupos se dibujan en el orden dado.
    """
    groups = list(dict.fromkeys(table["grupo"]))
    models = [m for m in MODEL_LABELS if m in set(table["model"])]
    colors = {"best_model": COLOR_INCLUDED, "prototype": COLOR_SERIES_2}
    fig = Figure(figsize=(8.6, 4.6), layout="constrained")
    ax = fig.add_subplot()
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        rows = table[table["model"] == model].set_index("grupo").loc[groups]
        x = np.arange(len(groups)) + (k - (len(models) - 1) / 2) * width
        ax.bar(x, rows["f1_weighted"], width=width * 0.9, color=colors[model], label=MODEL_LABELS[model])
        for xi, value in zip(x, rows["f1_weighted"]):
            ax.annotate(f"{value:.3f}".replace(".", ","), (xi, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_SECONDARY)
    baselines = table.drop_duplicates("grupo").set_index("grupo").loc[groups, "baseline_f1"]
    for x, value in enumerate(baselines):
        ax.plot([x - 0.44, x + 0.44], [value, value], color=INK_PRIMARY, linestyle=":", linewidth=1.4,
                label="Predecir siempre NORMAL" if x == 0 else None)
    ax.set_xticks(range(len(groups)), groups, fontsize=9)
    ax.set_ylim(0, max(0.7, float(table["f1_weighted"].max()) + 0.1))
    ax.set_ylabel("F1 ponderada", color=INK_SECONDARY, fontsize=9.5)
    ax.yaxis.set_major_formatter(TWO_DECIMALS)
    _style_axes(ax)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    fig.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=8.5)
    fig.suptitle("Validación externa: F1 ponderada en el test de ICBHI y en Fraiwan (por grabación)", x=0.01,
                 ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Orquestacion


def run_external_validation(cfg: Config, n_jobs: int | None = None) -> dict[str, Any]:
    """Aplica los modelos congelados a Fraiwan y compara con el test de ICBHI."""
    paths = cfg.paths
    out = paths.results_dir / "external"
    classes = tuple(cfg.labels.modes[cfg.labels.label_mode])
    code_of = {name: i for i, name in enumerate(classes)}
    models = load_frozen_models(cfg)
    hashes_before = {name: _sha256(paths.model_dir / MODEL_FILES[name]) for name in models}

    recordings = map_labels(build_fraiwan_metadata(paths.raw_fraiwan_dir), cfg)
    included = recordings[recordings["label"] != ""].reset_index(drop=True)
    exclusions = (recordings[recordings["label"] == ""].drop_duplicates("patient_id")
                  .groupby(["sound_type", "exclusion_reason"]).size().reset_index(name="n_pacientes"))
    jobs = cfg.external.n_jobs if n_jobs is None else n_jobs
    index, matrix = extract_windows(included, cfg, jobs)
    windows = index.merge(included[["recording_id", "patient_id", "filter_code", "filter_mode", "sound_type",
                                    "label"]], on="recording_id", validate="many_to_one")
    y_window = windows["label"].map(code_of).to_numpy()

    result_rows, recording_tables, reports = [], [], []
    internal = {
        "best_model": pd.read_csv(paths.results_dir / "resultados_12_modelos.csv").query("designated_by_cv").iloc[0],
        "prototype": pd.read_csv(paths.results_dir / "improvements" / "resultados_mejoras_prueba.csv").query(
            "designated_by_cv").iloc[0] if "prototype" in models else None,
    }
    icbhi_baseline = float(json.loads((paths.results_dir / "resultados_mejor_modelo.json").read_text(encoding="utf-8"))[
        "test_majority_baseline_f1_weighted"])
    for model_name, (model, feature_list, meta) in models.items():
        reference = internal[model_name]
        result_rows.append({
            "dataset": "ICBHI test", "model": model_name, "filter_code": "", "filter_mode": "", "level": "ciclo",
            "n": int(reference["n_test_samples"]), "accuracy": float(reference["test_accuracy"]),
            "precision_weighted": float(reference["test_precision_weighted"]),
            "recall_weighted": float(reference["test_recall_weighted"]),
            "f1_weighted": float(reference["test_f1_weighted"]), "f1_macro": float(reference["test_f1_macro"]),
            "auc_ovr_weighted": float(reference["test_auc_roc_weighted"]),
            **{f"recall_{c}": float(reference[f"test_recall_{c}"]) for c in classes},
            "baseline_always_normal_f1": icbhi_baseline, "generalization_loss_f1": 0.0,
            "generalization_loss_f1_pct": 0.0,
        })
        window_pred = model.predict(matrix[feature_list])
        window_proba = model.predict_proba(matrix[feature_list])
        windows[f"pred_{model_name}"] = window_pred
        for code in "BDE":
            mask = (windows["filter_code"] == code).to_numpy()
            grouped = windows[mask].groupby("recording_id", sort=True)
            rows = []
            for recording_id, group in grouped:
                positions = np.flatnonzero(mask)[grouped.indices[recording_id]]
                winner, mean_proba = aggregate_recording(window_pred[positions], window_proba[positions],
                                                         len(classes))
                rows.append({"recording_id": recording_id, "patient_id": group["patient_id"].iloc[0],
                             "filter_code": code, "sound_type": group["sound_type"].iloc[0],
                             "y_true": code_of[group["label"].iloc[0]], "y_pred": winner,
                             "n_windows": len(group), "model": model_name,
                             **{f"proba_{c}": float(mean_proba[i]) for i, c in enumerate(classes)},
                             **{f"votes_{c}": int(np.sum(window_pred[positions] == i))
                                for i, c in enumerate(classes)}})
            per_recording = pd.DataFrame(rows)
            recording_tables.append(per_recording)
            y_true, y_pred = per_recording["y_true"].to_numpy(), per_recording["y_pred"].to_numpy()
            proba = per_recording[[f"proba_{c}" for c in classes]].to_numpy()
            baseline = float(f1_score(y_true, np.zeros_like(y_true), average="weighted", zero_division=0))
            for level, (yt, yp, pr) in {"grabacion": (y_true, y_pred, proba),
                                        "ventana": (y_window[mask], window_pred[mask], window_proba[mask])}.items():
                metrics = metric_row(yt, yp, pr, classes)
                loss = generalization_loss(float(reference["test_f1_weighted"]), metrics["f1_weighted"])
                result_rows.append({
                    "dataset": "Fraiwan", "model": model_name, "filter_code": code,
                    "filter_mode": FILTER_MODES[code], "level": level, **metrics,
                    "baseline_always_normal_f1": baseline if level == "grabacion" else float(
                        f1_score(yt, np.zeros_like(yt), average="weighted", zero_division=0)),
                    "generalization_loss_f1": loss[0], "generalization_loss_f1_pct": loss[1],
                })
            report = per_class_report(y_true, y_pred, classes).assign(model=model_name, filter_code=code)
            reports.append(report)
            if code == cfg.external.primary_filter:
                label = f"{MODEL_LABELS[model_name]}\nFraiwan, modo {FILTER_MODES[code]}, por grabación"
                f1 = f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.3f}".replace(".", ",")
                plot_confusion_matrix(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))), classes,
                                      f"Matriz de confusión (validación externa)\n{label}\nF1 ponderada = {f1}",
                                      paths.figures_dir / "confusion_matrices" / f"externa__{FIGURE_NAMES[model_name]}.png")

    results = pd.DataFrame(result_rows)
    results["primary"] = (results["dataset"] == "Fraiwan") & (results["level"] == "grabacion") & (
        results["filter_code"] == cfg.external.primary_filter)
    per_recording_all = pd.concat(recording_tables, ignore_index=True)
    plot_rows = [{"grupo": "ICBHI test\n(por ciclo)", "model": r.model, "f1_weighted": r.f1_weighted,
                  "baseline_f1": r.baseline_always_normal_f1} for r in results[results["dataset"] == "ICBHI test"].itertuples()]
    order = [cfg.external.primary_filter] + [c for c in "DEB" if c != cfg.external.primary_filter]
    for code in order:
        for r in results[(results["filter_code"] == code) & (results["level"] == "grabacion")].itertuples():
            plot_rows.append({"grupo": f"Fraiwan · {FILTER_MODES[code]}" + ("\n(principal)" if code == order[0] else ""),
                              "model": r.model, "f1_weighted": r.f1_weighted, "baseline_f1": r.baseline_always_normal_f1})
    outputs = {
        "results": save_dataframe(results, paths.results_dir / "resultados_validacion_externa.csv"),
        "recordings": save_dataframe(per_recording_all, out / "predicciones_por_grabacion.csv"),
        "reports": save_dataframe(pd.concat(reports, ignore_index=True), out / "reportes_clasificacion.csv"),
        "exclusions": save_dataframe(exclusions, out / "exclusiones.csv"),
        "labels": save_dataframe(
            included.drop_duplicates("patient_id").groupby(["sound_type", "label"]).size().reset_index(
                name="n_pacientes"), out / "equivalencia_etiquetas_aplicada.csv"),
        "figure": plot_external_comparison(pd.DataFrame(plot_rows), paths.figures_dir / "validacion_externa_f1.png"),
    }
    out.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out / "predicciones_por_ventana.parquet", index=False)
    unchanged = all(_sha256(paths.model_dir / MODEL_FILES[name]) == digest for name, digest in hashes_before.items())
    checks = {
        "n_patients_total": int(recordings["patient_id"].nunique()),
        "n_patients_included": int(included["patient_id"].nunique()),
        "n_patients_excluded": int(recordings["patient_id"].nunique() - included["patient_id"].nunique()),
        "n_recordings_evaluated": int(len(included)),
        "n_windows": int(len(windows)),
        "windows_per_recording_min": int(windows.groupby("recording_id").size().min()),
        "windows_per_recording_median": float(windows.groupby("recording_id").size().median()),
        "model_files_unchanged": bool(unchanged),
    }
    save_json({
        "fase": "10 - validacion externa (Fraiwan)", "level": "grabacion (aproximada)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": cfg.external.as_dict(), "checks": checks,
        "models": {name: f"{meta['variant']}__{meta['representation']}__{meta['classifier']}"
                   for name, (_, _, meta) in models.items()},
        "outputs": {k: to_posix_relative(v, paths.root) for k, v in outputs.items()},
        **environment_info(LIBRARIES),
    }, paths.audit_dir / "ejecucion_validacion_externa.json")
    return {"results": results, "recordings": per_recording_all, "exclusions": exclusions, "checks": checks,
            "outputs": outputs}
