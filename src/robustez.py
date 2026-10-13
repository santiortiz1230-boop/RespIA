"""Robustez al ruido de los modelos congelados (Fase 9).

Flujo de cada ciclo de test, identico al de entrenamiento salvo por el ruido::

    segmento -> mono -> remuestreo a TARGET_SR -> [+ ruido blanco] -> pasa-banda
             -> normalizacion RMS -> caracteristicas -> modelo congelado

Definicion del SNR
------------------
``SNR_dB = 20 * log10(RMS_senal / RMS_ruido)``. Para lograrlo exactamente se usa el
RMS empirico de la realizacion de ruido ``w`` (no su desviacion teorica)::

    ganancia = RMS_senal / (RMS(w) * 10 ** (SNR_dB / 20))
    senal_con_ruido = senal + ganancia * w

Dos referencias para ``RMS_senal`` y ``RMS(w)`` (ambas fijadas antes de evaluar):

* ``fullband`` (METODOLOGIA ORIGINAL): se miden sobre la senal remuestreada completa.
* ``inband`` (ANALISIS COMPLEMENTARIO): se miden despues de pasar senal y ruido, por
  separado, por el mismo pasa-banda del preprocesamiento. Como el filtro es lineal,
  el SNR dentro de la banda util queda exactamente en el valor pedido.

En los dos casos el ruido se suma antes del filtro. Cada ciclo usa la misma realizacion
de ruido en todos los niveles de SNR de una repeticion (solo cambia la ganancia); la
semilla depende de ``random_state``, la repeticion y el ``cycle_id``, no del orden de
proceso. No se reentrena ni se ajusta nada: solo se aplican los modelos congelados.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from src.auditoria import COLOR_INCLUDED, COLOR_SERIES_2, GRID_COLOR, INK_PRIMARY, INK_SECONDARY, _style_axes
from src.configuracion import Config, PreprocessingConfig
from src.carga_datos import load_cycle_metadata
from src.constructor_datos import load_test_data, log_test_access
from src.descriptores import compute_descriptors, descriptor_names
from src.evaluacion import auc_scores, classification_metrics, per_class_report
from src.extraccion_caracteristicas import extract_representations, feature_names
from src.evaluacion_final import FREEZE_NAME, EvaluationSetupError
from src.preprocesamiento import apply_bandpass, design_bandpass, preprocess_audio, rms
from src.segmentacion import iter_cycle_segments
from src.utilidades import environment_info, save_dataframe, save_json, to_posix_relative

CLEAN = "clean"
PURPOSE = "Fase 9 - robustez al ruido de los modelos congelados"
LIBRARIES = ("numpy", "pandas", "scipy", "librosa", "scikit-learn", "joblib", "PyWavelets")
MODEL_FILES = {"best_model": "mejor_modelo.joblib", "prototype": "modelo_prototipo.joblib"}
MODEL_LABELS = {"best_model": "Modelo elegido (RMS + SVM)", "prototype": "Prototipo (voto, fusión + descriptores)"}
REFERENCE_LABELS = {"fullband": "SNR medido sobre la señal completa\n(metodología original)",
                    "inband": "SNR medido en la banda útil, 100-2000 Hz\n(análisis complementario)"}
TWO_DECIMALS = FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))


@dataclass(frozen=True)
class Condition:
    """Una condicion de evaluacion: limpia, o ruido con una referencia, un SNR y una repeticion."""

    reference: str            # "clean" | "fullband" | "inband"
    snr_db: float | None
    repetition: int

    @property
    def name(self) -> str:
        """Identificador unico (p. ej. ``fullband_20dB_rep0``)."""
        return CLEAN if self.reference == CLEAN else f"{self.reference}_{self.snr_db:g}dB_rep{self.repetition}"


def build_conditions(cfg: Config) -> list[Condition]:
    """Condicion limpia + todas las combinaciones referencia x SNR x repeticion de ``config.yaml``."""
    r = cfg.robustness
    return [Condition(CLEAN, None, 0)] + [Condition(reference, snr, repetition) for reference in r.references
                                           for snr in r.snr_db for repetition in range(r.repetitions)]


# Ruido


def white_noise(n_samples: int, cycle_id: str, base_seed: int, repetition: int) -> np.ndarray:
    """Ruido blanco gaussiano N(0, 1) reproducible, propio de cada ciclo y repeticion."""
    seed = np.random.SeedSequence([int(base_seed), int(repetition), zlib.crc32(str(cycle_id).encode("utf-8"))])
    return np.random.default_rng(seed).standard_normal(n_samples)


def noise_gain(signal_rms: float, noise_rms: float, snr_db: float) -> float:
    """Factor por el que hay que multiplicar el ruido para obtener exactamente ``snr_db``."""
    if signal_rms <= 0 or noise_rms <= 0:
        return 0.0
    return signal_rms / (noise_rms * 10.0 ** (snr_db / 20.0))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR en dB de ``signal`` frente a ``noise`` (NaN si alguno es nulo)."""
    s, n = rms(signal), rms(noise)
    return float(20.0 * np.log10(s / n)) if s > 0 and n > 0 else float("nan")


@dataclass
class NoisyCycle:
    """Senal con ruido y los SNR realmente obtenidos."""

    audio: np.ndarray
    gain: float
    snr_fullband_db: float
    snr_inband_db: float


def add_noise(signal: np.ndarray, noise: np.ndarray, snr: float, reference: str,
              pcfg: PreprocessingConfig) -> NoisyCycle:
    """Suma ``noise`` escalado a ``signal`` (ya remuestreada) para el SNR pedido en la referencia dada.

    Si la senal de referencia es nula no se anade ruido (ganancia 0 y SNR = NaN).
    """
    if reference not in ("fullband", "inband"):
        raise ValueError(f"Referencia de SNR desconocida: {reference}")
    sos = design_bandpass(pcfg.bandpass_low_hz, pcfg.bandpass_high_hz, pcfg.bandpass_order, pcfg.target_sr)
    signal_band = apply_bandpass(signal, sos, pcfg.zero_phase)
    noise_band = apply_bandpass(noise, sos, pcfg.zero_phase)
    if reference == "fullband":
        gain = noise_gain(rms(signal), rms(noise), snr)
    else:
        gain = noise_gain(rms(signal_band), rms(noise_band), snr)
    scaled = gain * noise
    return NoisyCycle(audio=signal + scaled, gain=gain, snr_fullband_db=snr_db(signal, scaled),
                      snr_inband_db=snr_db(signal_band, gain * noise_band))


# Extraccion (un trabajador por grabacion)


def feature_columns(cfg: Config) -> list[str]:
    """Columnas del vector completo: las 4 representaciones (en su orden) + los descriptores."""
    names = feature_names(cfg.features)
    return [c for rep in cfg.features.representations for c in names[rep]] + descriptor_names(cfg.descriptors)


def extract_condition(audio: np.ndarray, sample_rate: int, cycle_id: str, condition: Condition,
                      cfg: Config) -> tuple[np.ndarray, dict[str, float]]:
    """Vector completo de caracteristicas de un ciclo bajo una condicion, y los SNR obtenidos."""
    info: dict[str, float] = {"gain": 0.0, "snr_fullband_db": float("nan"), "snr_inband_db": float("nan")}
    hook = None
    if condition.reference != CLEAN:
        def hook(resampled: np.ndarray, _sr: int) -> np.ndarray:
            noise = white_noise(len(resampled), cycle_id, cfg.random_state, condition.repetition)
            noisy = add_noise(resampled, noise, float(condition.snr_db), condition.reference, cfg.preprocessing)
            info.update(gain=noisy.gain, snr_fullband_db=noisy.snr_fullband_db, snr_inband_db=noisy.snr_inband_db)
            return noisy.audio
    processed = preprocess_audio(audio, sample_rate, cfg.preprocessing, before_filter=hook)
    parts = extract_representations(processed.audio, cfg.features, processed.sample_rate)
    descriptors = compute_descriptors(processed.audio, processed.sample_rate, cfg.descriptors, cfg.features)
    vector = np.concatenate([parts[rep] for rep in cfg.features.representations] + [descriptors])
    return vector, info


def _process_recording(group: pd.DataFrame, dataset_dir: Path, cfg: Config,
                       conditions: list[Condition]) -> list[dict[str, Any]]:
    """Trabajador: lee una grabacion y extrae todos sus ciclos bajo todas las condiciones."""
    records = []
    for segment in iter_cycle_segments(group, dataset_dir, show_progress=False):
        for condition in conditions:
            vector, info = extract_condition(segment.audio, segment.sample_rate, segment.cycle_id, condition, cfg)
            records.append({"cycle_id": segment.cycle_id, "condition": condition.name, "vector": vector, **info})
    return records


def extract_all_conditions(cycles: pd.DataFrame, cycle_ids: np.ndarray, cfg: Config, conditions: list[Condition],
                           n_jobs: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Matrices de caracteristicas (filas en el orden de ``cycle_ids``) por condicion, y SNR por ciclo."""
    selected = cycles[cycles["cycle_id"].isin(set(cycle_ids))]
    groups = [group for _, group in selected.groupby("source_audio", sort=True)]
    batches = Parallel(n_jobs=n_jobs)(delayed(_process_recording)(g, cfg.paths.raw_icbhi_dir, cfg, conditions)
                                      for g in groups)
    records = [record for batch in batches for record in batch]
    columns = feature_columns(cfg)
    position = {cid: i for i, cid in enumerate(cycle_ids)}
    matrices = {c.name: np.full((len(cycle_ids), len(columns)), np.nan) for c in conditions}
    for record in records:
        matrices[record["condition"]][position[record["cycle_id"]]] = record["vector"]
    for name, matrix in matrices.items():
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Condicion {name}: faltan ciclos o hay valores no finitos.")
    snr = pd.DataFrame([{k: v for k, v in record.items() if k != "vector"} for record in records])
    return {name: pd.DataFrame(matrix, columns=columns) for name, matrix in matrices.items()}, snr


# Modelos congelados y metricas


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_models(cfg: Config) -> dict[str, tuple[Any, list[str], dict[str, Any]]]:
    """Carga BEST_MODEL y prototipo comprobando que son byte a byte los sellados antes del test."""
    manifest = json.loads((cfg.paths.results_dir / FREEZE_NAME).read_text(encoding="utf-8"))
    sealed = {"best_model": manifest["designated_main"], "prototype": manifest["designated_prototype"]}
    metadata_files = {"best_model": "metadatos_modelo.json", "prototype": "metadatos_modelo_prototipo.json"}
    models = {}
    for name, file_name in MODEL_FILES.items():
        path = cfg.paths.model_dir / file_name
        if sealed[name] is None or not path.is_file():
            continue
        if _sha256(path) != manifest["models"][sealed[name]]["sha256"]:
            raise EvaluationSetupError(f"{file_name} no coincide con el modelo sellado antes de abrir el test.")
        meta = json.loads((cfg.paths.model_dir / metadata_files[name]).read_text(encoding="utf-8"))
        models[name] = (joblib.load(path), list(meta["feature_names"]), meta)
    if "best_model" not in models:
        raise EvaluationSetupError("Falta models/mejor_modelo.joblib (ejecute la Fase 7).")
    return models


def condition_metrics(model: Any, X: pd.DataFrame, y: np.ndarray, classes: tuple[str, ...]) -> dict[str, Any]:
    """Metricas de un modelo congelado sobre una matriz de caracteristicas."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    metrics: dict[str, Any] = {**classification_metrics(y, y_pred), **auc_scores(y, proba, classes)}
    for record in per_class_report(y, y_pred, classes).itertuples(index=False):
        metrics[f"recall_{record[0]}"] = float(record.recall)
    counts = np.bincount(y_pred, minlength=len(classes))
    for code, name in enumerate(classes):
        metrics[f"n_pred_{name}"] = int(counts[code])
    metrics["y_pred"] = y_pred
    return metrics


def degradation(f1_clean: float, f1_noisy: float) -> tuple[float, float]:
    """Degradacion absoluta (``F1_limpio - F1_ruido``) y relativa en % respecto a ``F1_limpio``."""
    absolute = f1_clean - f1_noisy
    return absolute, 100.0 * absolute / f1_clean if f1_clean > 0 else float("nan")


def summarize(by_repetition: pd.DataFrame) -> pd.DataFrame:
    """Promedia las repeticiones de cada (modelo, referencia, SNR) y calcula la degradacion de la F1."""
    metric_columns = [c for c in by_repetition.columns
                      if c not in ("model", "reference", "snr_db", "repetition", "condition")]
    rows = []
    for (model, reference, snr), group in by_repetition.groupby(["model", "reference", "snr_db"], sort=False,
                                                                dropna=False):
        row: dict[str, Any] = {"model": model, "reference": reference, "snr_db": snr,
                               "n_repetitions": len(group)}
        for column in metric_columns:
            row[column] = float(group[column].mean())
        row["f1_weighted_std_reps"] = float(group["f1_weighted"].std(ddof=0))
        rows.append(row)
    table = pd.DataFrame(rows)
    clean = table[table["reference"] == CLEAN].set_index("model")["f1_weighted"]
    pairs = [degradation(float(clean[m]), float(f)) for m, f in zip(table["model"], table["f1_weighted"])]
    table["degradacion_absoluta_f1"] = [p[0] for p in pairs]
    table["degradacion_relativa_f1_pct"] = [p[1] for p in pairs]
    return table


# Figura


def plot_robustness(summary: pd.DataFrame, baseline: float, target: float, path: Path) -> Path:
    """F1 ponderada frente al nivel de ruido: un panel por referencia del SNR, una linea por modelo."""
    references = [r for r in ("fullband", "inband") if r in set(summary["reference"])]
    colors = {"best_model": COLOR_INCLUDED, "prototype": COLOR_SERIES_2}
    markers = {"best_model": "o", "prototype": "s"}
    fig = Figure(figsize=(4.6 * len(references) + 0.6, 4.6), layout="constrained")
    axes = fig.subplots(1, len(references), sharey=True, squeeze=False)[0]
    for ax, reference in zip(axes, references):
        levels = sorted(summary.loc[summary["reference"] == reference, "snr_db"].unique(), reverse=True)
        ticks = ["Sin ruido"] + [f"{snr:g} dB" for snr in levels]
        series = {}
        for model in [m for m in MODEL_LABELS if m in set(summary["model"])]:
            rows = summary[summary["model"] == model]
            series[model] = [float(rows.loc[rows["reference"] == CLEAN, "f1_weighted"].iloc[0])] + [
                float(rows.loc[(rows["reference"] == reference) & (rows["snr_db"] == snr), "f1_weighted"].iloc[0])
                for snr in levels]
        for model, values in series.items():
            ax.plot(range(len(values)), values, color=colors[model], marker=markers[model], markersize=7,
                    linewidth=2, markeredgecolor="white", markeredgewidth=1.5, label=MODEL_LABELS[model])
            for x, value in enumerate(values):  # la etiqueta del valor mas alto va arriba y la del mas bajo, abajo
                on_top = all(value >= other[x] for other in series.values())
                ax.annotate(f"{value:.3f}".replace(".", ","), (x, value), xytext=(0, 9 if on_top else -15),
                            textcoords="offset points", ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.axhline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=9.5)
        ax.set_xlim(-0.35, len(ticks) - 0.65)
        ax.set_xlabel("Nivel de ruido (SNR)", color=INK_SECONDARY, fontsize=9.5)
        ax.yaxis.set_major_formatter(TWO_DECIMALS)
        _style_axes(ax)
        ax.set_title(REFERENCE_LABELS[reference], loc="left", fontsize=10, color=INK_PRIMARY)
    axes[0].set_ylabel("F1 ponderada en test", color=INK_SECONDARY, fontsize=9.5)
    low = min(float(summary["f1_weighted"].min()), baseline) - 0.05
    axes[0].set_ylim(max(0.0, low), max(float(summary["f1_weighted"].max()) + 0.06, baseline + 0.05))
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D  # noqa: PLC0415
    handles.append(Line2D([0], [0], color=INK_SECONDARY, linestyle=":"))
    labels.append(f"Predecir siempre la clase mayoritaria ({baseline:.3f})".replace(".", ","))
    fig.legend(handles, labels, loc="outside lower center", ncols=len(handles), frameon=False, fontsize=8.5)
    fig.suptitle("Robustez al ruido blanco: F1 ponderada de los modelos congelados", x=0.01, ha="left",
                 fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Orquestacion


def run_robustness(cfg: Config, n_jobs: int | None = None) -> dict[str, Any]:
    """Evalua los modelos congelados con y sin ruido sobre los mismos ciclos de test."""
    paths = cfg.paths
    out = paths.results_dir / "robustness"
    models = load_frozen_models(cfg)
    conditions = build_conditions(cfg)
    test = load_test_data(cfg, "fusion_plus", PURPOSE)
    cycles = load_cycle_metadata(paths.processed_dir / "metadatos_ciclos.csv")
    jobs = cfg.robustness.n_jobs if n_jobs is None else n_jobs
    matrices, snr = extract_all_conditions(cycles, test.cycle_ids, cfg, conditions, jobs)
    log_test_access(paths.results_dir, f"audio crudo: limpio + {len(conditions) - 1} condiciones con ruido",
                    test.label_mode, test.n_samples, PURPOSE)

    # 1) La ruta "audio -> caracteristicas" sin ruido debe reproducir la cache y las predicciones de la Fase 7
    columns = feature_columns(cfg)
    if list(test.X.columns) != columns:
        raise RuntimeError("Las columnas del vector de robustez no coinciden con las de la cache.")
    clean_diff = float(np.max(np.abs(matrices[CLEAN].to_numpy() - test.X.to_numpy())))
    reproduced: dict[str, bool] = {}

    # 2) Metricas por condicion
    rows = []
    predictions = pd.DataFrame({"cycle_id": test.cycle_ids, "patient_id": test.groups, "y_true": test.y})
    for model_name, (model, feature_list, meta) in models.items():
        key = f"{meta['variant']}__{meta['representation']}__{meta['classifier']}"
        saved = pd.read_parquet(paths.results_dir / "predictions" / f"{key}.parquet")
        if not np.array_equal(saved["cycle_id"].to_numpy(), test.cycle_ids):
            raise RuntimeError("Las predicciones guardadas no tienen las mismas filas que el test.")
        for condition in conditions:
            metrics = condition_metrics(model, matrices[condition.name][feature_list], test.y, test.classes)
            y_pred = metrics.pop("y_pred")
            if condition.reference == CLEAN:
                reproduced[model_name] = bool(np.array_equal(y_pred, saved["y_pred"].to_numpy()))
            if condition.repetition == 0:
                predictions[f"{model_name}__{condition.name}"] = y_pred
            rows.append({"model": model_name, "reference": condition.reference, "snr_db": condition.snr_db,
                         "repetition": condition.repetition, "condition": condition.name, **metrics})
    by_repetition = pd.DataFrame(rows)
    summary = summarize(by_repetition)

    # 3) Comprobacion numerica del SNR
    noisy = snr[snr["condition"] != CLEAN].copy()
    parsed = {c.name: c for c in conditions}
    noisy["reference"] = noisy["condition"].map(lambda n: parsed[n].reference)
    noisy["snr_nominal_db"] = noisy["condition"].map(lambda n: parsed[n].snr_db)
    noisy["repetition"] = noisy["condition"].map(lambda n: parsed[n].repetition)
    noisy["snr_controlado_db"] = np.where(noisy["reference"] == "fullband", noisy["snr_fullband_db"],
                                          noisy["snr_inband_db"])
    noisy["error_db"] = (noisy["snr_controlado_db"] - noisy["snr_nominal_db"]).abs()
    snr_check = noisy.groupby(["reference", "snr_nominal_db"], sort=False).agg(
        n=("cycle_id", "size"), ciclos_sin_ruido=("gain", lambda g: int((g == 0).sum())),
        error_max_db=("error_db", "max"),
        snr_fullband_mediana_db=("snr_fullband_db", "median"),
        snr_inband_mediana_db=("snr_inband_db", "median"),
        snr_inband_p10_db=("snr_inband_db", lambda v: float(np.nanpercentile(v, 10))),
        snr_inband_p90_db=("snr_inband_db", lambda v: float(np.nanpercentile(v, 90))),
    ).reset_index()
    summary = summary.merge(
        snr_check.rename(columns={"snr_nominal_db": "snr_db"})[["reference", "snr_db", "snr_fullband_mediana_db",
                                                                 "snr_inband_mediana_db"]],
        on=["reference", "snr_db"], how="left")

    baseline = float(json.loads((paths.results_dir / "resultados_mejor_modelo.json").read_text(encoding="utf-8"))[
        "test_majority_baseline_f1_weighted"])
    outputs = {
        "robustness_results": save_dataframe(summary, paths.results_dir / "resultados_robustez.csv"),
        "by_repetition": save_dataframe(by_repetition, out / "robustez_por_repeticion.csv"),
        "snr_check": save_dataframe(snr_check, out / "comprobacion_snr.csv"),
        "figure": plot_robustness(summary, baseline, 0.80, paths.figures_dir / "robustez_f1.png"),
    }
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "predicciones_rep0.parquet", index=False)
    noisy.drop(columns=["error_db"]).to_parquet(out / "snr_por_ciclo.parquet", index=False)
    checks = {
        "clean_features_max_abs_diff_vs_cache": clean_diff,
        "clean_predictions_equal_phase7": reproduced,
        "snr_max_error_db": float(snr_check["error_max_db"].max()),
        "cycles_without_noise": int(snr_check["ciclos_sin_ruido"].sum()),
    }
    save_json({
        "fase": "9 - robustez al ruido",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": cfg.robustness.as_dict(), "random_state": cfg.random_state,
        "n_test_cycles": test.n_samples, "n_conditions": len(conditions),
        "models": {name: f"{meta['variant']}__{meta['representation']}__{meta['classifier']}"
                   for name, (_, _, meta) in models.items()},
        "checks": checks, "baseline_f1_weighted": baseline,
        "outputs": {k: to_posix_relative(v, paths.root) for k, v in outputs.items()},
        **environment_info(LIBRARIES),
    }, paths.audit_dir / "ejecucion_robustez.json")
    return {"summary": summary, "by_repetition": by_repetition, "snr_check": snr_check, "checks": checks,
            "outputs": outputs, "baseline": baseline}
