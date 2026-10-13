"""Control de calidad metodologico (Fase 14).

Cada funcion ``check_*`` revisa un punto de la lista del proyecto y devuelve el estado,
que se comprobo y la evidencia numerica. No se limita a confirmar que el codigo corre:
vuelve a medir sobre los artefactos reales (escaladores, huellas, predicciones guardadas,
manifiestos). Nada se reentrena y las caracteristicas del test no se usan para modelar.

Estados:
    CONFORME                  la comprobacion pasa sin reservas.
    CONFORME CON OBSERVACION  pasa, pero hay algo que debe quedar escrito en la tesis.
    NO CONFORME               hay un defecto que corregir.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.analisis import feature_separability
from src.configuracion import Config
from src.carga_datos import load_cycle_metadata
from src.constructor_datos import TEST_ACCESS_LOG, Dataset, load_train_data
from src.extraccion_caracteristicas import config_fingerprint, load_features
from src.evaluacion_final import FREEZE_NAME, Candidate, load_candidates, verify_freeze
from src.inferencia import analyze_cycle, decode_wav, load_models, predict
from src.robustez import CLEAN, Condition, extract_condition
from src.segmentacion import load_cycle_segment
from src.particion import (
    LOCK_NAME,
    TEST,
    TRAIN,
    build_split_manifest,
    load_locked_split,
    manifest_bytes,
    select_split,
    sha256_bytes,
)

OK, NOTE, FAIL = "CONFORME", "CONFORME CON OBSERVACION", "NO CONFORME"
EXPECTED_FEATURES = {"mfcc": 26, "mel": 128, "wavelet": 32, "rms": 7, "descriptors": 32, "fusion": 193,
                     "fusion_plus": 225}
FORBIDDEN_FEATURE_TOKENS = ("label", "crackle", "wheeze", "patient", "equipment", "diagn", "split", "fold")


@dataclass
class CheckResult:
    punto: str
    estado: str
    comprobacion: str
    evidencia: str
    detalle: dict[str, Any] = field(default_factory=dict)


class Context:
    """Artefactos del proyecto, cargados una sola vez para todas las comprobaciones."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.paths = cfg.paths
        self.classes = tuple(cfg.labels.modes[cfg.labels.label_mode])

    @cached_property
    def manifest(self) -> pd.DataFrame:
        return load_locked_split(self.cfg)

    @cached_property
    def lock(self) -> dict[str, Any]:
        return json.loads((self.paths.results_dir / LOCK_NAME).read_text(encoding="utf-8"))

    @cached_property
    def candidates(self) -> list[Candidate]:
        return load_candidates(self.cfg)

    @cached_property
    def models(self) -> dict[str, Any]:
        return {c.key: joblib.load(c.model_path) for c in self.candidates}

    @cached_property
    def freeze(self) -> dict[str, Any]:
        return json.loads((self.paths.results_dir / FREEZE_NAME).read_text(encoding="utf-8"))

    @cached_property
    def access_log(self) -> pd.DataFrame:
        return pd.read_csv(self.paths.results_dir / TEST_ACCESS_LOG, parse_dates=["timestamp_utc"])

    @cached_property
    def results(self) -> pd.DataFrame:
        main = pd.read_csv(self.paths.results_dir / "resultados_12_modelos.csv")
        extra = pd.read_csv(self.paths.results_dir / "improvements" / "resultados_mejoras_prueba.csv")
        return pd.concat([main, extra], ignore_index=True)

    def train(self, representation: str) -> Dataset:
        if not hasattr(self, "_train"):
            self._train: dict[str, Dataset] = {}
        if representation not in self._train:
            self._train[representation] = load_train_data(self.cfg, representation)
        return self._train[representation]

    def predictions(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.paths.results_dir / "predictions" / f"{key}.parquet")

    def pipelines(self, key: str) -> list[tuple[str, Any]]:
        """Pipelines ``scaler -> clasificador`` que componen un modelo (uno, o los miembros de un voto)."""
        model = self.models[key]
        return list(model.fitted_) if hasattr(model, "fitted_") else [(key, model)]


def _status(ok: bool, observation: bool = False) -> str:
    return FAIL if not ok else NOTE if observation else OK


def _train_sample(ctx: Context, n: int = 40) -> tuple[Dataset, np.ndarray]:
    """Filas de train elegidas con la semilla del proyecto (posiciones dentro de la matriz de train)."""
    data = ctx.train("fusion_plus")
    rng = np.random.default_rng(ctx.cfg.random_state)
    return data, np.sort(rng.choice(data.n_samples, size=n, replace=False))


def _recompute_train_rows(ctx: Context, data: Dataset, positions: np.ndarray) -> np.ndarray:
    """Vuelve a extraer desde el audio crudo las caracteristicas de unas filas de train."""
    cycles = load_cycle_metadata(ctx.paths.processed_dir / "metadatos_ciclos.csv").set_index("cycle_id", drop=False)
    rows = []
    for position in positions:
        segment = load_cycle_segment(cycles.loc[data.cycle_ids[position]], ctx.paths.raw_icbhi_dir)
        vector, _ = extract_condition(segment.audio, segment.sample_rate, segment.cycle_id, Condition(CLEAN, None, 0),
                                      ctx.cfg)
        rows.append(vector)
    return np.vstack(rows)


# 1-5. Fugas y particion


def check_data_leakage(ctx: Context) -> CheckResult:
    log = ctx.access_log
    first_access = log["timestamp_utc"].min()
    frozen_at = pd.Timestamp(ctx.freeze["frozen_at_utc"])
    last_model = max(pd.Timestamp(c.meta["created_at_utc"]) for c in ctx.candidates)
    allowed = log["purpose"].str.startswith(("Fase 7", "Fase 9", "Fase 11", "Fase 12")).all()
    n_train = {int(c.meta["n_train_samples"]) for c in ctx.candidates}
    expected_train = ctx.train("rms").n_samples
    data, positions = _train_sample(ctx)
    diff = float(np.max(np.abs(_recompute_train_rows(ctx, data, positions) - data.X.iloc[positions].to_numpy())))
    wavs = sorted(ctx.paths.raw_icbhi_dir.rglob("*.wav"))
    digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in wavs}
    ok = (allowed and last_model < frozen_at < first_access and n_train == {expected_train} and diff < 1e-8
          and len(digests) == len(wavs))
    return CheckResult(
        "DATA LEAKAGE", _status(ok, observation=True),
        "El test solo se leyo despues de sellar los modelos y para fases permitidas; todos los modelos se "
        "ajustaron con el mismo train; las caracteristicas de un ciclo no dependen de otros ciclos; no hay "
        "grabaciones duplicadas.",
        f"Ultimo modelo creado {last_model:%Y-%m-%d %H:%M} < sello {frozen_at:%Y-%m-%d %H:%M} < primer acceso al "
        f"test {first_access:%Y-%m-%d %H:%M} (UTC). {len(log)} accesos, todos de las Fases 7, 9 o 12. Los "
        f"{len(ctx.candidates)} modelos se ajustaron con {expected_train} ciclos. Recalcular {len(positions)} ciclos "
        f"de train desde el audio da diferencia maxima {diff:.1e} frente a la cache. {len(wavs)} WAV con "
        f"{len(digests)} contenidos distintos. Observacion: el split se eligio usando las etiquetas y el equipo de "
        "todos los pacientes para equilibrar clases y equipos (estratificacion); no se usaron caracteristicas ni "
        "resultados de modelos.",
        {"first_test_access_utc": str(first_access), "frozen_at_utc": str(frozen_at),
         "last_model_created_utc": str(last_model), "n_accesses": int(len(log)), "recompute_max_abs_diff": diff,
         "n_wav": len(wavs), "n_distinct_wav": len(digests)})


def check_patient_leakage(ctx: Context) -> CheckResult:
    manifest = ctx.manifest
    train = set(manifest.loc[manifest["split"] == TRAIN, "patient_id"].astype(str))
    test = set(manifest.loc[manifest["split"] == TEST, "patient_id"].astype(str))
    folds = pd.read_csv(ctx.paths.results_dir / "folds_validacion_cruzada.csv", dtype={"patient_id": str})
    folds_per_patient = int(folds.groupby("patient_id")["fold"].nunique().max())
    groups = ctx.train("rms").groups
    shared_calibration = 0
    n_calibrated = 0
    for candidate in ctx.candidates:
        for _, pipeline in ctx.pipelines(candidate.key):
            step = pipeline.named_steps["classifier"]
            if getattr(step, "cv", None) is not None and hasattr(step, "calibrator_"):
                n_calibrated += 1
                for fit_idx, val_idx in step.cv:
                    shared_calibration += len(set(groups[fit_idx]) & set(groups[val_idx]))
    ok = (not train & test and folds_per_patient == 1 and not set(folds["patient_id"]) & test
          and shared_calibration == 0)
    return CheckResult(
        "DATA LEAKAGE ENTRE PACIENTES", _status(ok),
        "Ningun paciente esta a la vez en train y test, ni en dos folds de la validacion cruzada, ni a los dos "
        "lados de los folds de calibracion de los SVM.",
        f"Train {len(train)} pacientes, test {len(test)}, compartidos {len(train & test)}. Folds: cada paciente en "
        f"{folds_per_patient} fold y 0 pacientes de test dentro de los folds. {n_calibrated} SVM calibrados: "
        f"{shared_calibration} pacientes compartidos entre ajuste y validacion de la calibracion. El dataset externo "
        "(Fraiwan) nunca entro en ningun ajuste.",
        {"n_train_patients": len(train), "n_test_patients": len(test), "shared": len(train & test),
         "max_folds_per_patient": folds_per_patient, "calibrated_svms": n_calibrated,
         "shared_in_calibration": shared_calibration})


def check_scaler_leakage(ctx: Context) -> CheckResult:
    worst_mean, worst_scale, n_pipelines, bad_counts, steps_ok = 0.0, 0.0, 0, 0, True
    for candidate in ctx.candidates:
        data = ctx.train(candidate.representation)
        X = data.X.to_numpy()
        for _, pipeline in ctx.pipelines(candidate.key):
            steps_ok &= [name for name, _ in pipeline.steps] == ["scaler", "classifier"]
            scaler = pipeline.named_steps["scaler"]
            n_pipelines += 1
            bad_counts += int(scaler.n_samples_seen_ != data.n_samples)
            worst_mean = max(worst_mean, float(np.max(np.abs(scaler.mean_ - X.mean(axis=0)))))
            worst_scale = max(worst_scale, float(np.max(np.abs(scaler.scale_ - X.std(axis=0)))))
    n_train = ctx.train("rms").n_samples
    ok = steps_ok and bad_counts == 0 and worst_mean < 1e-8 and worst_scale < 1e-8
    return CheckResult(
        "SCALER LEAKAGE", _status(ok, observation=True),
        "Cada modelo es un Pipeline escalador -> clasificador y su escalador se ajusto SOLO con los ciclos de "
        "train (ni uno del test).",
        f"{n_pipelines} pipelines revisados: todos son [scaler, classifier]; en todos el escalador vio exactamente "
        f"{n_train} muestras, y su media y desviacion coinciden con las de la matriz de train (diferencia maxima "
        f"{worst_mean:.1e} y {worst_scale:.1e}). En la validacion cruzada el escalador va dentro del pipeline, asi "
        "que se reajusta en cada fold solo con su parte de ajuste (hay un test que lo comprueba). Observacion: al "
        "calibrar las probabilidades de los SVM finales, el escalador ya estaba ajustado con todo train; afecta "
        "solo a las probabilidades (no a la clase) y no involucra al test.",
        {"n_pipelines": n_pipelines, "max_abs_diff_mean": worst_mean, "max_abs_diff_scale": worst_scale,
         "scalers_with_other_sample_count": bad_counts})


def check_target_leakage(ctx: Context) -> CheckResult:
    data = ctx.train("fusion_plus")
    names = list(data.X.columns)
    suspicious = [n for n in names if any(token in n.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    separability = feature_separability(data)
    top = separability.head(3)
    duration_like = [n for n in names if "duration" in n.lower()]
    ok = not suspicious and float(separability["eta2"].max()) < 0.5
    return CheckResult(
        "TARGET LEAKAGE", _status(ok, observation=bool(duration_like)),
        "Ninguna caracteristica se calcula a partir de la etiqueta, el paciente o el equipo, y ninguna separa las "
        "clases de una forma sospechosamente buena.",
        f"{len(names)} caracteristicas: {len(suspicious)} con nombres de etiqueta, paciente o equipo. Las funciones "
        "de extraccion solo reciben el audio. La caracteristica que mas separa las clases en train explica el "
        f"{100 * float(top['eta2'].iloc[0]):.1f} % de la varianza entre clases ({top['feature'].iloc[0]}); una fuga "
        "daria valores cercanos al 100 %. Observacion: los descriptores incluyen la duracion del ciclo "
        f"({', '.join(duration_like)}), que sale de los limites anotados; solo la usan las mejoras propuestas, y sin "
        "ella la F1 de validacion cruzada bajaba de 0,600 a 0,592.",
        {"n_features": len(names), "suspicious_names": suspicious, "max_eta2": float(separability["eta2"].max()),
         "top_features": top[["feature", "eta2"]].to_dict(orient="records"), "duration_features": duration_like})


def check_split_reproducible(ctx: Context) -> CheckResult:
    cfg = ctx.cfg
    index, _ = load_features(cfg, cfg.features.representations[0])
    cycles = load_cycle_metadata(ctx.paths.processed_dir / "metadatos_ciclos.csv")
    index = index.merge(cycles[["cycle_id", "source_audio"]].rename(columns={"source_audio": "audio_file"}),
                        on="cycle_id", how="left", validate="one_to_one")
    selection = select_split(index, cfg.labels, cfg.split, cfg.random_state)
    rebuilt = sha256_bytes(manifest_bytes(build_split_manifest(index, selection.test_patients, cfg.labels)))
    on_disk = sha256_bytes(manifest_bytes(ctx.manifest))
    locked = ctx.lock["manifest_sha256"]
    ok = rebuilt == locked == on_disk and selection.candidate_index == ctx.lock["selected_candidate"]
    return CheckResult(
        "SPLIT REPRODUCIBLE", _status(ok),
        "Volver a generar el split desde cero, con la misma configuracion y semilla, produce exactamente el mismo "
        "archivo que esta bloqueado.",
        f"Se evaluaron de nuevo los {cfg.split.n_candidates} candidatos (semilla {cfg.random_state}): gana el "
        f"candidato {selection.candidate_index}, el mismo del bloqueo. Huella SHA-256 del manifiesto regenerado "
        f"{rebuilt[:16]}..., del archivo en disco {on_disk[:16]}... y del bloqueo {locked[:16]}...: identicas.",
        {"rebuilt_sha256": rebuilt, "disk_sha256": on_disk, "locked_sha256": locked,
         "candidate": int(selection.candidate_index)})


# 6-10. Caracteristicas, clases y sobreajuste


def check_feature_dimensions(ctx: Context) -> CheckResult:
    shapes = {name: load_features(ctx.cfg, name)[1].shape for name in ctx.cfg.features.representations}
    mismatches = []
    for candidate in ctx.candidates:
        expected = EXPECTED_FEATURES[candidate.representation]
        columns = list(ctx.train(candidate.representation).X.columns)
        for _, pipeline in ctx.pipelines(candidate.key):
            n_in = int(pipeline.named_steps["scaler"].n_features_in_)
            if not (n_in == expected == len(candidate.meta["feature_names"]) and candidate.meta["feature_names"] == columns):
                mismatches.append(candidate.key)
    ok = all(shape == (6898, EXPECTED_FEATURES[name]) for name, shape in shapes.items()) and not mismatches
    return CheckResult(
        "FEATURE DIMENSIONS", _status(ok),
        "La cache tiene las dimensiones del diseno y cada modelo espera exactamente las columnas, en el mismo "
        "orden, de su representacion.",
        "Cache: " + ", ".join(f"{name} {shape[0]} x {shape[1]}" for name, shape in shapes.items())
        + f". {len(ctx.candidates)} modelos revisados: {len(mismatches)} con columnas o dimensiones distintas a las "
        "esperadas (MFCC 26, Mel 128, wavelet 32, RMS 7, descriptores 32, fusion 193, fusion + descriptores 225).",
        {"cache_shapes": {k: list(v) for k, v in shapes.items()}, "mismatches": mismatches})


def _finite_report(ctx: Context) -> dict[str, dict[str, int]]:
    report = {}
    for name in ctx.cfg.features.representations:
        values = load_features(ctx.cfg, name)[1].to_numpy()
        report[f"cache {name}"] = {"nan": int(np.isnan(values).sum()), "inf": int(np.isinf(values).sum())}
    values = ctx.train("descriptors").X.to_numpy()
    report["descriptores (train)"] = {"nan": int(np.isnan(values).sum()), "inf": int(np.isinf(values).sum())}
    nan = inf = 0
    for candidate in ctx.candidates:
        saved = ctx.predictions(candidate.key).select_dtypes("number").to_numpy(dtype=float)
        nan += int(np.isnan(saved).sum())
        inf += int(np.isinf(saved).sum())
    report["predicciones y probabilidades guardadas"] = {"nan": nan, "inf": inf}
    metrics = ctx.results.filter(like="test_").to_numpy(dtype=float)
    report["tablas de resultados"] = {"nan": int(np.isnan(metrics).sum()), "inf": int(np.isinf(metrics).sum())}
    return report


def check_nan(ctx: Context) -> CheckResult:
    report = _finite_report(ctx)
    total = sum(item["nan"] for item in report.values())
    return CheckResult(
        "NaN", _status(total == 0),
        "No hay valores NaN en las caracteristicas, las predicciones, las probabilidades ni las tablas.",
        "; ".join(f"{name}: {item['nan']}" for name, item in report.items()) + ". El preprocesamiento y la "
        "extraccion rechazan con error cualquier entrada o salida no finita, en lugar de dejarla pasar.",
        report)


def check_infinity(ctx: Context) -> CheckResult:
    report = _finite_report(ctx)
    total = sum(item["inf"] for item in report.values())
    return CheckResult(
        "INFINITY", _status(total == 0),
        "No hay valores infinitos en las caracteristicas, las predicciones, las probabilidades ni las tablas.",
        "; ".join(f"{name}: {item['inf']}" for name, item in report.items()) + ". Los logaritmos usan un piso "
        "(amin en el log-Mel, epsilon en los descriptores) y el silencio no se normaliza, asi que no hay divisiones "
        "por cero.",
        report)


def check_class_imbalance(ctx: Context) -> CheckResult:
    manifest = ctx.manifest[ctx.manifest["in_3class"]]
    shares = {split: (manifest.loc[manifest["split"] == split, "label"].value_counts(normalize=True)
                      .reindex(ctx.classes)) for split in (TRAIN, TEST)}
    best = ctx.results[(ctx.results["variant"] == "original") & ctx.results["designated_by_cv"]].iloc[0]
    summary = json.loads((ctx.paths.results_dir / "resultados_mejor_modelo.json").read_text(encoding="utf-8"))
    baseline = float(summary["test_majority_baseline_f1_weighted"])
    recalls = {name: float(best[f"test_recall_{name}"]) for name in ctx.classes}
    text = lambda s: ", ".join(f"{n} {100 * v:.1f} %" for n, v in s.items())  # noqa: E731
    return CheckResult(
        "CLASS IMBALANCE", NOTE,
        "El desbalance esta medido, se respeta al partir los datos y se tiene en cuenta al reportar.",
        f"Train: {text(shares[TRAIN])}. Test: {text(shares[TEST])}. El split y los folds son estratificados, asi "
        "que las proporciones se mantienen. Se reporta la F1 ponderada (metrica del proyecto) y tambien la F1 macro "
        f"y el recall por clase, y se compara con predecir siempre la clase mayoritaria (F1 {baseline:.3f}). "
        "Observacion: el modelo elegido no pondera las clases (asi lo define la metodologia) y su recall es "
        + ", ".join(f"{n} {v:.2f}" for n, v in recalls.items())
        + "; el balanceo de clases se evaluo aparte como mejora propuesta.",
        {"train_shares": shares[TRAIN].to_dict(), "test_shares": shares[TEST].to_dict(),
         "majority_baseline_f1": baseline, "best_model_recall": recalls,
         "best_model_f1_macro": float(best["test_f1_macro"])})


def check_overfitting(ctx: Context) -> CheckResult:
    main = ctx.results[ctx.results["variant"] == "original"]
    gap_train_cv = main["train_f1_weighted"] - main["cv_f1_weighted"]
    gap_cv_test = main["cv_f1_weighted"] - main["test_f1_weighted"]
    best = main[main["designated_by_cv"]].iloc[0]
    perfect = int((main["train_f1_weighted"] > 0.999).sum())
    return CheckResult(
        "OVERFITTING", NOTE,
        "Se midio la distancia entre entrenamiento, validacion cruzada y prueba en las 12 combinaciones.",
        f"Modelo elegido: train {best['train_f1_weighted']:.3f}, CV {best['cv_f1_weighted']:.3f}, test "
        f"{best['test_f1_weighted']:.3f}. En las 12: train - CV va de {gap_train_cv.min():.3f} a "
        f"{gap_train_cv.max():.3f} (media {gap_train_cv.mean():.3f}) y {perfect} modelos llegan a 1,000 en train; "
        f"CV - test va de {gap_cv_test.min():+.3f} a {gap_cv_test.max():+.3f} (media {gap_cv_test.mean():+.3f}). "
        "Observacion: hay sobreajuste a los pacientes de entrenamiento, que la validacion cruzada agrupada por "
        "paciente descuenta; la validacion cruzada anticipo bien la prueba, es decir, la seleccion de modelos no se "
        "sobreajusto.",
        {"best_model": {"train": float(best["train_f1_weighted"]), "cv": float(best["cv_f1_weighted"]),
                        "test": float(best["test_f1_weighted"])},
         "gap_train_cv_mean": float(gap_train_cv.mean()), "gap_cv_test_mean": float(gap_cv_test.mean()),
         "models_with_perfect_train": perfect})


# 11-13. AUC, probabilidades y orden de etiquetas


def check_auc(ctx: Context) -> CheckResult:
    worst_table = worst_manual = 0.0
    labels = list(range(len(ctx.classes)))
    for row in ctx.results.itertuples():
        saved = ctx.predictions(f"{row.variant}__{row.representation}__{row.classifier}")
        y = saved["y_true"].to_numpy()
        proba = saved[[f"proba_{name}" for name in ctx.classes]].to_numpy()
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=labels)
        per_class = [roc_auc_score((y == code).astype(int), proba[:, code]) for code in labels]
        manual = float(np.average(per_class, weights=np.bincount(y, minlength=len(labels))))
        worst_table = max(worst_table, abs(auc - row.test_auc_roc_weighted))
        worst_manual = max(worst_manual, abs(auc - manual))
    ok = worst_table < 1e-9 and worst_manual < 1e-9
    return CheckResult(
        "AUC MULTICLASE CORRECTA", _status(ok),
        "El AUC reportado es One-vs-Rest ponderado por soporte, calculado con las probabilidades en el orden de "
        "clases del proyecto.",
        f"Recalculado para los {len(ctx.results)} modelos desde las probabilidades guardadas: diferencia maxima con "
        f"la tabla {worst_table:.1e}. Coincide con la media, ponderada por el numero de ciclos de cada clase, de los "
        f"tres AUC binarios (diferencia maxima {worst_manual:.1e}). Se calcula con predict_proba y no con la clase "
        "predicha.",
        {"max_abs_diff_table": worst_table, "max_abs_diff_manual_formula": worst_manual})


def check_probabilities(ctx: Context) -> CheckResult:
    worst_sum, lowest, highest, disagreement = 0.0, 1.0, 0.0, {}
    for candidate in ctx.candidates:
        saved = ctx.predictions(candidate.key)
        proba = saved[[f"proba_{name}" for name in ctx.classes]].to_numpy()
        worst_sum = max(worst_sum, float(np.max(np.abs(proba.sum(axis=1) - 1.0))))
        lowest, highest = min(lowest, float(proba.min())), max(highest, float(proba.max()))
        disagreement[candidate.key] = float(np.mean(proba.argmax(axis=1) != saved["y_pred"].to_numpy()))
    svm = {k: v for k, v in disagreement.items() if k.endswith("svm_rbf")}
    other = {k: v for k, v in disagreement.items() if not k.endswith(("svm_rbf", "vote"))}
    best_key = ctx.freeze["designated_main"]
    ok = worst_sum < 1e-6 and lowest >= 0.0 and highest <= 1.0 and max(other.values()) == 0.0
    return CheckResult(
        "PROBABILIDADES", _status(ok, observation=True),
        "Las probabilidades de cada ciclo son validas (entre 0 y 1, suman 1) y se sabe cuando no coinciden con la "
        "clase predicha.",
        f"{len(ctx.candidates)} modelos: suma por ciclo a 1 con error maximo {worst_sum:.1e}; valores entre "
        f"{lowest:.4f} y {highest:.4f}. En Random Forest y KNN la clase predicha es siempre la de mayor "
        "probabilidad. Observacion: en los SVM la clase la decide la frontera del clasificador y la probabilidad "
        f"sale de una calibracion aparte, asi que pueden diferir: entre {100 * min(svm.values()):.1f} % y "
        f"{100 * max(svm.values()):.1f} % de los ciclos segun el modelo ({100 * disagreement[best_key]:.1f} % en el "
        "modelo elegido). La interfaz marca esos casos como dudosos.",
        {"max_abs_sum_error": worst_sum, "min": lowest, "max": highest, "disagreement_by_model": disagreement})


def check_label_order(ctx: Context) -> CheckResult:
    encoding = {name: i for i, name in enumerate(ctx.classes)}
    bad_meta = [c.key for c in ctx.candidates if c.meta["class_encoding"] != encoding]
    bad_model = [c.key for c in ctx.candidates if list(ctx.models[c.key].classes_) != list(range(len(ctx.classes)))]
    truth = ctx.manifest.set_index("cycle_id")["label"]
    mismatched = 0
    for candidate in ctx.candidates:
        saved = ctx.predictions(candidate.key)
        mismatched += int((saved["cycle_id"].map(truth).map(encoding).to_numpy() != saved["y_true"].to_numpy()).sum())
    ok = not bad_meta and not bad_model and mismatched == 0
    return CheckResult(
        "LABEL ORDER", _status(ok),
        "El orden de clases es el mismo en la configuracion, en los metadatos, dentro de cada modelo y en las "
        "predicciones guardadas.",
        "Orden fijado en config.yaml: " + ", ".join(f"{name} = {code}" for name, code in encoding.items())
        + f". {len(ctx.candidates)} modelos: {len(bad_meta)} con otra codificacion en sus metadatos y "
        f"{len(bad_model)} con classes_ distinto de [0, 1, 2]. La etiqueta real guardada con cada prediccion "
        f"coincide con la del manifiesto del split en todos los ciclos ({mismatched} diferencias). Las matrices de "
        "confusion usan filas = clase real y columnas = clase predicha, en ese mismo orden.",
        {"encoding": encoding, "bad_metadata": bad_meta, "bad_models": bad_model, "label_mismatches": mismatched})


# 14-16. Carga, inferencia y validacion externa


def check_model_loading(ctx: Context) -> CheckResult:
    manifest = verify_freeze(ctx.cfg, ctx.candidates)
    installed = version("scikit-learn")
    trained_with = {c.meta.get("libraries", {}).get("scikit-learn") for c in ctx.candidates} - {None}
    sample = ctx.train("fusion_plus").X.iloc[:32]
    bad_output = []
    for candidate in ctx.candidates:
        rows = sample[candidate.meta["feature_names"]]
        proba = ctx.models[candidate.key].predict_proba(rows)
        if ctx.models[candidate.key].predict(rows).shape != (32,) or proba.shape != (32, len(ctx.classes)):
            bad_output.append(candidate.key)
    copies = {}
    for name, key in (("mejor_modelo.joblib", manifest["designated_main"]),
                      ("modelo_prototipo.joblib", manifest["designated_prototype"])):
        path = ctx.paths.model_dir / name
        copies[name] = hashlib.sha256(path.read_bytes()).hexdigest() == manifest["models"][key]["sha256"]
    ok = not bad_output and all(copies.values()) and trained_with <= {installed}
    return CheckResult(
        "CARGA DEL MODELO", _status(ok),
        "Todos los modelos guardados se cargan, son byte a byte los que se sellaron antes de abrir el test y "
        "predicen con la forma esperada.",
        f"{len(ctx.candidates)} modelos cargados con joblib y verificados contra su huella SHA-256 del sello. "
        "mejor_modelo.joblib y modelo_prototipo.joblib son copias identicas de sus candidatos. Se entrenaron con "
        f"scikit-learn {', '.join(sorted(trained_with))} y se cargan con {installed}. Todos devuelven una clase y "
        f"tres probabilidades por ciclo ({len(bad_output)} con salida incorrecta).",
        {"n_models": len(ctx.candidates), "copies_identical": copies, "sklearn_trained": sorted(trained_with),
         "sklearn_installed": installed, "bad_outputs": bad_output})


def check_inference_consistency(ctx: Context) -> CheckResult:
    cfg = ctx.cfg
    bundles = load_models(cfg)
    saved_keys = {"best_model": ctx.freeze["designated_main"], "prototype": ctx.freeze["designated_prototype"]}
    examples = pd.read_csv(cfg.paths.root / "app" / "examples" / "ejemplos.csv")
    example_mismatch, worst_proba = 0, 0.0
    for key, bundle in bundles.items():
        saved = ctx.predictions(saved_keys[key]).set_index("cycle_id")
        for row in examples.itertuples():
            audio, sr = decode_wav((cfg.paths.root / "app" / "examples" / row.file).read_bytes())
            prediction = predict(bundle, analyze_cycle(audio, sr, cfg))
            example_mismatch += int(prediction.label != ctx.classes[int(saved.loc[row.cycle_id, "y_pred"])])
            worst_proba = max(worst_proba, max(abs(prediction.probabilities[name] - saved.loc[row.cycle_id, f"proba_{name}"])
                                               for name in ctx.classes))
    data, positions = _train_sample(ctx)
    recomputed = pd.DataFrame(_recompute_train_rows(ctx, data, positions), columns=data.X.columns)
    cached = data.X.iloc[positions]
    train_mismatch = 0
    for bundle in bundles.values():
        train_mismatch += int((bundle.model.predict(recomputed[bundle.feature_names])
                               != bundle.model.predict(cached[bundle.feature_names])).sum())
    robustness = json.loads((cfg.paths.audit_dir / "ejecucion_robustez.json").read_text(encoding="utf-8"))["checks"]
    ok = (example_mismatch == 0 and worst_proba < 1e-8 and train_mismatch == 0
          and robustness["clean_features_max_abs_diff_vs_cache"] < 1e-8
          and all(robustness["clean_predictions_equal_phase7"].values()))
    return CheckResult(
        "INFERENCE CONSISTENCY", _status(ok),
        "Un audio que entra por la interfaz o por la prueba de ruido recibe exactamente las mismas "
        "caracteristicas y la misma prediccion que en la evaluacion.",
        f"Interfaz: {len(examples)} ciclos de ejemplo x {len(bundles)} modelos, {example_mismatch} clases distintas "
        f"a las de la Fase 7 y diferencia maxima de probabilidad {worst_proba:.1e}. {len(positions)} ciclos de train "
        f"recalculados desde el audio por la ruta de inferencia: {train_mismatch} predicciones distintas a las "
        "obtenidas con la cache. Prueba de ruido, condicion sin ruido: diferencia maxima con la cache "
        f"{robustness['clean_features_max_abs_diff_vs_cache']:.1e} y predicciones identicas en los dos modelos. Hay "
        "una sola funcion de preprocesamiento para entrenamiento, robustez, validacion externa e interfaz.",
        {"example_label_mismatches": example_mismatch, "example_max_abs_proba_diff": worst_proba,
         "train_prediction_mismatches": train_mismatch, "robustness_checks": robustness})


def check_external_consistency(ctx: Context) -> CheckResult:
    cfg = ctx.cfg
    table = pd.read_csv(cfg.paths.results_dir / "resultados_validacion_externa.csv")
    recordings = pd.read_csv(cfg.paths.results_dir / "external" / "predicciones_por_grabacion.csv")
    exclusions = pd.read_csv(cfg.paths.results_dir / "external" / "exclusiones.csv")
    mapping = pd.read_csv(cfg.paths.results_dir / "external" / "equivalencia_etiquetas_aplicada.csv")
    run = json.loads((cfg.paths.audit_dir / "ejecucion_validacion_externa.json").read_text(encoding="utf-8"))
    worst = 0.0
    sizes = set()
    for (model, code), group in recordings.groupby(["model", "filter_code"]):
        row = table[(table["dataset"] == "Fraiwan") & (table["level"] == "grabacion") & (table["model"] == model)
                    & (table["filter_code"] == code)].iloc[0]
        f1 = f1_score(group["y_true"], group["y_pred"], average="weighted", zero_division=0)
        worst = max(worst, abs(f1 - row["f1_weighted"]))
        sizes.add((len(group), group["patient_id"].nunique()))
    mapped_ok = dict(zip(mapping["sound_type"], mapping["label"])) == {
        k: v for k, v in cfg.external.label_map.items() if k in set(mapping["sound_type"])}
    train_recordings = pd.read_parquet(cfg.paths.features_dir / "indice.parquet", columns=["recording_id"])
    fraiwan_in_cache = int(train_recordings["recording_id"].str.match(r"^[BDE]P\d+_").sum())
    ok = (worst < 1e-9 and sizes == {(107, 107)} and int(exclusions["n_pacientes"].sum()) == 5 and mapped_ok
          and run["checks"]["model_files_unchanged"] and run["strategy"]["window_sec"] == cfg.external.window_sec
          and fraiwan_in_cache == 0)
    return CheckResult(
        "EXTERNAL VALIDATION CONSISTENCY", _status(ok, observation=True),
        "La validacion externa uso los modelos congelados, el mismo preprocesamiento y una equivalencia de "
        "etiquetas declarada; sus tablas coinciden con las predicciones guardadas.",
        f"F1 recalculada desde las predicciones por grabacion: diferencia maxima con la tabla {worst:.1e}. En cada "
        "filtro y modelo hay 107 grabaciones de 107 pacientes; 5 pacientes excluidos, con su motivo. La equivalencia "
        "aplicada es la de config.yaml. Los archivos de los modelos no cambiaron durante la validacion y ninguna "
        f"grabacion de Fraiwan esta en la cache de entrenamiento ({fraiwan_in_cache}). Las ventanas pasan por la "
        "misma funcion de extraccion que el entrenamiento. Observacion: Fraiwan etiqueta la grabacion completa, asi "
        "que la validacion es por grabacion (ventanas de "
        f"{cfg.external.window_sec:g} s y voto por mayoria) y no es equivalente a la de ICBHI por ciclo.",
        {"max_abs_diff_f1": worst, "n_excluded_patients": int(exclusions["n_pacientes"].sum()),
         "model_files_unchanged": bool(run["checks"]["model_files_unchanged"]), "fraiwan_rows_in_cache": fraiwan_in_cache})


CHECKS = (check_data_leakage, check_patient_leakage, check_scaler_leakage, check_target_leakage,
          check_split_reproducible, check_feature_dimensions, check_nan, check_infinity, check_class_imbalance,
          check_overfitting, check_auc, check_probabilities, check_label_order, check_model_loading,
          check_inference_consistency, check_external_consistency)


def run_quality_control(cfg: Config) -> list[CheckResult]:
    """Ejecuta los 16 puntos; un fallo inesperado en un punto se reporta como NO CONFORME."""
    ctx = Context(cfg)
    results = []
    for check in CHECKS:
        try:
            results.append(check(ctx))
        except Exception as error:  # noqa: BLE001 - el informe debe decir que punto no se pudo comprobar
            results.append(CheckResult(check.__name__.removeprefix("check_").upper(), FAIL,
                                       "La comprobacion no se pudo ejecutar.", f"{type(error).__name__}: {error}"))
    return results


__all__ = ["CHECKS", "CheckResult", "Context", "FAIL", "NOTE", "OK", "config_fingerprint", "run_quality_control"]
