"""Medicion de eficiencia (Fase 15).

Criterios del proyecto que aqui se comprueban con cifras:

* cada representacion se extrae una vez y se guarda en la cache; el entrenamiento de las
  12 combinaciones (y de las mejoras) lee esas matrices, nunca el audio;
* la extraccion y la busqueda de hiperparametros corren en paralelo (``n_jobs=-1``);
* el audio no se carga completo en memoria: cada proceso lee una grabacion, extrae sus
  ciclos y la suelta antes de pasar a la siguiente;
* el progreso se informa con ``tqdm`` (extraccion) y ``logging`` (resto).

Las mediciones de extraccion usan solo grabaciones de train y no guardan nada.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.configuracion import Config
from src.carga_datos import load_cycle_metadata
from src.constructor_datos import load_train_data
from src.extraccion_caracteristicas import extract_dataset_features
from src.evaluacion_final import load_candidates
from src.inferencia import analyze_cycle, decode_wav, load_models, predict

MB = 1024.0 * 1024.0


def peak_memory_mb() -> float:
    """Pico de memoria fisica usado por este proceso desde que arranco, en MB."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / MB
    import resource  # noqa: PLC0415

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / MB if sys.platform == "darwin" else usage / 1024.0


def folder_size_mb(folder: Path, pattern: str = "*") -> float:
    """Tamano en disco de los archivos de una carpeta (recursivo), en MB."""
    return sum(p.stat().st_size for p in Path(folder).rglob(pattern) if p.is_file()) / MB


def median_seconds(function: Any, repeats: int = 5) -> float:
    """Mediana del tiempo de ``repeats`` llamadas."""
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def train_recording_sample(cfg: Config, n_recordings: int) -> pd.DataFrame:
    """Ciclos de ``n_recordings`` grabaciones de train elegidas con la semilla del proyecto."""
    train = load_train_data(cfg, cfg.features.representations[0])
    cycles = load_cycle_metadata(cfg.paths.processed_dir / "metadatos_ciclos.csv")
    cycles = cycles[cycles["cycle_id"].isin(set(train.cycle_ids))]
    recordings = np.sort(cycles["recording_id"].unique())
    chosen = np.random.default_rng(cfg.random_state).choice(recordings, size=min(n_recordings, len(recordings)),
                                                             replace=False)
    return cycles[cycles["recording_id"].isin(set(chosen))].reset_index(drop=True)


def run_benchmark(cfg: Config, n_recordings: int = 200) -> pd.DataFrame:
    """Mide cada etapa y devuelve una tabla ``etapa, indicador, valor, unidad, detalle``."""
    paths = cfg.paths
    rows: list[dict[str, Any]] = []

    def add(stage: str, indicator: str, value: Any, unit: str = "", detail: str = "") -> None:
        rows.append({"etapa": stage, "indicador": indicator, "valor": value, "unidad": unit, "detalle": detail})

    n_cores = os.cpu_count() or 1
    add("equipo", "nucleos_logicos", n_cores)
    for name, value in (("features.n_jobs", cfg.features.n_jobs), ("training.n_jobs", cfg.training.n_jobs),
                        ("robustness.n_jobs", cfg.robustness.n_jobs), ("external_validation.n_jobs", cfg.external.n_jobs)):
        add("paralelismo", name, value, "", "-1 = todos los nucleos")

    # cache: se extrae una vez y se reutiliza
    raw_mb = folder_size_mb(paths.raw_icbhi_dir, "*.wav")
    cache_mb = folder_size_mb(paths.features_dir, "*.parquet")
    add("cache", "audio_original", round(raw_mb, 1), "MB", "920 WAV de ICBHI")
    add("cache", "matrices_en_cache", round(cache_mb, 1), "MB", "MFCC, Mel, wavelet, RMS y descriptores de 6 898 ciclos")
    add("cache", "reduccion", round(raw_mb / cache_mb, 0), "veces", "audio / cache")
    load_all = median_seconds(lambda: [load_train_data(cfg, r) for r in cfg.features.representations])
    add("cache", "cargar_las_4_representaciones_de_train", round(1000 * load_all, 0), "ms",
        "mediana de 5 lecturas; incluye verificar la huella y el split")

    # extraccion: un proceso frente a todos los nucleos (solo train, sin guardar)
    sample = train_recording_sample(cfg, n_recordings)
    n_cycles, n_recs = len(sample), sample["recording_id"].nunique()
    extract_dataset_features(sample.head(8), paths.raw_icbhi_dir, cfg, n_jobs=1, show_progress=False)  # calentamiento
    before = peak_memory_mb()
    started = time.perf_counter()
    serial = extract_dataset_features(sample, paths.raw_icbhi_dir, cfg, n_jobs=1, show_progress=False)
    serial_s = time.perf_counter() - started
    after = peak_memory_mb()
    started = time.perf_counter()
    parallel = extract_dataset_features(sample, paths.raw_icbhi_dir, cfg, n_jobs=-1, show_progress=False)
    cold_s = time.perf_counter() - started           # incluye arrancar los procesos y cargar las librerias
    started = time.perf_counter()
    extract_dataset_features(sample, paths.raw_icbhi_dir, cfg, n_jobs=-1, show_progress=False)
    parallel_s = time.perf_counter() - started       # procesos ya arrancados
    with threadpool_limits(limits=1):                # un solo hilo de algebra lineal, como en cada proceso hijo
        one_thread = extract_dataset_features(sample, paths.raw_icbhi_dir, cfg, n_jobs=1, show_progress=False)
    diff = max(float(np.max(np.abs(serial.matrices[k].to_numpy() - parallel.matrices[k].to_numpy())))
               for k in serial.matrices)
    relative = max(float(np.max(np.abs(serial.matrices[k].to_numpy() - parallel.matrices[k].to_numpy()))
                         / np.max(np.abs(parallel.matrices[k].to_numpy()))) for k in serial.matrices)
    same_threads = all(np.array_equal(one_thread.matrices[k].to_numpy(), parallel.matrices[k].to_numpy())
                       for k in serial.matrices)
    startup_s = max(0.0, cold_s - parallel_s)
    total_cycles = 6898
    add("extraccion", "muestra_medida", n_cycles, "ciclos", f"{n_recs} grabaciones de train, elegidas con la semilla")
    add("extraccion", "un_proceso", round(n_cycles / serial_s, 1), "ciclos/s", f"{serial_s:.1f} s en la muestra")
    add("extraccion", "todos_los_nucleos", round(n_cycles / parallel_s, 1), "ciclos/s",
        f"{parallel_s:.1f} s en la muestra con los procesos ya arrancados")
    add("extraccion", "aceleracion", round(serial_s / parallel_s, 2), "veces", f"con {n_cores} nucleos logicos")
    add("extraccion", "arranque_de_los_procesos", round(startup_s, 1), "s",
        "costo fijo de la primera llamada en paralelo (crear procesos y cargar librerias)")
    add("extraccion", "diferencia_maxima_serie_vs_paralelo", diff, "",
        f"relativa {relative:.1e}: ultimo bit, por el numero de hilos del algebra lineal en el producto del banco Mel")
    add("extraccion", "identico_con_el_mismo_numero_de_hilos", bool(same_threads), "",
        "un proceso limitado a 1 hilo da exactamente lo mismo que el modo paralelo")
    add("extraccion", "tiempo_estimado_dataset_completo_un_proceso", round(total_cycles * serial_s / n_cycles, 0), "s",
        "6 898 ciclos, las 4 representaciones en una sola pasada")
    add("extraccion", "tiempo_estimado_dataset_completo_en_paralelo",
        round(startup_s + total_cycles * parallel_s / n_cycles, 0), "s", "incluye el arranque de los procesos")

    # memoria: nunca se carga todo el audio
    recordings = pd.read_csv(paths.processed_dir / "metadatos_grabaciones.csv")
    largest_mb = float((recordings["n_frames"] * recordings["n_channels"]).max() * 4 / MB)
    all_audio_mb = float((recordings["n_frames"] * recordings["n_channels"]).sum() * 4 / MB)
    add("memoria", "todo_el_audio_en_float32", round(all_audio_mb, 0), "MB", "lo que ocuparia cargar las 920 grabaciones a la vez")
    add("memoria", "grabacion_mas_grande_en_float32", round(largest_mb, 1), "MB", "lo maximo que un proceso tiene de audio a la vez")
    add("memoria", "pico_del_proceso_durante_la_extraccion", round(after, 0), "MB",
        f"un proceso, {n_recs} grabaciones; antes de extraer el pico era {before:.0f} MB (librerias cargadas)")

    # entrenamiento: tiempos guardados con cada modelo
    candidates = load_candidates(cfg)
    main = [c for c in candidates if c.group == "original"]
    fits = sum(int(c.meta.get("n_grid_candidates") or cfg.models[c.classifier].n_grid_candidates)
               * int(c.meta["cv"]["n_splits"]) for c in main)
    search = sum(float(c.meta["search_time_seconds"]) for c in main)
    final = sum(float(c.meta["training_time_seconds"]) for c in main)
    add("entrenamiento", "ajustes_en_la_busqueda_de_las_12_combinaciones", fits, "ajustes",
        "combinaciones de hiperparametros x 5 folds; todos leen la cache")
    add("entrenamiento", "busqueda_de_hiperparametros_12_combinaciones", round(search / 60, 1), "min",
        "suma de los tiempos registrados, con todos los nucleos")
    add("entrenamiento", "ajuste_final_12_combinaciones", round(final, 1), "s")
    add("entrenamiento", "busqueda_de_las_35_configuraciones", round(
        sum(float(c.meta["search_time_seconds"]) for c in candidates) / 60, 1), "min", "incluye las mejoras propuestas")
    extraction_full = startup_s + total_cycles * parallel_s / n_cycles
    add("entrenamiento", "si_se_extrajera_el_audio_en_cada_ajuste", round(fits * extraction_full / 3600, 1), "h",
        f"{fits} ajustes x {extraction_full:.0f} s de extraccion; con la cache se extrae 1 vez")
    add("entrenamiento", "si_se_extrajera_una_vez_por_combinacion", round(12 * extraction_full / 60, 1), "min",
        "12 extracciones en lugar de 1")

    # inferencia
    results = pd.read_csv(paths.results_dir / "resultados_12_modelos.csv")
    best = results[results["designated_by_cv"]].iloc[0]
    add("inferencia", "modelo_elegido_solo_clasificar", round(1000 * best["inference_time_seconds"] / best["n_test_samples"], 3),
        "ms/ciclo", "1 965 ciclos de test en un lote, caracteristicas ya calculadas")
    bundles = load_models(cfg)
    examples_dir = paths.root / "app" / "examples"
    files = sorted(examples_dir.glob("*.wav"))
    if files:
        decoded = [decode_wav(p.read_bytes()) for p in files]
        analyze_cycle(*decoded[0], cfg)  # calentamiento
        analysis_ms = [analyze_cycle(audio, sr, cfg).elapsed_ms for audio, sr in decoded]
        add("inferencia", "interfaz_preprocesar_y_extraer", round(float(np.median(analysis_ms)), 0), "ms/ciclo",
            f"mediana de {len(files)} ciclos de ejemplo")
        for key, bundle in bundles.items():
            analyses = [analyze_cycle(audio, sr, cfg) for audio, sr in decoded]
            predict(bundle, analyses[0])
            add("inferencia", f"interfaz_clasificar_{key}", round(float(np.median(
                [predict(bundle, a).elapsed_ms for a in analyses])), 0), "ms/ciclo", "un ciclo a la vez")
    return pd.DataFrame(rows)
