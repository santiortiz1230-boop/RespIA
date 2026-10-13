"""Resumen de la auditoria del dataset ICBHI (Fase 1): tablas y figuras.

Separado de :mod:`src.carga_datos` porque solo se usa para reportar; la lectura de
datos (reutilizada por las demas fases) vive en ``data_loader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.configuracion import Config, LabelConfig
from src.carga_datos import FileInventory, IcbhiMetadata, parse_recording_name
from src.utilidades import to_posix_relative

MALFORMED_CATEGORIES = ("linea_malformada", "valor_no_numerico", "valor_no_finito", "bandera_invalida")
INVALID_REASONS = ("inicio_negativo", "fin_no_mayor_que_inicio", "fin_excede_duracion_audio")

# Paleta de figuras (tesis): azul = clase incluida; gris con trama = clase excluida.
COLOR_INCLUDED = "#2a78d6"
COLOR_EXCLUDED = "#c9c8c2"
COLOR_EXCLUDED_EDGE = "#8a8984"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e6e5e0"
FENCE_NAMES = {"mild": "leve", "extreme": "extrema"}


# Tablas


def match_filename_corrections(
    inventory: FileInventory, recordings: pd.DataFrame, file_name: str
) -> pd.DataFrame | None:
    """Cruza la lista de grabaciones renombradas del dataset con los archivos presentes.

    Para cada nombre listado busca la grabacion actual con el mismo paciente, indice,
    posicion y modo (es decir, el mismo registro con otro codigo de equipo). No se
    asume cual es el equipo correcto: solo se documenta lo que hay en disco.

    Returns:
        Tabla de correspondencias, o ``None`` si el archivo no existe en el dataset.
    """
    candidates = sorted(inventory.root.rglob(file_name))
    if not candidates:
        return None
    listed = [
        line.strip().strip("'\"")
        for line in candidates[0].read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]
    key = ["patient_id", "recording_index", "chest_location", "acquisition_mode"]
    known = recordings.dropna(subset=key)
    rows: list[dict[str, Any]] = []
    for name in listed:
        row: dict[str, Any] = {"listed_name": name, "listed_name_present": name in inventory.audio}
        try:
            parsed = parse_recording_name(name)
        except ValueError:
            rows.append({**row, "match_status": "nombre_listado_no_interpretable"})
            continue
        same = known[
            (known["patient_id"] == parsed.patient_id)
            & (known["recording_index"] == parsed.recording_index)
            & (known["chest_location"] == parsed.chest_location)
            & (known["acquisition_mode"] == parsed.acquisition_mode)
            & (known["recording_id"] != name)
        ]
        row.update(listed_equipment=parsed.equipment, n_matches=len(same))
        if len(same) == 1:
            current = same.iloc[0]
            row.update(
                match_status="encontrado_con_otro_equipo",
                current_recording_id=current["recording_id"],
                current_equipment=current["equipment"],
                sample_rate=current.get("sample_rate"),
                subtype=current.get("subtype"),
            )
        else:
            row["match_status"] = "sin_coincidencia" if same.empty else "coincidencia_ambigua"
        rows.append(row)
    return pd.DataFrame(rows)


def class_distribution(cycles: pd.DataFrame, labels: LabelConfig) -> pd.DataFrame:
    """Distribucion de etiquetas (todos los ciclos y solo validos) con pacientes y grabaciones."""
    flags = {label: key for key, label in labels.flag_to_label.items()}
    valid = cycles[cycles["is_valid"]]
    rows = []
    for label in labels.all_labels:
        all_label = cycles[cycles["label"] == label]
        valid_label = valid[valid["label"] == label]
        rows.append({
            "label": label,
            "crackle": flags[label][0],
            "wheeze": flags[label][1],
            "n_cycles": len(all_label),
            "pct_cycles": 100.0 * len(all_label) / max(len(cycles), 1),
            "n_valid_cycles": len(valid_label),
            "pct_valid_cycles": 100.0 * len(valid_label) / max(len(valid), 1),
            "n_patients": valid_label["patient_id"].nunique(),
            "n_recordings": valid_label["recording_id"].nunique(),
            f"in_{labels.label_mode}": label in labels.classes,
        })
    return pd.DataFrame(rows)


def patient_class_counts(cycles: pd.DataFrame, labels: LabelConfig) -> pd.DataFrame:
    """Ciclos validos por paciente y etiqueta (insumo para el split agrupado de la Fase 5)."""
    valid = cycles[cycles["is_valid"]]
    table = pd.crosstab(valid["patient_id"], valid["label"]).reindex(columns=list(labels.all_labels), fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.reset_index()
    table.columns.name = None
    return table


def crosstab_table(frame: pd.DataFrame, rows: str, cols: str, col_order: list[Any] | None = None) -> pd.DataFrame:
    """Tabla de contingencia con totales, lista para CSV."""
    table = pd.crosstab(frame[rows], frame[cols], margins=True, margins_name="total")
    if col_order is not None:
        table = table.reindex(columns=[c for c in col_order if c in table.columns] + ["total"], fill_value=0)
    table = table.reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def signal_quality_by_profile(recordings: pd.DataFrame, clipping_warning_fraction: float) -> pd.DataFrame:
    """Perfil tecnico por equipo, sample rate y codificacion (amplitud, DC, saturacion).

    Si existe la columna ``listed_in_filename_corrections`` se usa como grupo adicional
    para no mezclar las grabaciones renombradas con el resto del mismo equipo.
    """
    required = {"peak_abs", "rms", "dc_offset", "clipping_fraction"}
    if not required.issubset(recordings.columns):
        return pd.DataFrame()
    keys = ["equipment", "sample_rate", "subtype"]
    if "listed_in_filename_corrections" in recordings.columns:
        keys.append("listed_in_filename_corrections")
    frame = recordings.dropna(subset=["sample_rate"]).astype({"sample_rate": int})
    return (
        frame.groupby(keys)
        .agg(
            n_recordings=("recording_id", "size"),
            n_patients=("patient_id", "nunique"),
            peak_abs_min=("peak_abs", "min"),
            peak_abs_median=("peak_abs", "median"),
            rms_median=("rms", "median"),
            abs_dc_offset_median=("dc_offset", lambda s: float(s.abs().median())),
            clipping_fraction_median=("clipping_fraction", "median"),
            clipping_fraction_max=("clipping_fraction", "max"),
            n_saturated=("clipping_fraction", lambda s: int((s > clipping_warning_fraction).sum())),
        )
        .reset_index()
    )


def _describe(values: pd.Series) -> dict[str, float]:
    """Estadisticos descriptivos robustos de una serie numerica."""
    v = values.dropna().astype(float)
    if v.empty:
        return {}
    return {
        "min": float(v.min()),
        "p01": float(v.quantile(0.01)),
        "p05": float(v.quantile(0.05)),
        "mediana": float(v.median()),
        "media": float(v.mean()),
        "p95": float(v.quantile(0.95)),
        "p99": float(v.quantile(0.99)),
        "max": float(v.max()),
    }


def build_summary(meta: IcbhiMetadata, cfg: Config, corrections: pd.DataFrame | None) -> pd.DataFrame:
    """Resumen de auditoria en formato largo: ``section, metric, value, detail``."""
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, detail: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "detail": detail})

    inv, rec, cyc, iss = meta.inventory, meta.recordings, meta.cycles, meta.issues
    labels = cfg.labels
    count = lambda category: int((iss["category"] == category).sum())  # noqa: E731

    # 1-5: archivos y correspondencia WAV/TXT
    add("archivos", "carpeta_dataset", to_posix_relative(inv.root, cfg.paths.root))
    add("archivos", "archivos_wav", inv.n_audio_files)
    add("archivos", "archivos_txt_total", inv.n_txt_files)
    add("archivos", "archivos_txt_anotacion", inv.n_annotation_files)
    add("archivos", "archivos_txt_auxiliares", len(inv.auxiliary_txt), "; ".join(p.name for p in inv.auxiliary_txt))
    add("archivos", "otros_archivos", len(inv.other_files), "; ".join(p.name for p in inv.other_files))
    add("archivos", "nombres_de_grabacion_unicos", len(rec))
    paired = rec["has_audio"] & rec["has_annotation"]
    add("archivos", "pares_wav_txt", int(paired.sum()))
    no_txt = rec.loc[rec["has_audio"] & ~rec["has_annotation"], "recording_id"]
    no_wav = rec.loc[~rec["has_audio"] & rec["has_annotation"], "recording_id"]
    add("archivos", "wav_sin_anotacion", len(no_txt), "; ".join(no_txt))
    add("archivos", "anotacion_sin_wav", len(no_wav), "; ".join(no_wav))
    add("archivos", "nombres_duplicados", count("wav_duplicado") + count("txt_duplicado"))
    add("archivos", "nombres_no_interpretables", count("nombre_no_interpretable"))
    add("archivos", "codigos_desconocidos_en_nombre", count("codigo_desconocido"))

    # 6-8: sample rate, canales, duracion, integridad
    readable = rec.dropna(subset=["sample_rate"])
    add("audio", "audios_legibles", len(readable))
    add("audio", "audios_ilegibles_o_corruptos", count("audio_ilegible") + count("audio_corrupto"))
    for sr, n in readable["sample_rate"].astype(int).value_counts().sort_index().items():
        add("audio", f"sample_rate_{sr}_hz", int(n), f"{100 * n / len(readable):.1f} % de las grabaciones")
    for ch, n in readable["n_channels"].astype(int).value_counts().sort_index().items():
        add("audio", f"canales_{ch}", int(n))
    for subtype, n in readable["subtype"].value_counts().sort_index().items():
        add("audio", f"codificacion_{subtype}", int(n))
    for stat, value in _describe(readable["duration_sec"]).items():
        add("audio", f"duracion_grabacion_{stat}_s", round(value, 3))
    add("audio", "duracion_total_h", round(float(readable["duration_sec"].sum()) / 3600, 3))
    if "peak_abs" in rec.columns:
        add("audio", "audios_truncados", count("audio_truncado"))
        add("audio", "audios_con_nan_o_inf", count("audio_nan_inf"))
        add("audio", "audios_silenciosos", count("audio_silencioso"),
            f"pico < {cfg.audit.silence_peak_threshold}")
        normalized = (rec["peak_abs"] >= cfg.audit.clipping_level).sum()
        add("audio", "audios_con_pico_en_escala_completa", int(normalized),
            f"pico |x| >= {cfg.audit.clipping_level} (normalizados al pico en origen o saturados)")
        clip = rec["clipping_fraction"].fillna(0.0)
        for level in cfg.audit.clipping_report_fractions:
            add("audio", f"audios_con_mas_de_{100 * level:g}pct_muestras_en_tope", int((clip > level).sum()),
                f"fraccion de muestras con |x| >= {cfg.audit.clipping_level}")
        add("audio", "fraccion_maxima_muestras_en_tope", round(float(clip.max()), 6),
            str(rec.loc[clip.idxmax(), "recording_id"]))

    # 9: pacientes
    valid = cyc[cyc["is_valid"]]
    patients = rec["patient_id"].dropna()
    add("pacientes", "pacientes_unicos", patients.nunique(),
        f"ids {patients.astype(int).min()}-{patients.astype(int).max()}" if not patients.empty else "")
    per_patient_rec = rec.dropna(subset=["patient_id"]).groupby("patient_id").size()
    per_patient_cyc = valid.groupby("patient_id").size()
    for name, series in (("grabaciones", per_patient_rec), ("ciclos_validos", per_patient_cyc)):
        add("pacientes", f"{name}_por_paciente_min", int(series.min()))
        add("pacientes", f"{name}_por_paciente_mediana", float(series.median()))
        add("pacientes", f"{name}_por_paciente_max", int(series.max()), f"paciente {series.idxmax()}")
    for label in labels.all_labels:
        in_label = valid[valid["label"] == label]
        add("pacientes", f"pacientes_con_{label}", in_label["patient_id"].nunique())
        if not in_label.empty:
            share = in_label.groupby("patient_id").size() / len(in_label)
            add("pacientes", f"max_fraccion_de_{label}_en_un_paciente", round(float(share.max()), 4),
                f"paciente {share.idxmax()} aporta {int(share.max() * len(in_label))} de {len(in_label)} ciclos")

    # 10-11: equipos, posiciones, modo
    for column, section in (("equipment", "equipo"), ("chest_location", "posicion"), ("acquisition_mode", "modo")):
        for value, n in rec[column].value_counts().sort_index().items():
            n_cyc = int((valid[column] == value).sum())
            n_pat = valid.loc[valid[column] == value, "patient_id"].nunique()
            add("adquisicion", f"{section}_{value}", int(n), f"grabaciones; {n_cyc} ciclos validos; {n_pat} pacientes")
    if corrections is not None:
        found = corrections[corrections["match_status"] == "encontrado_con_otro_equipo"]
        add("adquisicion", "grabaciones_listadas_como_renombradas", len(corrections), cfg.icbhi.filename_corrections_file)
        add("adquisicion", "renombradas_presentes_con_nombre_listado", int(corrections["listed_name_present"].sum()))
        add("adquisicion", "renombradas_presentes_con_otro_equipo", len(found),
            "; ".join(f"{k}={v}" for k, v in found["current_equipment"].value_counts().items()))
        if not found.empty:
            profile = found.groupby(["sample_rate", "subtype"]).size()
            add("adquisicion", "renombradas_perfil_sample_rate", len(found),
                "; ".join(f"{int(sr)} Hz {st}: {n}" for (sr, st), n in profile.items()))

    # 12: anotaciones
    add("anotaciones", "filas_de_anotacion_leidas", int(rec["n_annotation_rows"].sum()))
    add("anotaciones", "lineas_malformadas", int(iss["category"].isin(MALFORMED_CATEGORIES).sum()))
    add("anotaciones", "lineas_vacias", count("linea_vacia"))
    add("anotaciones", "archivos_anotacion_sin_ciclos", count("anotacion_sin_ciclos"))
    add("anotaciones", "ciclos_totales", len(cyc))
    add("anotaciones", "ciclos_validos", len(valid))
    add("anotaciones", "ciclos_invalidos", int((~cyc["is_valid"]).sum()))
    for reason in INVALID_REASONS:
        add("anotaciones", f"invalidos_{reason}", int(cyc["invalid_reason"].str.contains(reason).sum()))
    add("anotaciones", "max_exceso_sobre_duracion_audio_s", round(float(cyc["exceeds_audio_by_sec"].max()), 6))
    add("anotaciones", "ciclos_solapados_con_el_anterior", int(cyc["overlaps_previous"].sum()),
        "informativo: no invalida el ciclo")
    for stat, value in _describe(valid["duration_sec"]).items():
        add("anotaciones", f"duracion_ciclo_{stat}_s", round(value, 4))

    # 13: etiquetas
    for label in labels.all_labels:
        n = int((valid["label"] == label).sum())
        add("etiquetas", f"ciclos_{label}", n, f"{100 * n / max(len(valid), 1):.1f} % de los ciclos validos")
    included = valid["label"].isin(labels.classes)
    excluded_counts = valid.loc[~included, "label"].value_counts()
    add("etiquetas", "label_mode", labels.label_mode, ", ".join(labels.classes))
    add("etiquetas", f"ciclos_incluidos_{labels.label_mode}", int(included.sum()))
    add("etiquetas", f"ciclos_excluidos_{labels.label_mode}", int((~included).sum()),
        "; ".join(f"{k}={v}" for k, v in excluded_counts.items()) + " (se conservan en metadatos_ciclos.csv)")
    return pd.DataFrame(rows)


def print_summary_report(summary: pd.DataFrame, issues: pd.DataFrame, title: str) -> None:
    """Imprime un resumen ``section/metric/value/detail`` y el conteo de incidencias."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for section, block in summary.groupby("section", sort=False):
        print(f"\n[{str(section).upper()}]")
        for row in block.itertuples(index=False):
            detail = f"   ({row.detail})" if isinstance(row.detail, str) and row.detail else ""
            print(f"  {row.metric:<48} {row.value}{detail}")
    print("\n[INCIDENCIAS POR SEVERIDAD Y CATEGORIA]")
    if issues.empty:
        print("  (ninguna)")
    else:
        for (severity, category), n in issues.groupby(["severity", "category"]).size().items():
            print(f"  {severity:<8} {category:<40} {n}")
    print("=" * 78 + "\n")


# Segmentacion (Fase 2)


def segment_quality_table(manifest: pd.DataFrame, labels: LabelConfig) -> pd.DataFrame:
    """Calidad de los segmentos por equipo y etiqueta (filas ``TODOS`` = total por etiqueta)."""
    ok = manifest[manifest["status"] == "ok"].assign(
        reaches_full_scale=lambda d: d["clipping_fraction"] > 0
    )

    def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return (
            frame.groupby(keys)
            .agg(
                n_cycles=("cycle_id", "size"),
                n_patients=("patient_id", "nunique"),
                duration_median_s=("segment_duration_sec", "median"),
                peak_abs_median=("peak_abs", "median"),
                rms_median=("rms", "median"),
                pct_reach_full_scale=("reaches_full_scale", lambda s: 100.0 * s.mean()),
                pct_saturated=("is_saturated", lambda s: 100.0 * s.mean()),
            )
            .reset_index()
        )

    by_equipment = aggregate(ok, ["equipment", "label"])
    overall = aggregate(ok, ["label"]).assign(equipment="TODOS")
    table = pd.concat([by_equipment, overall], ignore_index=True)
    order = {label: i for i, label in enumerate(labels.all_labels)}
    table["_order"] = table["label"].map(order)
    return table.sort_values(["equipment", "_order"]).drop(columns="_order").reset_index(drop=True)


def build_segmentation_summary(
    manifest: pd.DataFrame,
    cycles: pd.DataFrame,
    issues: pd.DataFrame,
    fences: dict[str, tuple[float, float]],
    cfg: Config,
    checks: list[tuple[str, Any, str]],
) -> pd.DataFrame:
    """Resumen de la segmentacion en formato largo: ``section, metric, value, detail``."""
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, detail: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "detail": detail})

    labels = cfg.labels
    ok = manifest[manifest["status"] == "ok"]
    errors = manifest[manifest["status"] != "ok"]
    count = lambda category: int((issues["category"] == category).sum())  # noqa: E731

    add("segmentacion", "modo_de_extraccion", "bajo_demanda",
        "intervalo [round(inicio*sr), round(fin*sr)) a sample rate nativo; sin copias en disco")
    add("segmentacion", "ciclos_en_metadatos", len(cycles))
    add("segmentacion", "ciclos_invalidos_omitidos", int((~cycles["is_valid"]).sum()))
    add("segmentacion", "grabaciones_leidas", manifest["source_audio"].nunique())
    add("segmentacion", "segmentos_extraidos_ok", len(ok))
    add("segmentacion", "segmentos_con_error", len(errors),
        "; ".join(f"{k}={v}" for k, v in errors["status_detail"].value_counts().items()))

    for category in ("segmento_vacio", "segmento_fuera_de_rango", "sample_rate_distinto", "audio_corrupto",
                     "segmento_nan_inf", "duracion_inconsistente", "segmento_silencioso"):
        add("integridad", category, count(category))
    if not ok.empty:
        add("integridad", "max_error_duracion_muestras", round(float(ok["duration_error_samples"].abs().max()), 4),
            "|duracion real - anotada| * sr; <= 1 por redondeo de los dos extremos")
        add("integridad", "muestras_por_segmento_min", int(ok["n_samples"].min()),
            str(ok.loc[ok["n_samples"].idxmin(), "cycle_id"]))
        add("integridad", "muestras_por_segmento_max", int(ok["n_samples"].max()),
            str(ok.loc[ok["n_samples"].idxmax(), "cycle_id"]))

    for name, (low, high) in fences.items():
        k = cfg.segmentation.duration_outlier_iqr_k[name]
        flagged = ok[ok[f"duration_outlier_{name}"]]
        n_low = int((flagged["segment_duration_sec"] < low).sum())
        by_label = "; ".join(f"{k_}={v}" for k_, v in flagged["label"].value_counts().items())
        add("duracion", f"cerca_{name}_k{k:g}_s", f"[{low:.3f}, {high:.3f}]", "Tukey sobre log(duracion)")
        add("duracion", f"ciclos_atipicos_{name}", len(flagged),
            f"cortos={n_low}; largos={len(flagged) - n_low}; {by_label} (informativo, no se excluyen)")

    reaches = ok["clipping_fraction"] > 0
    add("saturacion", "ciclos_que_alcanzan_escala_completa", int(reaches.sum()),
        f"{100 * reaches.mean():.1f} %; |x| >= {cfg.audit.clipping_level}")
    add("saturacion", "ciclos_saturados", int(ok["is_saturated"].sum()),
        f"> {100 * cfg.audit.clipping_warning_fraction:g} % de muestras en el tope")
    for label in labels.all_labels:
        in_label = ok[ok["label"] == label]
        if not in_label.empty:
            add("saturacion", f"pct_saturados_{label}", round(100 * float(in_label["is_saturated"].mean()), 2),
                f"{int(in_label['is_saturated'].sum())} de {len(in_label)}")
    if "has_flatline" in ok.columns:
        flat = ok[ok["has_flatline"]]
        add("saturacion", "segmentos_con_meseta", len(flat),
            f">= {1000 * cfg.segmentation.flatline_min_sec:g} ms de muestras identicas; "
            + "; ".join(f"{k}={v}" for k, v in flat["equipment"].value_counts().items())
            + " (solo se marca)")
        if not ok.empty:
            add("saturacion", "meseta_mas_larga_ms", round(float(ok["longest_flat_run_ms"].max()), 1),
                str(ok.loc[ok["longest_flat_run_ms"].idxmax(), "cycle_id"]))

    for label in labels.all_labels:
        add("etiquetas", f"segmentos_{label}", int((ok["label"] == label).sum()))
    included = ok["label"].isin(labels.classes)
    add("etiquetas", f"segmentos_utilizables_{labels.label_mode}", int(included.sum()), ", ".join(labels.classes))
    add("etiquetas", f"segmentos_excluidos_{labels.label_mode}", int((~included).sum()),
        "; ".join(f"{k}={v}" for k, v in ok.loc[~included, "label"].value_counts().items())
        + " (se conservan en el manifiesto)")

    total_samples = int(ok["n_samples"].sum())
    add("almacenamiento", "muestras_totales_en_segmentos", total_samples,
        f"{ok['segment_duration_sec'].sum() / 3600:.2f} h de audio")
    add("almacenamiento", "tamano_si_se_copiaran_float32_gb", round(total_samples * 4 / 1e9, 3),
        f"{len(ok)} archivos adicionales")
    add("almacenamiento", "tamano_si_se_copiaran_pcm16_gb", round(total_samples * 2 / 1e9, 3))

    for metric, value, detail in checks:
        add("verificaciones", metric, value, detail)
    return pd.DataFrame(rows)


# Figuras


def _style_axes(ax: Any) -> None:
    """Ejes sobrios: sin marco superior/derecho, rejilla horizontal tenue."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt_int(value: int) -> str:
    """Entero con separador de miles por espacio fino (convencion SI)."""
    return f"{value:,}".replace(",", " ")


def plot_class_distribution(distribution: pd.DataFrame, labels: LabelConfig, n_patients: int, path: Path) -> Path:
    """Barras de ciclos validos por clase; la(s) clase(s) excluida(s) del modo activo en gris con trama."""
    fig = Figure(figsize=(8.0, 4.8), layout="constrained")
    ax = fig.add_subplot()
    x = np.arange(len(distribution))
    counts = distribution["n_valid_cycles"].to_numpy()
    pcts = distribution["pct_valid_cycles"].to_numpy()
    included = distribution["label"].isin(labels.classes).to_numpy()

    for xi, n, pct, inc in zip(x, counts, pcts, included):
        ax.bar(
            xi, n, width=0.62,
            color=COLOR_INCLUDED if inc else COLOR_EXCLUDED,
            edgecolor="white" if inc else COLOR_EXCLUDED_EDGE,
            hatch=None if inc else "///", linewidth=0.8,
        )
        ax.annotate(_fmt_int(int(n)), (xi, n), xytext=(0, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK_PRIMARY)
        ax.annotate(f"{pct:.1f} %".replace(".", ","), (xi, n), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(x, [f"{r.label}\ncrackle={r.crackle} · wheeze={r.wheeze}" for r in distribution.itertuples()],
                  fontsize=9.5)
    ax.set_ylabel("Número de ciclos respiratorios", color=INK_SECONDARY)
    ax.set_ylim(0, counts.max() * 1.18)
    _style_axes(ax)
    ax.set_title(
        f"Distribución de ciclos por clase acústica — ICBHI 2017\n"
        f"{_fmt_int(int(counts.sum()))} ciclos válidos · {n_patients} pacientes",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_INCLUDED,
                  label=f"Incluida en el experimento principal ({len(labels.classes)} clases)"),
            Patch(facecolor=COLOR_EXCLUDED, edgecolor=COLOR_EXCLUDED_EDGE, hatch="///",
                  label="Excluida del experimento principal (se conserva en los datos)"),
        ],
        loc="upper right", frameon=False, fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_cycle_durations(
    cycles: pd.DataFrame, path: Path, bin_width_sec: float = 0.25, display_quantile: float = 0.999
) -> Path:
    """Histograma de la duracion de los ciclos validos, con la mediana marcada.

    El eje x se limita al cuantil ``display_quantile`` para que la cola extrema no
    comprima la figura; los ciclos fuera de escala se cuentan en la nota inferior.
    """
    durations = cycles.loc[cycles["is_valid"], "duration_sec"].to_numpy(dtype=float)
    fig = Figure(figsize=(8.0, 4.2), layout="constrained")
    ax = fig.add_subplot()
    x_max = float(np.ceil(np.quantile(durations, display_quantile)))
    bins = np.arange(0.0, x_max + bin_width_sec, bin_width_sec)
    ax.hist(durations[durations <= x_max], bins=bins, color=COLOR_INCLUDED, edgecolor="white", linewidth=0.6)
    n_outside = int((durations > x_max).sum())
    if n_outside:
        ax.text(1.0, -0.16, f"{n_outside} ciclo(s) > {x_max:g} s fuera de escala (máximo {durations.max():.2f} s)"
                .replace(".", ","), transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)
    median = float(np.median(durations))
    ax.axvline(median, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.annotate(f"mediana = {median:.2f} s".replace(".", ","), (median, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK_SECONDARY, va="top")
    ax.set_xlabel("Duración del ciclo (s)", color=INK_SECONDARY)
    ax.set_ylabel("Número de ciclos", color=INK_SECONDARY)
    _style_axes(ax)
    ax.set_title(
        f"Duración de los ciclos respiratorios anotados — ICBHI 2017\n"
        f"{_fmt_int(len(durations))} ciclos válidos · bins de {bin_width_sec:g} s".replace(".", ","),
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_duration_by_class(
    manifest: pd.DataFrame,
    labels: LabelConfig,
    fences: dict[str, tuple[float, float]],
    iqr_k: dict[str, float],
    path: Path,
    seed: int = 42,
) -> Path:
    """Cajas de duracion por clase (eje log) con las cercas de Tukey globales.

    Caja = Q1-Q3, linea = mediana, bigotes = percentiles 5 y 95; los ciclos fuera de
    la cerca mas estrecha se dibujan como puntos (desplazamiento horizontal fijo por semilla).
    """
    ok = manifest[manifest["status"] == "ok"]
    fig = Figure(figsize=(8.0, 4.6), layout="constrained")
    ax = fig.add_subplot()
    stats, colors, included = [], [], []
    for label in labels.all_labels:
        d = ok.loc[ok["label"] == label, "segment_duration_sec"].to_numpy(dtype=float)
        if d.size == 0:
            continue
        q = np.percentile(d, [5, 25, 50, 75, 95])
        stats.append({"label": f"{label}\nn = {_fmt_int(d.size)}", "whislo": q[0], "q1": q[1], "med": q[2],
                      "q3": q[3], "whishi": q[4], "fliers": []})
        inc = label in labels.classes
        included.append(inc)
        colors.append(COLOR_INCLUDED if inc else COLOR_EXCLUDED)

    parts = ax.bxp(stats, widths=0.5, patch_artist=True, showfliers=False,
                   medianprops={"linewidth": 2},
                   whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY})
    for box, color, inc in zip(parts["boxes"], colors, included):
        box.set_facecolor(color)
        box.set_edgecolor("white" if inc else COLOR_EXCLUDED_EDGE)
        if not inc:
            box.set_hatch("///")
    for median, inc in zip(parts["medians"], included):
        median.set_color("white" if inc else INK_PRIMARY)

    if fences:
        narrow = min(fences, key=lambda name: iqr_k[name])
        low, high = fences[narrow]
        rng = np.random.default_rng(seed)
        present = [label for label in labels.all_labels if (ok["label"] == label).any()]
        for position, label in enumerate(present, start=1):
            d = ok.loc[ok["label"] == label, "segment_duration_sec"].to_numpy(dtype=float)
            out = d[(d < low) | (d > high)]
            ax.scatter(position + rng.uniform(-0.12, 0.12, out.size), out, s=9, color=INK_SECONDARY,
                       alpha=0.55, linewidths=0, zorder=3)

    ax.set_yscale("log")
    all_d = ok["segment_duration_sec"].to_numpy(dtype=float)
    y_low, y_high = all_d.min() * 0.8, all_d.max() * 1.25
    ax.set_ylim(y_low, y_high)
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16], ["0,25", "0,5", "1", "2", "4", "8", "16"])
    ax.minorticks_off()
    styles = {name: ls for name, ls in zip(fences, (":", "--", "-."))}
    for name, (low, high) in fences.items():
        for value in (low, high):
            if y_low < value < y_high:
                ax.axhline(value, color=INK_SECONDARY, linestyle=styles[name], linewidth=1)
                display = FENCE_NAMES.get(name, name)
                ax.annotate(f"cerca {display}, k = {iqr_k[name]:g} ({value:.2f} s)".replace(".", ","), (1.0, value),
                            xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Duración del ciclo (s, escala log)", color=INK_SECONDARY)
    _style_axes(ax)
    ax.set_title(
        "Duración de los ciclos segmentados por clase — ICBHI 2017\n"
        "caja = Q1–Q3 · bigotes = p5–p95 · puntos = ciclos atípicos (Tukey sobre log(duración))",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Preprocesamiento (Fase 3)


COLOR_SERIES_2 = "#eb6834"  # segunda serie categorica (naranja)


def device_profile(equipment: str, sample_rate: int, renamed: bool) -> str:
    """Etiqueta legible de perfil de adquisicion: equipo, sample rate y renombrado."""
    rate = f"{sample_rate / 1000:g}".replace(".", ",")
    return f"{equipment} · {rate} kHz" + (" (renombrado)" if renamed else "")


def level_by_profile_table(qc: pd.DataFrame) -> pd.DataFrame:
    """Nivel en banda (dBFS) antes de normalizar, por perfil de adquisicion."""
    return (
        qc[(qc["status"] == "ok") & ~qc["is_silent"]]
        .groupby("device_profile")
        .agg(
            n_cycles=("cycle_id", "size"),
            n_patients=("patient_id", "nunique"),
            inband_dbfs_p25=("inband_rms_dbfs", lambda s: float(s.quantile(0.25))),
            inband_dbfs_median=("inband_rms_dbfs", "median"),
            inband_dbfs_p75=("inband_rms_dbfs", lambda s: float(s.quantile(0.75))),
            gain_db_median=("gain_db", "median"),
        )
        .sort_values("inband_dbfs_median")
        .reset_index()
    )


def build_preprocessing_summary(
    qc: pd.DataFrame, cfg: Config, checks: list[tuple[str, str, Any, str]]
) -> pd.DataFrame:
    """Resumen de la Fase 3 en formato largo: ``section, metric, value, detail``."""
    p = cfg.preprocessing
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, detail: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "detail": detail})

    add("parametros", "cadena", "mono -> remuestreo -> [ruido F9] -> pasa-banda -> normalizacion",
        "misma funcion para train, test, validacion externa e interfaz")
    add("parametros", "target_sr_hz", p.target_sr)
    add("parametros", "remuestreo", "resample_poly", f"polifasico racional, Kaiser beta={p.resample_kaiser_beta:g}")
    add("parametros", "pasa_banda_hz", f"[{p.bandpass_low_hz:g}, {p.bandpass_high_hz:g}]",
        f"Butterworth orden {p.bandpass_order}, SOS, {'fase cero (sosfiltfilt)' if p.zero_phase else 'causal'}")
    add("parametros", "normalizacion", p.normalization_method,
        f"nivel objetivo {p.normalization_target:g}; silencio si RMS en banda < {p.silence_rms_threshold:g}")

    ok = qc[qc["status"] == "ok"]
    errors = qc[qc["status"] != "ok"]
    add("ejecucion", "ciclos_procesados_ok", len(ok))
    add("ejecucion", "ciclos_con_error", len(errors),
        "; ".join(f"{k}={v}" for k, v in errors["status_detail"].value_counts().items()))
    add("ejecucion", "ciclos_silenciosos", int(ok["is_silent"].sum()))
    add("ejecucion", "salidas_con_nan_o_inf", int(ok["n_nonfinite"].sum()))
    add("ejecucion", "longitud_distinta_a_la_esperada",
        int((ok["output_n_samples"] != ok["expected_n_samples"]).sum()),
        "salida = ceil(n * up / down) del remuestreo polifasico")
    add("ejecucion", "muestras_salida_min", int(ok["output_n_samples"].min()),
        f"{ok['output_n_samples'].min() / p.target_sr:.3f} s")
    deviation = (ok.loc[~ok["is_silent"], "output_rms"] - p.normalization_target).abs().max()
    add("ejecucion", "rms_salida_max_desviacion", float(deviation), "|RMS de salida - objetivo|")
    add("niveles", "nivel_en_banda_min_dbfs", round(float(ok["inband_rms_dbfs"].min()), 2),
        str(ok.loc[ok["inband_rms_dbfs"].idxmin(), "cycle_id"]))
    add("niveles", "nivel_en_banda_mediana_dbfs", round(float(ok["inband_rms_dbfs"].median()), 2))
    add("niveles", "nivel_en_banda_max_dbfs", round(float(ok["inband_rms_dbfs"].max()), 2),
        str(ok.loc[ok["inband_rms_dbfs"].idxmax(), "cycle_id"]))
    spread = ok.groupby("device_profile")["inband_rms_dbfs"].median()
    add("niveles", "rango_de_medianas_entre_perfiles_db", round(float(spread.max() - spread.min()), 2),
        f"{spread.idxmin()} ({spread.min():.1f}) a {spread.idxmax()} ({spread.max():.1f}) dBFS")
    add("niveles", "factor_de_cresta_mediana_db", round(float(ok["crest_factor_db"].median()), 2),
        "pico/RMS tras normalizar: dinamica interna conservada")
    for section, metric, value, detail in checks:
        add(section, metric, value, detail)
    return pd.DataFrame(rows)


def plot_filter_response(sos: np.ndarray, cfg: Config, path: Path) -> Path:
    """Respuesta en magnitud del pasa-banda: una pasada y fase cero (efectiva)."""
    from scipy.signal import sosfreqz  # noqa: PLC0415

    p = cfg.preprocessing
    freqs = np.geomspace(10, p.target_sr / 2 * 0.999, 2000)
    _, h = sosfreqz(np.array(sos), worN=freqs, fs=p.target_sr)
    single = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    fig = Figure(figsize=(8.0, 4.4), layout="constrained")
    ax = fig.add_subplot()
    ax.axvspan(p.bandpass_low_hz, p.bandpass_high_hz, color=COLOR_INCLUDED, alpha=0.07, linewidth=0)
    ax.plot(freqs, single, color=INK_SECONDARY, linestyle="--", linewidth=1.5, label="Una pasada (sosfilt)")
    ax.plot(freqs, 2 * single, color=COLOR_INCLUDED, linewidth=2, label="Fase cero, efectiva (sosfiltfilt)")
    for level in (-3, -6):
        ax.axhline(level, color=GRID_COLOR, linewidth=1)
        ax.annotate(f"{level} dB", (freqs[0], level), xytext=(2, 2), textcoords="offset points",
                    fontsize=8, color=INK_SECONDARY)
    for fc in (p.bandpass_low_hz, p.bandpass_high_hz):
        ax.axvline(fc, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
        ax.annotate(f"{_fmt_int(int(fc))} Hz", (fc, -75), xytext=(3, 0), textcoords="offset points",
                    fontsize=8.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frecuencia (Hz, escala log)", color=INK_SECONDARY)
    ax.set_ylabel("Magnitud (dB)", color=INK_SECONDARY)
    _style_axes(ax)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.45), frameon=False, fontsize=9)
    ax.set_title(
        f"Filtro pasa-banda Butterworth orden {p.bandpass_order} (SOS) — fs = {_fmt_int(p.target_sr)} Hz\n"
        f"banda de interés {p.bandpass_low_hz:g}–{_fmt_int(int(p.bandpass_high_hz))} Hz sombreada",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_preprocessing_example(
    raw: np.ndarray, raw_sr: int, processed: np.ndarray, processed_sr: int, title: str, cfg: Config, path: Path
) -> Path:
    """Ejemplo de un ciclo: forma de onda original, preprocesada y densidad espectral."""
    from scipy.signal import welch  # noqa: PLC0415

    p = cfg.preprocessing
    fig = Figure(figsize=(8.0, 7.2), layout="constrained")
    ax_raw, ax_proc, ax_psd = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1, 1, 1.25]})
    t_raw = np.arange(raw.shape[0]) / raw_sr
    t_proc = np.arange(processed.shape[0]) / processed_sr
    ax_raw.plot(t_raw, raw, color=COLOR_SERIES_2, linewidth=0.5)
    ax_raw.set_title(f"a) Original — {_fmt_int(raw_sr)} Hz, amplitud del archivo", loc="left", fontsize=10,
                     color=INK_PRIMARY)
    ax_proc.plot(t_proc, processed, color=COLOR_INCLUDED, linewidth=0.5)
    band = f"{p.bandpass_low_hz:g}–{_fmt_int(int(p.bandpass_high_hz))} Hz"
    ax_proc.set_title(
        f"b) Preprocesada — {_fmt_int(processed_sr)} Hz, pasa-banda {band}, "
        f"RMS = {p.normalization_target:g}".replace(".", ","),
        loc="left", fontsize=10, color=INK_PRIMARY)
    for ax in (ax_raw, ax_proc):
        ax.set_xlim(0, max(t_raw[-1], t_proc[-1]))
        ax.set_ylabel("Amplitud", color=INK_SECONDARY)
        _style_axes(ax)
    ax_proc.set_xlabel("Tiempo (s)", color=INK_SECONDARY)

    for signal, sr, color, name in ((raw, raw_sr, COLOR_SERIES_2, "Original"),
                                    (processed, processed_sr, COLOR_INCLUDED, "Preprocesada")):
        f, pxx = welch(signal, fs=sr, nperseg=min(2048, len(signal)))
        keep = (f >= 20) & (f <= processed_sr / 2)
        db = 10 * np.log10(np.maximum(pxx[keep], 1e-20))
        ax_psd.plot(f[keep], db - db.max(), color=color, linewidth=1.6, label=name)
    ax_psd.axvspan(p.bandpass_low_hz, p.bandpass_high_hz, color=COLOR_INCLUDED, alpha=0.07, linewidth=0)
    ax_psd.set_xscale("log")
    ax_psd.set_ylim(-90, 5)
    ax_psd.set_xlabel("Frecuencia (Hz, escala log)", color=INK_SECONDARY)
    ax_psd.set_ylabel("DEP (dB re. máximo)", color=INK_SECONDARY)
    ax_psd.set_title("c) Densidad espectral de potencia (Welch)", loc="left", fontsize=10, color=INK_PRIMARY)
    ax_psd.legend(loc="lower left", frameon=False, fontsize=9)
    _style_axes(ax_psd)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def plot_level_by_profile(qc: pd.DataFrame, cfg: Config, path: Path) -> Path:
    """Nivel en banda antes de normalizar por perfil de adquisicion (evidencia de la decision)."""
    data = qc[(qc["status"] == "ok") & ~qc["is_silent"]]
    order = data.groupby("device_profile")["inband_rms_dbfs"].median().sort_values().index.tolist()
    stats = []
    for profile in order:
        d = data.loc[data["device_profile"] == profile, "inband_rms_dbfs"].to_numpy(dtype=float)
        q = np.percentile(d, [5, 25, 50, 75, 95])
        stats.append({"label": f"{profile}\nn = {_fmt_int(d.size)}", "whislo": q[0], "q1": q[1], "med": q[2],
                      "q3": q[3], "whishi": q[4], "fliers": []})
    fig = Figure(figsize=(8.0, 5.2), layout="constrained")
    ax = fig.add_subplot()
    parts = ax.bxp(stats, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                   medianprops={"color": "white", "linewidth": 2},
                   whiskerprops={"color": INK_SECONDARY}, capprops={"color": INK_SECONDARY})
    for box in parts["boxes"]:
        box.set_facecolor(COLOR_INCLUDED)
        box.set_edgecolor("white")
    target_db = 20 * np.log10(cfg.preprocessing.normalization_target)
    ax.axvline(target_db, color=INK_PRIMARY, linestyle="--", linewidth=1.2)
    ax.annotate(f"todos los ciclos tras normalizar: {target_db:.0f} dBFS", (target_db, len(stats) + 0.5),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5,
                color=INK_PRIMARY)
    ax.set_xlabel("RMS en banda 100–2000 Hz antes de normalizar (dBFS)", color=INK_SECONDARY)
    ax.tick_params(axis="y", labelsize=9)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_ylim(0.4, len(stats) + 0.9)
    ax.set_title(
        "Nivel de los ciclos según el perfil de adquisición — ICBHI 2017\n"
        "caja = Q1–Q3 · bigotes = p5–p95",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Caracteristicas (Fase 4)


def plot_feature_representations(
    log_mel: np.ndarray,
    mfcc: np.ndarray,
    subband_energy: dict[str, float],
    subband_ranges: dict[str, tuple[float, float]],
    rms: np.ndarray,
    hop_sec: float,
    rms_hop_sec: float,
    mel_edges_hz: tuple[float, float],
    dims: dict[str, int],
    title: str,
    path: Path,
) -> Path:
    """Las 4 representaciones de un mismo ciclo y la dimension del vector que producen."""
    from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415

    fig = Figure(figsize=(9.0, 7.4), layout="constrained")
    (ax_mel, ax_mfcc), (ax_dwt, ax_rms) = fig.subplots(2, 2)

    t_end = log_mel.shape[1] * hop_sec
    img = ax_mel.imshow(log_mel, origin="lower", aspect="auto", cmap="Blues",
                        extent=(0, t_end, 0, log_mel.shape[0]))
    ax_mel.set_yticks([0, log_mel.shape[0]], [f"{mel_edges_hz[0]:g} Hz", f"{_fmt_int(int(mel_edges_hz[1]))} Hz"])
    ax_mel.set_xlabel("Tiempo (s)", color=INK_SECONDARY)
    ax_mel.set_ylabel("Banda Mel", color=INK_SECONDARY)
    ax_mel.set_title(f"a) log-Mel, {log_mel.shape[0]} bandas → vector de {dims['mel']}", loc="left",
                     fontsize=10, color=INK_PRIMARY)
    fig.colorbar(img, ax=ax_mel, label="dB", shrink=0.85)

    coeffs = mfcc[1:]
    limit = float(np.max(np.abs(coeffs))) or 1.0
    img2 = ax_mfcc.imshow(coeffs, origin="lower", aspect="auto", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
                          extent=(0, t_end, 0.5, coeffs.shape[0] + 0.5))
    ax_mfcc.set_xlabel("Tiempo (s)", color=INK_SECONDARY)
    ax_mfcc.set_ylabel("Coeficiente (c1–c12)", color=INK_SECONDARY)
    ax_mfcc.set_title(f"b) MFCC c0–c12 (c0 no se dibuja) → vector de {dims['mfcc']}", loc="left",
                      fontsize=10, color=INK_PRIMARY)
    fig.colorbar(img2, ax=ax_mfcc, shrink=0.85)

    bands = list(subband_energy)
    positions = np.arange(len(bands))
    ax_dwt.barh(positions, [100 * subband_energy[b] for b in bands], height=0.65, color=COLOR_INCLUDED)
    def fmt_hz(value: float) -> str:
        return _fmt_int(int(value)) if float(value).is_integer() else f"{value:g}".replace(".", ",")

    ax_dwt.set_yticks(positions, [f"{b}  {fmt_hz(subband_ranges[b][0])}–{fmt_hz(subband_ranges[b][1])} Hz"
                                  for b in bands], fontsize=8.5)
    ax_dwt.invert_yaxis()
    ax_dwt.set_xlabel("Energía relativa (%)", color=INK_SECONDARY)
    ax_dwt.set_title(f"c) DWT db4, nivel 7 → vector de {dims['wavelet']}", loc="left", fontsize=10,
                     color=INK_PRIMARY)
    _style_axes(ax_dwt)
    ax_dwt.yaxis.grid(False)
    ax_dwt.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

    t_rms = np.arange(rms.shape[0]) * rms_hop_sec
    p25, median, p75 = np.percentile(rms, [25, 50, 75])
    ax_rms.axhspan(p25, p75, color=COLOR_INCLUDED, alpha=0.12, linewidth=0)
    ax_rms.plot(t_rms, rms, color=COLOR_INCLUDED, linewidth=1.4)
    ax_rms.axhline(median, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax_rms.text(0.99, 0.97, "línea discontinua = mediana\nbanda sombreada = p25–p75", transform=ax_rms.transAxes,
                ha="right", va="top", fontsize=8, color=INK_SECONDARY)
    peak = int(np.argmax(rms))
    ax_rms.scatter([t_rms[peak]], [rms[peak]], s=24, color=INK_PRIMARY, zorder=3)
    ax_rms.annotate("máx", (t_rms[peak], rms[peak]), xytext=(5, -2), textcoords="offset points", fontsize=8,
                    color=INK_PRIMARY)
    ax_rms.set_xlabel("Tiempo (s)", color=INK_SECONDARY)
    ax_rms.set_ylabel("RMS por trama", color=INK_SECONDARY)
    ax_rms.set_title(f"d) RMS por trama → vector de {dims['rms']}", loc="left", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax_rms)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Particion train/test (Fase 5)


def plot_split_distribution(summary: pd.DataFrame, labels: LabelConfig, path: Path) -> Path:
    """Distribucion de clases en train y test frente a la global (modo con todas las etiquetas)."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    mode = max(labels.modes, key=lambda m: len(labels.modes[m]))
    data = summary[summary["label_mode"] == mode]
    classes = list(labels.modes[mode])
    fig = Figure(figsize=(8.4, 4.9), layout="constrained")
    ax = fig.add_subplot()
    width = 0.36
    x = np.arange(len(classes))
    for offset, split, color in ((-width / 2, "train", COLOR_INCLUDED), (width / 2, "test", COLOR_SERIES_2)):
        rows = data[data["split"] == split].set_index("label")
        pct = rows.loc[classes, "pct_of_split"].to_numpy()
        n = rows.loc[classes, "n_cycles"].to_numpy()
        pts = rows.loc[classes, "n_patients"].to_numpy()
        ax.bar(x + offset, pct, width=width * 0.94, color=color, edgecolor="white", linewidth=0.8)
        for xi, p, ni, pi in zip(x + offset, pct, n, pts):
            ax.annotate(f"{_fmt_int(int(ni))}\n{pi} pac.", (xi, p), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    total = data[data["split"] == "total"].set_index("label")
    for xi, label in zip(x, classes):
        g = float(total.loc[label, "pct_of_split"])
        ax.plot([xi - width, xi + width], [g, g], color=INK_PRIMARY, linewidth=1.6)
    ax.set_xticks(x, classes)
    ax.set_ylabel("% de los ciclos de la partición", color=INK_SECONDARY)
    ax.set_ylim(0, float(data["pct_of_split"][data["label"] != "TOTAL"].max()) * 1.25)
    _style_axes(ax)
    train_tot = data[(data["split"] == "train") & (data["label"] == "TOTAL")].iloc[0]
    test_tot = data[(data["split"] == "test") & (data["label"] == "TOTAL")].iloc[0]
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_INCLUDED, label=f"Train: {_fmt_int(int(train_tot['n_cycles']))} ciclos, "
                                                  f"{int(train_tot['n_patients'])} pacientes"),
            Patch(facecolor=COLOR_SERIES_2, label=f"Test: {_fmt_int(int(test_tot['n_cycles']))} ciclos, "
                                                  f"{int(test_tot['n_patients'])} pacientes"),
            Line2D([0], [0], color=INK_PRIMARY, linewidth=1.6, label="Distribución global"),
        ],
        loc="upper right", frameon=False, fontsize=9,
    )
    ax.set_title(
        "Partición 70/30 agrupada por paciente — ICBHI 2017\n"
        "ningún paciente aparece en train y test · AMBOS se excluye del experimento de 3 clases",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Entrenamiento (Fase 6)


CLASSIFIER_STYLE = {
    "random_forest": ("Random Forest", "#2a78d6", "o"),
    "svm_rbf": ("SVM-RBF", "#eb6834", "s"),
    "knn": ("KNN", "#1baf7a", "^"),
}
REPRESENTATION_LABELS = {"mfcc": "MFCC", "mel": "Mel", "wavelet": "Wavelet", "rms": "RMS"}


def plot_cv_f1(summary: pd.DataFrame, n_splits: int, n_samples: int, n_patients: int, target: float,
               path: Path, baseline: float | None = None) -> Path:
    """F1 ponderada de CV (media +/- desviacion entre folds) de las 12 combinaciones.

    ``baseline``: F1 de la linea base trivial (clase mayoritaria) con los mismos folds.
    """
    reps = [r for r in REPRESENTATION_LABELS if r in set(summary["representation"])]
    fig = Figure(figsize=(8.4, 4.8), layout="constrained")
    ax = fig.add_subplot()
    offsets = {"random_forest": -0.22, "svm_rbf": 0.0, "knn": 0.22}
    for clf, (label, color, marker) in CLASSIFIER_STYLE.items():
        rows = summary[summary["classifier"] == clf].set_index("representation")
        xs = [i + offsets[clf] for i, r in enumerate(reps) if r in rows.index]
        ys = [float(rows.loc[r, "cv_f1_weighted"]) for r in reps if r in rows.index]
        es = [float(rows.loc[r, "cv_f1_weighted_std"]) for r in reps if r in rows.index]
        ax.errorbar(xs, ys, yerr=es, fmt=marker, color=color, markersize=7, capsize=3, linewidth=1.2,
                    markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}".replace(".", ","), (x, y), xytext=(6, -3), textcoords="offset points",
                        fontsize=7.5, color=INK_SECONDARY)
    ax.axhline(target, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.annotate(f"meta del proyecto F1 = {target:.2f} (se evalúa en test, Fase 7)".replace(".", ","),
                (len(reps) - 0.5, target), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.5,
                color=INK_PRIMARY)
    if baseline is not None:
        ax.axhline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
        ax.annotate(f"línea base trivial: predecir siempre la clase mayoritaria = {baseline:.3f}".replace(".", ","),
                    (len(reps) - 0.5, baseline), xytext=(0, -11), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(range(len(reps)), [REPRESENTATION_LABELS[r] for r in reps])
    ax.set_xlim(-0.5, len(reps) - 0.5)
    low = float((summary["cv_f1_weighted"] - summary["cv_f1_weighted_std"]).min())
    if baseline is not None:
        low = min(low, baseline - 0.02)
    high = float((summary["cv_f1_weighted"] + summary["cv_f1_weighted_std"]).max())
    ax.set_ylim(max(0.0, low - 0.05), max(target, high) + 0.05)
    ax.set_ylabel("F1 ponderada (validación cruzada)", color=INK_SECONDARY)
    _style_axes(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncols=3)
    ax.set_title(
        "Validación cruzada agrupada por paciente — solo el 70 % de train\n"
        f"media ± desviación entre {n_splits} folds · {_fmt_int(n_samples)} ciclos de {n_patients} pacientes",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


# Mejoras propuestas (Fase 6B)


def plot_improvements(table: pd.DataFrame, best_original: float, path: Path) -> Path:
    """Antes -> despues (original vs balanceado) por combinacion y fusion, en F1 de CV."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    names = {"random_forest": "RF", "svm_rbf": "SVM", "knn": "KNN"}
    reps = {"mfcc": "MFCC", "mel": "Mel", "wavelet": "Wavelet", "rms": "RMS", "fusion": "Fusión"}
    original = table[table["variant"] == "original"].set_index(["representation", "classifier"])
    balanced = table[table["variant"] == "balanced"].set_index(["representation", "classifier"])
    fusion = table[table["variant"] == "fusion"].set_index(["representation", "classifier"])
    items = [(r, c) for r in ("mfcc", "mel", "wavelet", "rms") for c in ("random_forest", "svm_rbf", "knn")]
    items += [("fusion", c) for c in ("random_forest", "svm_rbf", "knn") if ("fusion", c) in fusion.index]
    fig = Figure(figsize=(8.4, 6.4), layout="constrained")
    ax = fig.add_subplot()
    for y, key in enumerate(items):
        label_y = len(items) - 1 - y
        if key[0] == "fusion":
            value = float(fusion.loc[key, "cv_f1_weighted"])
            ax.scatter([value], [label_y], s=60, marker="D", color=COLOR_SERIES_2, zorder=3)
            continue
        before = float(original.loc[key, "cv_f1_weighted"])
        ax.scatter([before], [label_y], s=45, color=GRID_COLOR, edgecolor=INK_SECONDARY, linewidth=0.8, zorder=3)
        if key in balanced.index:
            after = float(balanced.loc[key, "cv_f1_weighted"])
            ax.annotate("", xy=(after, label_y), xytext=(before, label_y),
                        arrowprops={"arrowstyle": "->", "color": COLOR_INCLUDED, "lw": 1.4})
            ax.scatter([after], [label_y], s=45, color=COLOR_INCLUDED, zorder=3)
    ax.set_yticks(range(len(items)), [f"{reps[r]} + {names[c]}" for r, c in reversed(items)], fontsize=9)
    ax.axvline(best_original, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.annotate(f"mejor original = {best_original:.3f}".replace(".", ","), (best_original, len(items) - 0.4),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK_PRIMARY, va="center")
    ax.set_xlabel("F1 ponderada (validación cruzada, solo train)", color=INK_SECONDARY)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRID_COLOR, markeredgecolor=INK_SECONDARY,
               markersize=7, label="Original (experimento principal)"),
        Line2D([0], [0], marker="o", color=COLOR_INCLUDED, markersize=7, label="Con clases balanceadas"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLOR_SERIES_2, markersize=7,
               label="Fusión de las 4 huellas"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=3, frameon=False, fontsize=8.5)
    ax.set_title("Mejoras propuestas frente al experimento principal — solo validación cruzada\n"
                 "mismos 5 folds agrupados por paciente; el test sigue sin usarse",
                 loc="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path


def improvement_label(representation: str, classifier: str, variant: str) -> str:
    """Etiqueta legible de una configuracion (experimento principal o mejora)."""
    reps = {"mfcc": "MFCC", "mel": "Mel", "wavelet": "Wavelet", "rms": "RMS", "fusion": "Fusión",
            "descriptors": "Descriptores", "fusion_plus": "Fusión + descriptores"}
    clfs = {"random_forest": "RF", "svm_rbf": "SVM", "knn": "KNN", "vote": "voto de los 3"}
    suffix = " (balanceado)" if "balanced" in variant else ""
    return f"{reps.get(representation, representation)} · {clfs.get(classifier, classifier)}{suffix}"


def plot_improvement_ranking(table: pd.DataFrame, best_original: float, baseline: float | None, path: Path,
                             top: int = 16) -> Path:
    """Ranking de las mejores configuraciones por F1 de CV (media +/- desviacion entre folds)."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    ranked = table.sort_values("cv_f1_weighted", ascending=False).head(top).iloc[::-1].reset_index(drop=True)
    colors = {"original": INK_SECONDARY, "voting": COLOR_SERIES_2}
    fig = Figure(figsize=(8.6, 0.36 * len(ranked) + 1.9), layout="constrained")
    ax = fig.add_subplot()
    for y, row in ranked.iterrows():
        color = colors.get(row["variant"], COLOR_INCLUDED)
        marker = {"original": "o", "voting": "D"}.get(row["variant"], "s")
        ax.errorbar(row["cv_f1_weighted"], y, xerr=row["cv_f1_weighted_std"], fmt=marker, color=color,
                    markersize=6.5, capsize=2.5, linewidth=1, markeredgecolor="white", markeredgewidth=1)
        ax.annotate(f"{row['cv_f1_weighted']:.3f}".replace(".", ","),
                    (row["cv_f1_weighted"] + row["cv_f1_weighted_std"], y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(range(len(ranked)), [improvement_label(r.representation, r.classifier, r.variant)
                                       for r in ranked.itertuples()], fontsize=9)
    ax.axvline(best_original, color=INK_PRIMARY, linestyle="--", linewidth=1)
    if baseline is not None:
        ax.axvline(baseline, color=INK_SECONDARY, linestyle=":", linewidth=1.2)
    ax.set_xlabel("F1 ponderada (validación cruzada, solo train)", color=INK_SECONDARY)
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.yaxis.grid(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_SECONDARY, markersize=7,
               label="Experimento principal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_INCLUDED, markersize=7,
               label="Mejora propuesta"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLOR_SERIES_2, markersize=7,
               label="Voto de RF + SVM + KNN"),
        Line2D([0], [0], color=INK_PRIMARY, linestyle="--", label=f"Mejor original ({best_original:.3f})"
               .replace(".", ",")),
    ]
    if baseline is not None:
        handles.append(Line2D([0], [0], color=INK_SECONDARY, linestyle=":",
                              label=f"Línea base trivial ({baseline:.3f})".replace(".", ",")))
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"Las {len(ranked)} mejores configuraciones en validación cruzada (media ± desviación entre 5 folds)\n"
                 "CV agrupada por paciente, solo train; el test sigue sin usarse",
                 x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    return path
