"""Carga y validacion de la configuracion unica del proyecto (``config.yaml``).

Todas las rutas se resuelven con :mod:`pathlib` respecto a la carpeta del archivo
de configuracion, de modo que el proyecto funciona igual en Windows y Linux sin
rutas absolutas en el codigo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
CONFIG_ENV_VAR: str = "RESPIA_CONFIG"


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto (todas absolutas tras la carga)."""

    root: Path
    data_dir: Path
    output_dir: Path
    results_dir: Path
    model_dir: Path
    raw_icbhi_dir: Path
    raw_fraiwan_dir: Path

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def audit_dir(self) -> Path:
        return self.results_dir / "audit"

    def ensure_output_dirs(self) -> None:
        """Crea (si no existen) las carpetas de salida. Nunca toca los datos crudos."""
        for directory in (
            self.processed_dir,
            self.features_dir,
            self.results_dir,
            self.figures_dir,
            self.audit_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IcbhiConfig:
    audio_extension: str
    annotation_extension: str
    filename_corrections_file: str
    known_chest_locations: tuple[str, ...]
    known_acquisition_modes: tuple[str, ...]
    known_equipment: tuple[str, ...]
    time_tolerance_sec: float


@dataclass(frozen=True)
class LabelConfig:
    """Correspondencia (crackle, wheeze) -> etiqueta y modos de experimento."""

    flag_to_label: dict[tuple[int, int], str]
    label_mode: str
    modes: dict[str, tuple[str, ...]]

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Todas las etiquetas definidas, en el orden del archivo de configuracion."""
        return tuple(self.flag_to_label.values())

    @property
    def classes(self) -> tuple[str, ...]:
        """Clases (y su orden canonico) del modo de experimento activo."""
        return self.modes[self.label_mode]

    def label_for(self, crackle: int, wheeze: int) -> str:
        """Devuelve la etiqueta de un ciclo segun sus banderas crackle/wheeze.

        Raises:
            KeyError: si la combinacion de banderas no esta definida.
        """
        return self.flag_to_label[(int(crackle), int(wheeze))]


@dataclass(frozen=True)
class SegmentationConfig:
    duration_outlier_iqr_k: dict[str, float]
    flatline_min_sec: float
    consistency_check_cycles: int


NORMALIZATION_METHODS: tuple[str, ...] = ("rms", "peak", "none")
KNOWN_REPRESENTATIONS: tuple[str, ...] = ("mfcc", "mel", "wavelet", "rms")


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convierte una duracion de trama a muestras exigiendo un resultado entero exacto."""
    value = seconds * sample_rate
    rounded = int(round(value))
    if abs(value - rounded) > 1e-6 or rounded < 1:
        raise ValueError(f"{seconds} s a {sample_rate} Hz = {value} muestras: debe ser un entero >= 1.")
    return rounded


@dataclass(frozen=True)
class FeatureConfig:
    representations: tuple[str, ...]
    n_jobs: int
    stft_window_sec: float
    stft_hop_sec: float
    stft_window: str
    n_mels: int
    mel_fmin_hz: float
    mel_fmax_hz: float
    mel_htk: bool
    mel_norm: str
    mel_power: float
    mel_amin: float
    n_mfcc: int
    wavelet_name: str
    wavelet_level: int
    wavelet_mode: str
    rms_frame_sec: float
    rms_hop_sec: float
    constant_std_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class PreprocessingConfig:
    target_sr: int
    mono: str
    resample_kaiser_beta: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    bandpass_order: int
    zero_phase: bool
    normalization_method: str
    normalization_target: float
    silence_rms_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    n_candidates: int
    balance_by: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    scoring: str
    preferred_n_splits: int
    fallback_n_splits: int
    shuffle: bool
    min_patients_per_class_per_fold: int
    min_cycles_per_class_per_fold: int
    n_jobs: int


@dataclass(frozen=True)
class ModelSpec:
    """Parametros fijos, grid de busqueda y ajustes finales de un clasificador."""

    fixed: dict[str, Any]
    grid: dict[str, list[Any]]
    final_params: dict[str, Any]
    calibration: str | None = None

    @property
    def n_grid_candidates(self) -> int:
        total = 1
        for values in self.grid.values():
            total *= len(values)
        return total


COMPOSITE_REPRESENTATIONS: tuple[str, ...] = ("fusion", "descriptors", "fusion_plus")


@dataclass(frozen=True)
class DescriptorConfig:
    """Parametros de los descriptores especificos (MEJORA PROPUESTA; umbrales fijados a priori)."""

    tonal_median_filter_bins: int
    tonal_prominence_db: float
    tonal_freq_tolerance_bins: int
    band_edges_hz: tuple[float, ...]
    rolloff_fraction: float
    envelope_frame_sec: float
    envelope_hop_sec: float
    peak_mad_factor: float
    peak_min_distance_sec: float
    zcr_frame_sec: float
    zcr_hop_sec: float

    def as_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class RobustnessConfig:
    """Diseno de la prueba de robustez al ruido (Fase 9), fijado antes de ejecutarla."""

    noise: str
    snr_db: tuple[float, ...]
    references: tuple[str, ...]
    repetitions: int
    n_jobs: int

    def as_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ExternalValidationConfig:
    """Estrategia de la validacion externa por grabacion (Fase 10), fijada antes de evaluar."""

    window_sec: float
    aggregation: str
    primary_filter: str
    label_map: dict[str, str]
    excluded: dict[str, str]
    n_jobs: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StatisticsConfig:
    n_bootstrap: int
    confidence: float


@dataclass(frozen=True)
class AppConfig:
    default_model: str
    min_duration_sec: float
    typical_max_duration_sec: float
    max_duration_sec: float
    clipping_warning_fraction: float
    examples_per_class: int


@dataclass(frozen=True)
class ImprovementVariant:
    """Mejora propuesta evaluada solo con CV de train (fuera del experimento principal)."""

    name: str
    description: str
    representations: tuple[str, ...]
    classifiers: tuple[str, ...]
    fixed_overrides: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AuditConfig:
    signal_check: bool
    clipping_level: float
    clipping_warning_fraction: float
    clipping_report_fractions: tuple[float, ...]
    silence_peak_threshold: float


@dataclass(frozen=True)
class Config:
    project_name: str
    experiment_version: str
    random_state: int
    paths: Paths
    icbhi: IcbhiConfig
    labels: LabelConfig
    segmentation: SegmentationConfig
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    split: SplitConfig
    training: TrainingConfig
    models: dict[str, ModelSpec]
    best_model_selection: str
    designated_main: dict[str, str]
    designated_prototype: dict[str, str]
    improvements: dict[str, ImprovementVariant]
    descriptors: DescriptorConfig
    robustness: RobustnessConfig
    external: ExternalValidationConfig
    statistics: StatisticsConfig
    app: AppConfig
    audit: AuditConfig
    source_file: Path
    raw: dict[str, Any]


def _resolve(base: Path, value: str | os.PathLike[str]) -> Path:
    """Resuelve ``value`` respecto a ``base`` salvo que ya sea absoluta."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    """Obtiene una clave obligatoria con un mensaje de error claro."""
    if key not in section:
        raise ValueError(f"Falta la clave obligatoria '{where}.{key}' en la configuracion.")
    return section[key]


def _parse_labels(section: dict[str, Any]) -> LabelConfig:
    mapping = _require(section, "mapping", "labels")
    flag_to_label: dict[tuple[int, int], str] = {}
    for label, flags in mapping.items():
        if not isinstance(flags, (list, tuple)) or len(flags) != 2 or any(f not in (0, 1) for f in flags):
            raise ValueError(f"labels.mapping.{label} debe ser [crackle, wheeze] con valores 0/1; recibido {flags!r}.")
        key = (int(flags[0]), int(flags[1]))
        if key in flag_to_label:
            raise ValueError(f"Combinacion {key} asignada a dos etiquetas: {flag_to_label[key]} y {label}.")
        flag_to_label[key] = str(label)
    if len(flag_to_label) != 4:
        raise ValueError("labels.mapping debe cubrir las 4 combinaciones (crackle, wheeze).")

    modes = {str(k): tuple(str(c) for c in v) for k, v in _require(section, "modes", "labels").items()}
    for mode, classes in modes.items():
        unknown = set(classes) - set(flag_to_label.values())
        if unknown:
            raise ValueError(f"labels.modes.{mode} contiene etiquetas no definidas: {sorted(unknown)}.")
        if len(set(classes)) != len(classes):
            raise ValueError(f"labels.modes.{mode} contiene etiquetas repetidas.")

    label_mode = str(_require(section, "label_mode", "labels"))
    if label_mode not in modes:
        raise ValueError(f"labels.label_mode='{label_mode}' no existe en labels.modes ({sorted(modes)}).")
    return LabelConfig(flag_to_label=flag_to_label, label_mode=label_mode, modes=modes)


def _parse_preprocessing(section: dict[str, Any]) -> PreprocessingConfig:
    bandpass = _require(section, "bandpass", "preprocessing")
    normalization = _require(section, "normalization", "preprocessing")
    cfg = PreprocessingConfig(
        target_sr=int(_require(section, "target_sr", "preprocessing")),
        mono=str(_require(section, "mono", "preprocessing")),
        resample_kaiser_beta=float(_require(section, "resample_kaiser_beta", "preprocessing")),
        bandpass_low_hz=float(_require(bandpass, "low_hz", "preprocessing.bandpass")),
        bandpass_high_hz=float(_require(bandpass, "high_hz", "preprocessing.bandpass")),
        bandpass_order=int(_require(bandpass, "order", "preprocessing.bandpass")),
        zero_phase=bool(_require(bandpass, "zero_phase", "preprocessing.bandpass")),
        normalization_method=str(_require(normalization, "method", "preprocessing.normalization")),
        normalization_target=float(_require(normalization, "target_level", "preprocessing.normalization")),
        silence_rms_threshold=float(_require(normalization, "silence_rms_threshold", "preprocessing.normalization")),
    )
    nyquist = cfg.target_sr / 2
    if not 0 < cfg.bandpass_low_hz < cfg.bandpass_high_hz < nyquist:
        raise ValueError(
            f"Se requiere 0 < low_hz < high_hz < Nyquist ({nyquist:g} Hz); "
            f"recibido [{cfg.bandpass_low_hz:g}, {cfg.bandpass_high_hz:g}] Hz."
        )
    if cfg.bandpass_order < 1:
        raise ValueError("preprocessing.bandpass.order debe ser >= 1.")
    if cfg.mono != "mean":
        raise ValueError("preprocessing.mono solo admite 'mean'.")
    if cfg.normalization_method not in NORMALIZATION_METHODS:
        raise ValueError(f"normalization.method debe ser uno de {NORMALIZATION_METHODS}.")
    if cfg.normalization_target <= 0 or cfg.silence_rms_threshold <= 0:
        raise ValueError("normalization.target_level y silence_rms_threshold deben ser > 0.")
    return cfg


def _parse_features(section: dict[str, Any], target_sr: int) -> FeatureConfig:
    where = "features"
    stft = _require(section, "stft", where)
    mel = _require(section, "mel", where)
    mfcc = _require(section, "mfcc", where)
    wavelet = _require(section, "wavelet", where)
    rms_section = _require(section, "rms", where)
    cfg = FeatureConfig(
        representations=tuple(str(r) for r in _require(section, "representations", where)),
        n_jobs=int(_require(section, "n_jobs", where)),
        stft_window_sec=float(_require(stft, "window_sec", f"{where}.stft")),
        stft_hop_sec=float(_require(stft, "hop_sec", f"{where}.stft")),
        stft_window=str(_require(stft, "window", f"{where}.stft")),
        n_mels=int(_require(mel, "n_mels", f"{where}.mel")),
        mel_fmin_hz=float(_require(mel, "fmin_hz", f"{where}.mel")),
        mel_fmax_hz=float(_require(mel, "fmax_hz", f"{where}.mel")),
        mel_htk=bool(_require(mel, "htk", f"{where}.mel")),
        mel_norm=str(_require(mel, "norm", f"{where}.mel")),
        mel_power=float(_require(mel, "power", f"{where}.mel")),
        mel_amin=float(_require(mel, "amin", f"{where}.mel")),
        n_mfcc=int(_require(mfcc, "n_mfcc", f"{where}.mfcc")),
        wavelet_name=str(_require(wavelet, "name", f"{where}.wavelet")),
        wavelet_level=int(_require(wavelet, "level", f"{where}.wavelet")),
        wavelet_mode=str(_require(wavelet, "mode", f"{where}.wavelet")),
        rms_frame_sec=float(_require(rms_section, "frame_sec", f"{where}.rms")),
        rms_hop_sec=float(_require(rms_section, "hop_sec", f"{where}.rms")),
        constant_std_threshold=float(_require(section, "constant_std_threshold", where)),
    )
    unknown = set(cfg.representations) - set(KNOWN_REPRESENTATIONS)
    if unknown or not cfg.representations:
        raise ValueError(f"features.representations contiene valores no validos: {sorted(unknown)}")
    for seconds in (cfg.stft_window_sec, cfg.stft_hop_sec, cfg.rms_frame_sec, cfg.rms_hop_sec):
        seconds_to_samples(seconds, target_sr)
    if not 0 <= cfg.mel_fmin_hz < cfg.mel_fmax_hz <= target_sr / 2:
        raise ValueError("Se requiere 0 <= mel.fmin_hz < mel.fmax_hz <= Nyquist.")
    if not 1 <= cfg.n_mfcc <= cfg.n_mels:
        raise ValueError("Se requiere 1 <= mfcc.n_mfcc <= mel.n_mels.")
    if cfg.wavelet_level < 1:
        raise ValueError("features.wavelet.level debe ser >= 1.")
    return cfg


KNOWN_CLASSIFIERS: tuple[str, ...] = ("random_forest", "svm_rbf", "knn")


def _parse_training(section: dict[str, Any], models_section: dict[str, Any]) -> tuple[TrainingConfig,
                                                                                     dict[str, ModelSpec]]:
    cv = _require(section, "cv", "training")
    training = TrainingConfig(
        scoring=str(_require(section, "scoring", "training")),
        preferred_n_splits=int(_require(cv, "preferred_n_splits", "training.cv")),
        fallback_n_splits=int(_require(cv, "fallback_n_splits", "training.cv")),
        shuffle=bool(_require(cv, "shuffle", "training.cv")),
        min_patients_per_class_per_fold=int(_require(cv, "min_patients_per_class_per_fold", "training.cv")),
        min_cycles_per_class_per_fold=int(_require(cv, "min_cycles_per_class_per_fold", "training.cv")),
        n_jobs=int(_require(section, "n_jobs", "training")),
    )
    if not 2 <= training.fallback_n_splits <= training.preferred_n_splits:
        raise ValueError("Se requiere 2 <= training.cv.fallback_n_splits <= preferred_n_splits.")
    if training.scoring != "f1_weighted":
        raise ValueError("La metrica principal de la metodologia es f1_weighted.")
    models: dict[str, ModelSpec] = {}
    for name in KNOWN_CLASSIFIERS:
        spec = _require(models_section, name, "models")
        grid = {str(k): list(v) for k, v in (_require(spec, "grid", f"models.{name}") or {}).items()}
        if not grid or any(len(v) == 0 for v in grid.values()):
            raise ValueError(f"models.{name}.grid no puede estar vacio.")
        calibration = spec.get("calibration")
        if calibration not in (None, "sigmoid", "isotonic"):
            raise ValueError(f"models.{name}.calibration debe ser sigmoid, isotonic o vacio.")
        models[name] = ModelSpec(fixed=dict(spec.get("fixed") or {}), grid=grid,
                                 final_params=dict(spec.get("final_params") or {}), calibration=calibration)
    unknown = set(models_section) - set(KNOWN_CLASSIFIERS)
    if unknown:
        raise ValueError(f"Clasificadores no reconocidos en models: {sorted(unknown)}")
    return training, models


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Carga ``config.yaml`` y devuelve un objeto :class:`Config` validado.

    Prioridad del archivo: argumento ``path`` > variable de entorno
    ``RESPIA_CONFIG`` > ``config.yaml`` en la raiz del proyecto.
    """
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    base = config_path.parent
    project = _require(raw, "project", "")
    paths_raw = _require(raw, "paths", "")
    data_dir = _resolve(base, _require(paths_raw, "data_dir", "paths"))
    paths = Paths(
        root=base,
        data_dir=data_dir,
        output_dir=_resolve(base, _require(paths_raw, "output_dir", "paths")),
        results_dir=_resolve(base, _require(paths_raw, "results_dir", "paths")),
        model_dir=_resolve(base, _require(paths_raw, "model_dir", "paths")),
        raw_icbhi_dir=_resolve(data_dir, _require(paths_raw, "icbhi_subdir", "paths")),
        raw_fraiwan_dir=_resolve(data_dir, _require(paths_raw, "fraiwan_subdir", "paths")),
    )

    icbhi_raw = _require(raw, "icbhi", "")
    icbhi = IcbhiConfig(
        audio_extension=str(_require(icbhi_raw, "audio_extension", "icbhi")).lower(),
        annotation_extension=str(_require(icbhi_raw, "annotation_extension", "icbhi")).lower(),
        filename_corrections_file=str(_require(icbhi_raw, "filename_corrections_file", "icbhi")),
        known_chest_locations=tuple(_require(icbhi_raw, "known_chest_locations", "icbhi")),
        known_acquisition_modes=tuple(_require(icbhi_raw, "known_acquisition_modes", "icbhi")),
        known_equipment=tuple(_require(icbhi_raw, "known_equipment", "icbhi")),
        time_tolerance_sec=float(_require(icbhi_raw, "time_tolerance_sec", "icbhi")),
    )

    seg_raw = _require(raw, "segmentation", "")
    iqr_k = {str(name): float(k) for name, k in _require(seg_raw, "duration_outlier_iqr_k", "segmentation").items()}
    if not iqr_k or any(k <= 0 for k in iqr_k.values()):
        raise ValueError("segmentation.duration_outlier_iqr_k debe contener factores k > 0.")
    segmentation = SegmentationConfig(
        duration_outlier_iqr_k=iqr_k,
        flatline_min_sec=float(_require(seg_raw, "flatline_min_sec", "segmentation")),
        consistency_check_cycles=int(_require(seg_raw, "consistency_check_cycles", "segmentation")),
    )

    preprocessing = _parse_preprocessing(_require(raw, "preprocessing", ""))
    features = _parse_features(_require(raw, "features", ""), preprocessing.target_sr)
    split_raw = _require(raw, "split", "")
    split = SplitConfig(
        test_size=float(_require(split_raw, "test_size", "split")),
        n_candidates=int(_require(split_raw, "n_candidates", "split")),
        balance_by=tuple(str(a) for a in _require(split_raw, "balance_by", "split")),
    )
    if not 0 < split.test_size < 1 or split.n_candidates < 1:
        raise ValueError("split.test_size debe estar en (0, 1) y split.n_candidates >= 1.")
    if "label" not in split.balance_by or not set(split.balance_by) <= {"label", "equipment"}:
        raise ValueError("split.balance_by debe incluir 'label' y solo admite 'label' y 'equipment'.")
    training, models = _parse_training(_require(raw, "training", ""), _require(raw, "models", ""))
    best_model_selection = str(_require(_require(raw, "evaluation", ""), "best_model_selection", "evaluation"))
    if best_model_selection not in ("cv", "test"):
        raise ValueError("evaluation.best_model_selection debe ser 'cv' o 'test'.")
    evaluation_raw = raw["evaluation"]
    designated_main = {str(k): str(v) for k, v in _require(evaluation_raw, "designated_main", "evaluation").items()}
    designated_prototype = {str(k): str(v)
                            for k, v in _require(evaluation_raw, "designated_prototype", "evaluation").items()}
    for name, spec in (("designated_main", designated_main), ("designated_prototype", designated_prototype)):
        if not {"representation", "classifier"} <= set(spec):
            raise ValueError(f"evaluation.{name} requiere 'representation' y 'classifier'.")
    improvements: dict[str, ImprovementVariant] = {}
    for name, spec in ((raw.get("improvements") or {}).get("variants") or {}).items():
        variant = ImprovementVariant(
            name=str(name),
            description=str(spec.get("description", "")),
            representations=tuple(str(r) for r in _require(spec, "representations", f"improvements.{name}")),
            classifiers=tuple(str(c) for c in _require(spec, "classifiers", f"improvements.{name}")),
            fixed_overrides={str(k): dict(v or {}) for k, v in (spec.get("fixed_overrides") or {}).items()},
        )
        bad_reps = set(variant.representations) - set(KNOWN_REPRESENTATIONS) - set(COMPOSITE_REPRESENTATIONS)
        bad_clfs = (set(variant.classifiers) | set(variant.fixed_overrides)) - set(KNOWN_CLASSIFIERS)
        if bad_reps or bad_clfs:
            raise ValueError(f"improvements.{name}: valores no validos {sorted(bad_reps | bad_clfs)}")
        improvements[variant.name] = variant
    d_raw = _require(raw, "descriptors", "")
    descriptors = DescriptorConfig(
        tonal_median_filter_bins=int(_require(d_raw, "tonal_median_filter_bins", "descriptors")),
        tonal_prominence_db=float(_require(d_raw, "tonal_prominence_db", "descriptors")),
        tonal_freq_tolerance_bins=int(_require(d_raw, "tonal_freq_tolerance_bins", "descriptors")),
        band_edges_hz=tuple(float(v) for v in _require(d_raw, "band_edges_hz", "descriptors")),
        rolloff_fraction=float(_require(d_raw, "rolloff_fraction", "descriptors")),
        envelope_frame_sec=float(_require(d_raw, "envelope_frame_sec", "descriptors")),
        envelope_hop_sec=float(_require(d_raw, "envelope_hop_sec", "descriptors")),
        peak_mad_factor=float(_require(d_raw, "peak_mad_factor", "descriptors")),
        peak_min_distance_sec=float(_require(d_raw, "peak_min_distance_sec", "descriptors")),
        zcr_frame_sec=float(_require(d_raw, "zcr_frame_sec", "descriptors")),
        zcr_hop_sec=float(_require(d_raw, "zcr_hop_sec", "descriptors")),
    )
    edges = descriptors.band_edges_hz
    if len(edges) < 3 or list(edges) != sorted(edges) or edges[-1] > preprocessing.target_sr / 2:
        raise ValueError("descriptors.band_edges_hz debe ser creciente, con >= 3 valores y <= Nyquist.")
    if descriptors.tonal_median_filter_bins % 2 == 0 or not 0 < descriptors.rolloff_fraction < 1:
        raise ValueError("descriptors: tonal_median_filter_bins debe ser impar y rolloff_fraction en (0, 1).")
    for seconds in (descriptors.envelope_frame_sec, descriptors.envelope_hop_sec, descriptors.zcr_frame_sec,
                    descriptors.zcr_hop_sec):
        seconds_to_samples(seconds, preprocessing.target_sr)

    r_raw = _require(raw, "robustness", "")
    robustness = RobustnessConfig(
        noise=str(_require(r_raw, "noise", "robustness")),
        snr_db=tuple(float(v) for v in _require(r_raw, "snr_db", "robustness")),
        references=tuple(str(v) for v in _require(r_raw, "references", "robustness")),
        repetitions=int(_require(r_raw, "repetitions", "robustness")),
        n_jobs=int(_require(r_raw, "n_jobs", "robustness")),
    )
    if robustness.noise != "white_gaussian" or not robustness.snr_db or robustness.repetitions < 1:
        raise ValueError("robustness: noise debe ser 'white_gaussian', con al menos un SNR y repetitions >= 1.")
    if not robustness.references or not set(robustness.references) <= {"fullband", "inband"}:
        raise ValueError("robustness.references solo admite 'fullband' e 'inband'.")

    e_raw = _require(raw, "external_validation", "")
    external = ExternalValidationConfig(
        window_sec=float(_require(e_raw, "window_sec", "external_validation")),
        aggregation=str(_require(e_raw, "aggregation", "external_validation")),
        primary_filter=str(_require(e_raw, "primary_filter", "external_validation")),
        label_map={str(k): str(v) for k, v in _require(e_raw, "label_map", "external_validation").items()},
        excluded={str(k): str(v) for k, v in _require(e_raw, "excluded", "external_validation").items()},
        n_jobs=int(_require(e_raw, "n_jobs", "external_validation")),
    )
    if external.window_sec <= 0 or external.aggregation != "majority_vote" or external.primary_filter not in "BDE":
        raise ValueError("external_validation: window_sec > 0, aggregation 'majority_vote' y primary_filter B/D/E.")
    if set(external.label_map) & set(external.excluded):
        raise ValueError("external_validation: una etiqueta no puede estar en label_map y en excluded a la vez.")
    labels = _parse_labels(_require(raw, "labels", ""))
    if not set(external.label_map.values()) <= set(labels.modes[labels.label_mode]):
        raise ValueError("external_validation.label_map solo puede apuntar a clases del modo de etiquetas activo.")

    s_raw = _require(raw, "statistics", "")
    statistics = StatisticsConfig(n_bootstrap=int(_require(s_raw, "n_bootstrap", "statistics")),
                                  confidence=float(_require(s_raw, "confidence", "statistics")))
    if statistics.n_bootstrap < 1000 or not 0.5 < statistics.confidence < 1:
        raise ValueError("statistics: n_bootstrap >= 1000 y confidence en (0,5; 1).")

    a_raw = _require(raw, "app", "")
    app = AppConfig(
        default_model=str(_require(a_raw, "default_model", "app")),
        min_duration_sec=float(_require(a_raw, "min_duration_sec", "app")),
        typical_max_duration_sec=float(_require(a_raw, "typical_max_duration_sec", "app")),
        max_duration_sec=float(_require(a_raw, "max_duration_sec", "app")),
        clipping_warning_fraction=float(_require(a_raw, "clipping_warning_fraction", "app")),
        examples_per_class=int(_require(a_raw, "examples_per_class", "app")),
    )
    if app.default_model not in ("prototype", "best_model"):
        raise ValueError("app.default_model debe ser 'prototype' o 'best_model'.")
    if not 0 < app.min_duration_sec < app.typical_max_duration_sec <= app.max_duration_sec:
        raise ValueError("app: se requiere 0 < min_duration_sec < typical_max_duration_sec <= max_duration_sec.")

    audit_raw = _require(raw, "audit", "")
    audit = AuditConfig(
        signal_check=bool(_require(audit_raw, "signal_check", "audit")),
        clipping_level=float(_require(audit_raw, "clipping_level", "audit")),
        clipping_warning_fraction=float(_require(audit_raw, "clipping_warning_fraction", "audit")),
        clipping_report_fractions=tuple(float(f) for f in _require(audit_raw, "clipping_report_fractions", "audit")),
        silence_peak_threshold=float(_require(audit_raw, "silence_peak_threshold", "audit")),
    )

    return Config(
        project_name=str(_require(project, "name", "project")),
        experiment_version=str(_require(project, "experiment_version", "project")),
        random_state=int(_require(project, "random_state", "project")),
        paths=paths,
        icbhi=icbhi,
        labels=labels,
        segmentation=segmentation,
        preprocessing=preprocessing,
        features=features,
        split=split,
        training=training,
        models=models,
        best_model_selection=best_model_selection,
        designated_main=designated_main,
        designated_prototype=designated_prototype,
        improvements=improvements,
        descriptors=descriptors,
        robustness=robustness,
        external=external,
        statistics=statistics,
        app=app,
        audit=audit,
        source_file=config_path,
        raw=raw,
    )
