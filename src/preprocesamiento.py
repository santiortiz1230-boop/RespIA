"""Cadena unica de preprocesamiento de audio (Fase 3).

La misma funcion, :func:`preprocess_audio`, se usa para entrenamiento, prueba,
validacion externa e interfaz; no existe otra ruta de procesamiento::

    audio -> mono -> remuestreo a TARGET_SR -> [ruido, solo Fase 9]
          -> pasa-banda Butterworth (SOS) -> normalizacion -> (caracteristicas, Fase 4)

Decisiones (justificadas en ``config.yaml`` y ``ESTADO_DEL_PROYECTO.md``):

* **Remuestreo** polifasico racional (``scipy.signal.resample_poly``, Kaiser).
* **Pasa-banda** Butterworth en secciones de segundo orden (SOS, numericamente
  estable con cortes bajos como 100 Hz a 8 kHz) aplicado en **fase cero**
  (``sosfiltfilt``) para no desplazar ni deformar transitorios como los crepitantes.
* **Normalizacion por RMS del ciclo filtrado** (despues del filtro, para que el
  nivel refleje solo la energia en banda). Elimina la ganancia del equipo, que en
  ICBHI no es fisiologica (grabaciones normalizadas al pico en origen, ganancias
  distintas por equipo, saturacion), y conserva la dinamica temporal interna del
  ciclo: tras normalizar, el RMS por frame describe la intensidad de cada tramo
  relativa al ciclo (p. ej. picos explosivos). El RMS en banda previo a normalizar
  se devuelve en :class:`PreprocessingResult` para trazabilidad.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

from src.configuracion import PreprocessingConfig

BeforeFilterHook = Callable[[np.ndarray, int], np.ndarray]


class PreprocessingError(ValueError):
    """Entrada que no puede preprocesarse (vacia, no finita o demasiado corta)."""


@dataclass(frozen=True)
class PreprocessingResult:
    """Senal preprocesada y trazabilidad de cada etapa."""

    audio: np.ndarray            # float64, mono, a ``sample_rate``, filtrada y normalizada
    sample_rate: int             # TARGET_SR
    input_sample_rate: int
    input_n_samples: int
    inband_rms: float            # RMS tras el pasa-banda y antes de normalizar (escala completa = 1)
    gain: float                  # factor aplicado por la normalizacion (1.0 si 'none' o silencio)
    is_silent: bool              # RMS en banda < umbral: no se normalizo

    @property
    def duration_sec(self) -> float:
        return self.audio.shape[0] / self.sample_rate


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convierte a mono promediando canales (formato soundfile: muestras x canales)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise PreprocessingError(f"Forma de audio no soportada: {x.shape}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int, kaiser_beta: float) -> np.ndarray:
    """Remuestreo polifasico racional ``orig_sr -> target_sr`` (factor exacto up/down).

    La salida tiene ``ceil(n * up / down)`` muestras y el filtro FIR anti-aliasing
    (ventana Kaiser) tiene retardo compensado: la senal no se desplaza en el tiempo.
    """
    x = np.asarray(audio, dtype=np.float64)
    if orig_sr == target_sr:
        return x.copy()
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(x, up=int(target_sr) // g, down=int(orig_sr) // g, window=("kaiser", float(kaiser_beta)))


@lru_cache(maxsize=16)
def design_bandpass(low_hz: float, high_hz: float, order: int, sample_rate: int) -> np.ndarray:
    """Disena el pasa-banda Butterworth en formato SOS (coeficientes de solo lectura)."""
    if not 0 < low_hz < high_hz < sample_rate / 2:
        raise PreprocessingError(f"Cortes invalidos [{low_hz}, {high_hz}] Hz para fs = {sample_rate} Hz")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    sos.setflags(write=False)
    return sos


def sos_padlen(sos: np.ndarray) -> int:
    """Longitud de extension por defecto de ``sosfiltfilt`` (la senal debe superarla)."""
    n_zeros = min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * (2 * len(sos) + 1 - n_zeros)


def apply_bandpass(audio: np.ndarray, sos: np.ndarray, zero_phase: bool) -> np.ndarray:
    """Filtra con el pasa-banda; en fase cero si ``zero_phase``.

    Raises:
        PreprocessingError: si la senal es mas corta que la extension que requiere el
            filtrado en fase cero (no se rellena en silencio).
    """
    sos = np.array(sos, dtype=np.float64)  # copia escribible: scipy no acepta buffers de solo lectura
    if zero_phase:
        padlen = sos_padlen(sos)
        if audio.shape[0] <= padlen:
            raise PreprocessingError(
                f"Senal demasiado corta para filtrar en fase cero: {audio.shape[0]} <= {padlen} muestras"
            )
        return sosfiltfilt(sos, audio)
    return sosfilt(sos, audio)


def rms(audio: np.ndarray) -> float:
    """Valor cuadratico medio de la senal completa."""
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def normalize_amplitude(
    audio: np.ndarray, method: str, target_level: float, silence_rms_threshold: float
) -> tuple[np.ndarray, float, bool]:
    """Normaliza la amplitud del ciclo sin dividir nunca por cero.

    * ``rms``: ``x * target / RMS(x)``  -> RMS de salida = ``target``.
    * ``peak``: ``x * target / max|x|`` -> pico de salida = ``target``.
    * ``none``: sin cambios.

    Si el RMS de la senal es menor que ``silence_rms_threshold`` se considera
    silencio: se devuelve sin escalar (ganancia 1) y ``is_silent=True``.

    Returns:
        ``(audio_normalizado, ganancia, is_silent)``.
    """
    if method not in ("rms", "peak", "none"):
        raise PreprocessingError(f"Metodo de normalizacion desconocido: {method}")
    level_rms = rms(audio)
    if level_rms < silence_rms_threshold:
        return audio.copy(), 1.0, True
    if method == "none":
        return audio.copy(), 1.0, False
    reference = level_rms if method == "rms" else float(np.max(np.abs(audio)))
    gain = target_level / reference
    return audio * gain, gain, False


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    cfg: PreprocessingConfig,
    before_filter: BeforeFilterHook | None = None,
) -> PreprocessingResult:
    """Aplica la cadena completa de preprocesamiento a un ciclo respiratorio.

    ``audio`` puede venir en mono (1-D) o como muestras x canales (2-D), en cualquier escala.
    ``before_filter`` es una transformacion opcional que se aplica despues del remuestreo y
    antes del filtro y recibe ``(senal, TARGET_SR)``. Solo se usa en la Fase 9 para sumar
    ruido; en el resto de fases debe ser ``None``.

    Raises:
        PreprocessingError: entrada vacia, con NaN/Inf o demasiado corta.
    """
    x = to_mono(audio)
    if x.size == 0:
        raise PreprocessingError("Audio vacio.")
    if not np.all(np.isfinite(x)):
        raise PreprocessingError("La entrada contiene NaN o Inf.")
    input_n = int(x.shape[0])

    x = resample_audio(x, int(sample_rate), cfg.target_sr, cfg.resample_kaiser_beta)
    if before_filter is not None:
        x = np.asarray(before_filter(x, cfg.target_sr), dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise PreprocessingError("La transformacion previa al filtro produjo NaN o Inf.")

    sos = design_bandpass(cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order, cfg.target_sr)
    x = apply_bandpass(x, sos, cfg.zero_phase)
    inband = rms(x)
    x, gain, is_silent = normalize_amplitude(
        x, cfg.normalization_method, cfg.normalization_target, cfg.silence_rms_threshold
    )
    if not np.all(np.isfinite(x)):
        raise PreprocessingError("El preprocesamiento produjo NaN o Inf.")
    return PreprocessingResult(
        audio=x,
        sample_rate=cfg.target_sr,
        input_sample_rate=int(sample_rate),
        input_n_samples=input_n,
        inband_rms=inband,
        gain=float(gain),
        is_silent=is_silent,
    )


def expected_output_length(n_samples: int, orig_sr: int, target_sr: int) -> int:
    """Longitud exacta que produce el remuestreo polifasico: ``ceil(n * up / down)``."""
    if orig_sr == target_sr:
        return int(n_samples)
    g = gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // g, int(orig_sr) // g
    return -(-int(n_samples) * up // down)
