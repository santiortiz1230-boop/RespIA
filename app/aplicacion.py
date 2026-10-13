"""FASE 12 - Interfaz de RespIA.

Desde RespIA/:

    streamlit run app/aplicacion.py

Clasifica el patron acustico de un ciclo respiratorio por archivo WAV, con los modelos
congelados de la Fase 7 y exactamente el preprocesamiento de entrenamiento. No hay
segmentacion automatica: el usuario entrega (o recorta a mano) el tramo de un ciclo.

La pantalla es un solo componente (``instrumento.js`` e ``instrumento.css``); este archivo
le entrega los datos de cada estado y recibe lo que el usuario hace en el.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for folder in (str(ROOT), str(APP_DIR)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from estado import analyze, apply_events, audio_view, build_payload, new_session  # noqa: E402

from src.configuracion import load_config  # noqa: E402
from src.inferencia import InferenceError, analyze_cycle, load_models, predict  # noqa: E402

KEY = "instrumento"
EXAMPLES_DIR = APP_DIR / "examples"
EVENTS = ("file", "example", "model", "selection", "analyze")
FONTS = (
    ("IBM Plex Sans", 400, "ibm-plex-sans-latin-400-normal.woff2"),
    ("IBM Plex Sans", 500, "ibm-plex-sans-latin-500-normal.woff2"),
    ("IBM Plex Sans", 600, "ibm-plex-sans-latin-600-normal.woff2"),
    ("IBM Plex Mono", 400, "ibm-plex-mono-latin-400-normal.woff2"),
    ("IBM Plex Mono", 500, "ibm-plex-mono-latin-500-normal.woff2"),
)


@st.cache_resource(show_spinner=False)
def page_css() -> str:
    """Estilo de la pagina con la tipografia incrustada, para que no dependa de internet."""
    faces = []
    for family, weight, name in FONTS:
        path = APP_DIR / "fuentes" / name
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            faces.append(f"@font-face {{ font-family: '{family}'; font-weight: {weight}; font-style: normal; "
                         f"font-display: block; src: url(data:font/woff2;base64,{data}) format('woff2'); }}")
    return "\n".join(faces) + "\n" + (APP_DIR / "tema.css").read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def instrument():
    return st.components.v2.component(
        "respia_instrumento",
        css=(APP_DIR / "instrumento.css").read_text(encoding="utf-8"),
        js=(APP_DIR / "instrumento.js").read_text(encoding="utf-8"),
    )


@st.cache_resource(show_spinner=False)
def resources():
    cfg = load_config()
    models = load_models(cfg)
    # La primera llamada a las librerias de audio y a los modelos es lenta; con esta pasada
    # en vacio el primer analisis del usuario ya responde rapido.
    warmup = analyze_cycle(np.random.default_rng(0).standard_normal(2 * cfg.preprocessing.target_sr) * 0.01,
                           cfg.preprocessing.target_sr, cfg)
    for bundle in models.values():
        predict(bundle, warmup)
    examples_path = EXAMPLES_DIR / "ejemplos.csv"
    examples = pd.read_csv(examples_path) if examples_path.is_file() else pd.DataFrame()
    return cfg, models, examples


@st.cache_data(show_spinner=False, max_entries=8)
def view(data: bytes) -> dict:
    return audio_view(data, resources()[0])


@st.cache_data(show_spinner=False, max_entries=32)
def run(data: bytes, start: float, end: float):
    cfg, models, _ = resources()
    return analyze(data, start, end, cfg, models)


def main() -> None:
    st.set_page_config(page_title="RespIA", page_icon=str(APP_DIR / "icono.png"), layout="wide")
    st.html(f"<style>{page_css()}</style>")
    try:
        cfg, models, examples = resources()
    except (InferenceError, FileNotFoundError) as error:
        st.error(f"La aplicación no puede iniciar. {error}")
        return
    default_model = cfg.app.default_model if cfg.app.default_model in models else next(iter(models))
    session = st.session_state.setdefault("respia", new_session(default_model))

    apply_events(session, st.session_state.get(KEY), examples, EXAMPLES_DIR, set(models))
    events = instrument()(key=KEY, data=build_payload(session, cfg, models, examples, view, run),
                          default={"model": session["model"], "selection": None},
                          **{f"on_{name}_change": lambda: None for name in EVENTS})
    if apply_events(session, events, examples, EXAMPLES_DIR, set(models)):
        st.rerun()


main()
