"""Utilidades transversales: logging, semillas, rutas portables y escritura de archivos."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_console() -> None:
    """Fuerza UTF-8 en stdout/stderr (evita errores de codificacion en consolas Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger con formato homogeneo para todo el proyecto."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Fija semillas de ``random``, ``numpy`` y ``PYTHONHASHSEED``.

    scikit-learn se controla pasando ``random_state`` explicitamente a cada estimador
    y particionador (no depende de un estado global).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_posix_relative(path: Path, base: Path) -> str:
    """Ruta relativa a ``base`` en formato POSIX (portable entre Windows y Linux)."""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Guarda un DataFrame como CSV UTF-8 sin indice, creando la carpeta si falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def environment_info(libraries: tuple[str, ...]) -> dict[str, Any]:
    """Version de Python, sistema operativo y librerias (para registros de ejecucion)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    def safe_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "no instalado"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {name: safe_version(name) for name in libraries},
    }


def save_json(obj: Any, path: Path) -> Path:
    """Guarda ``obj`` como JSON legible (UTF-8), creando la carpeta si falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)
    return path
