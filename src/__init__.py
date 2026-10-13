"""RespIA: clasificacion de sonidos respiratorios adventicios con Machine Learning clasico."""

import importlib
import importlib.abc
import importlib.util
import sys

__version__ = "0.1.0"

# Los modelos guardados con joblib recuerdan el nombre que tenia el modulo de sus clases cuando se
# entrenaron (``src.models`` y ``src.ensemble``). Los modulos ahora se llaman ``src.modelos`` y
# ``src.ensamble``; este alias permite cargar esos mismos archivos sin volver a guardarlos, de modo que
# su huella SHA-256 (la del sello previo a la prueba) no cambia.
_ALIAS_DE_MODULOS = {"src.models": "src.modelos", "src.ensemble": "src.ensamble"}


class _BuscadorDeAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resuelve los nombres antiguos de modulo devolviendo el modulo actual."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001, ARG002
        if fullname in _ALIAS_DE_MODULOS:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):  # noqa: ANN001
        return importlib.import_module(_ALIAS_DE_MODULOS[spec.name])

    def exec_module(self, module) -> None:  # noqa: ANN001
        """El modulo real ya esta inicializado: no hay nada que ejecutar."""


if not any(isinstance(finder, _BuscadorDeAlias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _BuscadorDeAlias())
