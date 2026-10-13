"""Voto por mayoria de los 3 clasificadores (MEJORA PROPUESTA, fuera del experimento principal).

Cada miembro es un pipeline completo ``StandardScaler -> clasificador`` con los
hiperparametros que ya eligio la validacion cruzada. La clase final es la mas votada;
si los tres miembros discrepan, decide el primero de la lista (el de mayor F1 de CV).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def majority_vote(votes: np.ndarray, n_classes: int) -> np.ndarray:
    """Clase mas votada por fila; en empate gana el voto del primer miembro si esta entre las empatadas.

    Args:
        votes: enteros de forma ``(n_muestras, n_miembros)``; la columna 0 es el miembro de desempate.
        n_classes: numero de clases (codigos 0..n_classes-1).
    """
    votes = np.asarray(votes, dtype=int)
    counts = np.stack([(votes == c).sum(axis=1) for c in range(n_classes)], axis=1)
    winners = counts == counts.max(axis=1, keepdims=True)
    first = votes[:, 0]
    first_is_winner = winners[np.arange(len(votes)), first]
    return np.where(first_is_winner, first, winners.argmax(axis=1))


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    """Jurado de pipelines: ``predict`` = voto por mayoria; ``predict_proba`` = media de probabilidades."""

    def __init__(self, members: list[tuple[str, Any]] | None = None) -> None:
        self.members = members

    def fit(self, X: Any, y: Any) -> MajorityVoteClassifier:  # noqa: N803
        """Ajusta una copia de cada miembro con los mismos datos."""
        self.classes_ = np.unique(y)
        self.fitted_ = [(name, clone(model).fit(X, y)) for name, model in (self.members or [])]
        if not self.fitted_:
            raise ValueError("El jurado necesita al menos un miembro.")
        return self

    def member_predictions(self, X: Any) -> np.ndarray:  # noqa: N803
        """Votos de cada miembro, forma ``(n_muestras, n_miembros)``."""
        return np.column_stack([model.predict(X) for _, model in self.fitted_])

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803
        """Clase mas votada (empate: decide el primer miembro)."""
        return majority_vote(self.member_predictions(X), len(self.classes_))

    def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
        """Media de las probabilidades de los miembros (para ROC/AUC e interfaz)."""
        return np.mean([model.predict_proba(X) for _, model in self.fitted_], axis=0)
