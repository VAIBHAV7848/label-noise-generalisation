"""Decision Tree and Random Forest wrappers for tabular and baseline evaluation."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Tuple


class DecisionTreeModel:
    """Decision Tree Classifier wrapper."""

    def __init__(self, max_depth: Optional[int] = 15, criterion: str = "gini", random_state: int = 42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)


class RandomForestModel:
    """Random Forest Classifier wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 15, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)
