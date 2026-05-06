import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC
import joblib
import os

class HistGBClassifierWrapper:
    def __init__(self, max_iter: int = 150, learning_rate: float = 0.1, random_state: int = 42):
        self.max_iter      = max_iter
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model_        = None
        self.classes_      = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[HistGB] Training Gradient Boosted Trees on {X_train.shape[0]} samples, {len(self.classes_)} classes …")
        self.model_ = _HGBC(
            max_iter      = self.max_iter,
            learning_rate = self.learning_rate,
            random_state  = self.random_state,
        )
        self.model_.fit(X_train, y_train)
        print(f"[HistGB] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def save(self, path: str = "models/histgb.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model_, path)

    @staticmethod
    def load(path: str = "models/histgb.pkl") -> "HistGBClassifierWrapper":
        clf = HistGBClassifierWrapper()
        clf.model_ = joblib.load(path)
        clf.classes_ = clf.model_.classes_
        return clf
