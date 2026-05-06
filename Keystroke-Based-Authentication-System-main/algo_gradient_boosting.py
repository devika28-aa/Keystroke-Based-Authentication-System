import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib, os

class GradientBoostingClassifierWrapper:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_ = None
        self.classes_ : list = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=42
        )
        self.model_.fit(X_train, y_train)
        print("  [GradientBoosting] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def save(self, path: str = "models/gradient_boosting.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "models/gradient_boosting.pkl") -> "GradientBoostingClassifierWrapper":
        return joblib.load(path)
