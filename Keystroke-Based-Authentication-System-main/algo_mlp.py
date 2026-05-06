import numpy as np
from sklearn.neural_network import MLPClassifier as _MLPC
import joblib
import os

class MLPClassifierWrapper:
    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter           = max_iter
        self.random_state       = random_state
        self.model_             = None
        self.classes_           = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[MLP] Training Neural Network (layers={self.hidden_layer_sizes}) on {X_train.shape[0]} samples, {len(self.classes_)} classes …")
        self.model_ = _MLPC(
            hidden_layer_sizes = self.hidden_layer_sizes,
            max_iter           = self.max_iter,
            random_state       = self.random_state,
            early_stopping     = True,
        )
        self.model_.fit(X_train, y_train)
        print(f"[MLP] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def save(self, path: str = "models/mlp.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model_, path)

    @staticmethod
    def load(path: str = "models/mlp.pkl") -> "MLPClassifierWrapper":
        clf = MLPClassifierWrapper()
        clf.model_ = joblib.load(path)
        clf.classes_ = clf.model_.classes_
        return clf
