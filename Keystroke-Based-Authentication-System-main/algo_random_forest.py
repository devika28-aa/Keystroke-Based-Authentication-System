import numpy as np
from sklearn.ensemble import RandomForestClassifier as _RFC
import joblib
import os

class RandomForestClassifier:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_       = None
        self.classes_     = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[RandomForest] Training {self.n_estimators} trees on {X_train.shape[0]} samples, {len(self.classes_)} classes …")
        self.model_ = _RFC(
            n_estimators = self.n_estimators,
            random_state = self.random_state,
            n_jobs       = -1,
            class_weight = "balanced"
        )
        self.model_.fit(X_train, y_train)
        print(f"[RandomForest] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def save(self, path: str = "models/randomforest.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model_, path)

    @staticmethod
    def load(path: str = "models/randomforest.pkl") -> "RandomForestClassifier":
        clf = RandomForestClassifier()
        clf.model_ = joblib.load(path)
        clf.classes_ = clf.model_.classes_
        return clf
