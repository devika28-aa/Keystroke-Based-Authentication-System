import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class IsolationForestClassifier:
    def __init__(self, n_estimators: int = 150, contamination: float = 0.05, random_state: int = 42):
        self.n_estimators   = n_estimators
        self.contamination  = contamination
        self.random_state   = random_state
        self.models_: dict  = {}        
        self.classes_: list = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[IsolationForest] Training {len(self.classes_)} one-class models …")
        for subj in self.classes_:
            mask  = y_train == subj
            X_sub = X_train[mask]
            clf   = IsolationForest(
                n_estimators  = self.n_estimators,
                contamination = self.contamination,
                random_state  = self.random_state,
                n_jobs        = -1,
            )
            clf.fit(X_sub)
            self.models_[subj] = clf
        print(f"[IsolationForest] Training complete.")

    def _score_matrix(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            self.models_[s].decision_function(X) for s in self.classes_
        ])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._score_matrix(X)
        idx    = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._score_matrix(X)                    
        scores -= scores.min(axis=1, keepdims=True)
        row_sum = scores.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1                         
        return scores / row_sum

    def save(self, path: str = "models/isolation_forest.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models_, path)

    @staticmethod
    def load(path: str = "models/isolation_forest.pkl") -> "IsolationForestClassifier":
        clf = IsolationForestClassifier()
        clf.models_ = joblib.load(path)
        clf.classes_ = list(clf.models_.keys())
        return clf

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.predict(X_test)
        acc   = np.mean(preds == y_test)
        print(f"[IsolationForest] Accuracy: {acc * 100:.2f}%")
        return float(acc)
