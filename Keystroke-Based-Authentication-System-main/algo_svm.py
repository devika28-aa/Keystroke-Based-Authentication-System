import numpy as np
from sklearn.svm import SVC
import joblib
import os

class SVMClassifier:
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, random_state: int = 42):
        self.kernel       = kernel
        self.C            = C
        self.random_state = random_state
        self.model_       = None
        self.classes_     = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[SVM] Training on {X_train.shape[0]} samples, {len(self.classes_)} classes  (kernel={self.kernel}) …")
        self.model_ = SVC(
            kernel       = self.kernel,
            C            = self.C,
            probability  = True,
            random_state = self.random_state,
        )
        self.model_.fit(X_train, y_train)
        print(f"[SVM] Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def save(self, path: str = "models/svm.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model_, path)

    @staticmethod
    def load(path: str = "models/svm.pkl") -> "SVMClassifier":
        clf = SVMClassifier()
        clf.model_ = joblib.load(path)
        clf.classes_ = clf.model_.classes_
        return clf
