import numpy as np
from sklearn.neighbors import KNeighborsClassifier as _KNN
import joblib, os
class KNNClassifier:
    def __init__(self, n_neighbors: int = 7, metric: str = "minkowski",
                 weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.metric      = metric
        self.weights     = weights
        self.model_      = None
        self.classes_    : list = []
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[KNN] Fitting kNN (k={self.n_neighbors}, metric={self.metric}) "
              )
        self.model_ = _KNN(
            n_neighbors = self.n_neighbors,
            metric      = self.metric,
            weights     = self.weights,
            algorithm   = "auto",
            n_jobs      = -1,
        )
        self.model_.fit(X_train, y_train)
        print(f"[KNN] Training complete.")
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)
    def save(self, path: str = "models/knn.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[KNN] Saved → {path}")
    @staticmethod
    def load(path: str = "models/knn.pkl") -> "KNNClassifier":
        clf = joblib.load(path)
        print(f"[KNN] Loaded ← {path}")
        return clf
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.predict(X_test)
        acc   = np.mean(preds == y_test)
        print(f"[KNN] Accuracy: {acc * 100:.2f}%")
        return acc
if __name__ == "__main__":
    import sys
    from data_loader import prepare_data
    csv = sys.argv[1] if len(sys.argv) > 1 else "DSL-StrongPasswordData.csv"
    X_train, X_test, y_train, y_test, le, scaler, _ = prepare_data(csv)
    clf = KNNClassifier()
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save()
