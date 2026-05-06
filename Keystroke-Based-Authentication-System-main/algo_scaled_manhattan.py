import numpy as np
import joblib, os
class ScaledManhattanClassifier:
    def __init__(self, eps: float = 1e-8):
        self.eps      = eps          
        self.classes_ : list = []
        self.means_   : dict = {}    
        self.mads_    : dict = {}    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classes_ = sorted(set(y_train))
        print(f"[ScaledManhattan] Building prototypes for {len(self.classes_)} subjects …")
        for subj in self.classes_:
            X_sub               = X_train[y_train == subj]
            mu                  = X_sub.mean(axis=0)
            mad                 = np.mean(np.abs(X_sub - mu), axis=0)
            mad[mad < self.eps] = self.eps          
            self.means_[subj]   = mu
            self.mads_[subj]    = mad
        print(f"[ScaledManhattan] Prototypes built.")
    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        dists = np.column_stack([
            np.sum(np.abs(X - self.means_[s]) / self.mads_[s], axis=1)
            for s in self.classes_
        ])
        return dists                             
    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = self._distance_matrix(X)
        idx   = np.argmin(dists, axis=1)
        return np.array([self.classes_[i] for i in idx])
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists  = self._distance_matrix(X)
        inv    = 1.0 / (dists + self.eps)
        row_sum = inv.sum(axis=1, keepdims=True)
        return inv / row_sum
    def save(self, path: str = "models/scaled_manhattan.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[ScaledManhattan] Saved → {path}")
    @staticmethod
    def load(path: str = "models/scaled_manhattan.pkl") -> "ScaledManhattanClassifier":
        clf = joblib.load(path)
        print(f"[ScaledManhattan] Loaded ← {path}")
        return clf
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.predict(X_test)
        acc   = np.mean(preds == y_test)
        print(f"[ScaledManhattan] Accuracy: {acc * 100:.2f}%")
        return acc
if __name__ == "__main__":
    import sys
    from data_loader import prepare_data
    csv = sys.argv[1] if len(sys.argv) > 1 else "DSL-StrongPasswordData.csv"
    X_train, X_test, y_train, y_test, le, scaler, _ = prepare_data(csv)
    clf = ScaledManhattanClassifier()
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save()
