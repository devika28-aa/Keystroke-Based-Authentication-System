import numpy as np
from collections import Counter
import joblib
import os

from algo_extra_trees       import ExtraTreesClassifierWrapper
from algo_hist_gb           import HistGBClassifierWrapper
from algo_svm               import SVMClassifier
from algo_random_forest     import RandomForestClassifier
from algo_mlp               import MLPClassifierWrapper
from algo_knn               import KNNClassifier
from algo_isolation_forest  import IsolationForestClassifier
from algo_scaled_manhattan  import ScaledManhattanClassifier
from algo_adaboost          import AdaBoostClassifierWrapper
from algo_gradient_boosting import GradientBoostingClassifierWrapper

class EnsembleVoter:

    ALGORITHM_NAMES = [
        "ExtraTrees", "HistGB", "SVM", "RandomForest", "MLP",
        "KNN", "IsolationForest", "ScaledManhattan", "AdaBoost", "GradientBoosting"
    ]

    def __init__(self):
        self.classifiers = {
            "ExtraTrees":       ExtraTreesClassifierWrapper(),
            "HistGB":           HistGBClassifierWrapper(),
            "SVM":              SVMClassifier(),
            "RandomForest":     RandomForestClassifier(),
            "MLP":              MLPClassifierWrapper(),
            "KNN":              KNNClassifier(),
            "IsolationForest":  IsolationForestClassifier(),
            "ScaledManhattan":  ScaledManhattanClassifier(),
            "AdaBoost":          AdaBoostClassifierWrapper(),
            "GradientBoosting": GradientBoostingClassifierWrapper(),
        }
        self.le_       = None
        self.classes_  = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray, le=None) -> None:
        self.le_      = le
        self.classes_ = sorted(set(y_train))

        for name, clf in self.classifiers.items():
            print(f"  Training {name} ...")
            clf.train(X_train, y_train)
        
        print("\n[Ensemble] All 10 models trained and ready. ✓")

    def _vote_int(self, X: np.ndarray) -> tuple:
        preds  = {}
        probas = {}

        for name, clf in self.classifiers.items():
            preds[name]  = clf.predict(X)
            # Some models might not have predict_proba or it might fail if one-class
            try:
                probas[name] = clf.predict_proba(X)
            except:
                probas[name] = None

        n_samples = X.shape[0]
        winners      = np.empty(n_samples, dtype=int)
        vote_details = []

        for i in range(n_samples):
            votes_i = {name: int(preds[name][i]) for name in self.ALGORITHM_NAMES}
            counter = Counter(votes_i.values())
            max_votes = max(counter.values())

            candidates = [cls for cls, cnt in counter.items() if cnt == max_votes]

            if len(candidates) == 1:
                winner = candidates[0]
            else:
                # Tie-breaking with probabilities
                best_score = -np.inf
                winner     = candidates[0]
                for cand in candidates:
                    score = 0.0
                    for name, clf in self.classifiers.items():
                        if probas[name] is not None:
                            # if cand is in clf.classes_
                            if hasattr(clf, "classes_") and cand in clf.classes_:
                                col = list(clf.classes_).index(cand)
                                score += probas[name][i, col]
                    if score > best_score:
                        best_score = score
                        winner     = cand

            winners[i]      = winner
            vote_details.append({
                "votes":   votes_i,
                "tally":   dict(counter),
                "winner":  winner,
            })

        return winners, vote_details

    def predict(self, X: np.ndarray) -> np.ndarray:
        winners, _ = self._vote_int(X)
        if self.le_ is not None:
            return self.le_.inverse_transform(winners)
        return winners

    def predict_with_details(self, X: np.ndarray) -> tuple:
        winners, details = self._vote_int(X)
        if self.le_ is not None:
            names = self.le_.inverse_transform(winners)
        else:
            names = winners
        return names, details

    def predict_single(self, x: np.ndarray, verbose: bool = True) -> str:
        X = x.reshape(1, -1)
        names, details = self.predict_with_details(X)
        if verbose:
            d = details[0]
            print(f"  Votes: {d['votes']}")
            print(f"  Winner: {names[0]}")
        return names[0]

    def save_all(self, model_dir: str = "models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        for name, clf in self.classifiers.items():
            path = os.path.join(model_dir, f"{name.lower()}.pkl")
            joblib.dump(clf, path)
        if self.le_:
            joblib.dump(self.le_, os.path.join(model_dir, "label_encoder.pkl"))

    def load_all(self, model_dir: str = "models") -> None:
        mapping = {n.lower(): n for n in self.ALGORITHM_NAMES}
        for fname in os.listdir(model_dir):
            stem = fname.replace(".pkl", "").lower()
            if stem in mapping:
                self.classifiers[mapping[stem]] = joblib.load(os.path.join(model_dir, fname))
        
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        if os.path.exists(le_path):
            self.le_ = joblib.load(le_path)
