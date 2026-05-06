import argparse
import time
import numpy as np
import joblib
import os
import sys
import threading
import math
from collections import defaultdict

try:
    from pynput import keyboard as kb
except ImportError:
    sys.exit("[Error] pynput not installed. Run: pip install pynput")

# QWERTY Coordinate Map for Finger Distance Calculation
KEY_COORDS = {
    '1': (0, 0), '2': (1,0), '3': (2,0), '4': (3,0), '5': (4,0), '6': (5,0), '7': (6,0), '8': (7,0), '9': (8,0), '0': (9,0),
    'q': (0.5, 1), 'w': (1.5,1), 'e': (2.5,1), 'r': (3.5,1), 't': (4.5,1), 'y': (5.5,1), 'u': (6.5,1), 'i': (7.5,1), 'o': (8.5,1), 'p': (9.5,1),
    'a': (0.75, 2), 's': (1.75,2), 'd': (2.75,2), 'f': (3.75,2), 'g': (4.75,2), 'h': (5.75,2), 'j': (6.75,2), 'k': (7.75,2), 'l': (8.75,2),
    'z': (1.25, 3), 'x': (2.25,3), 'c': (3.25,3), 'v': (4.25,3), 'b': (5.25,3), 'n': (6.25,3), 'm': (7.25,3),
}

FEATURE_COLS = [
    "mean_ht", "std_ht",          # Muscle Speed & Rhythm (Hold Time)
    "mean_ft", "std_ft",          # Rhythm (Flight Time)
    "cps",                        # Typing Speed (Chars Per Second)
    "mean_dist_weighted_ft",      # Finger Distance / Muscle coordination
    "mean_p1_p3_tri",             # Trigraph rhythm
    "p1_p2_mean", "p2_p3_mean"    # Digraph/Trigraph components
]

# Add individual key hold times as features (a-z, 0-9)
ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789"
for char in ALPHANUM:
    FEATURE_COLS.append(f"ht_{char}")

class KeyCapture:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()
        self._done_ev = threading.Event()

    def _on_press(self, key):
        t = time.perf_counter()
        try:
            ch = getattr(key, 'char', None)
            if ch:
                ch = ch.lower()
                print(ch, end='', flush=True)
                with self._lock:
                    self.events.append((t, 'down', ch))
            elif key == kb.Key.enter:
                print('[ENTER]', end='', flush=True)
                with self._lock:
                    self.events.append((t, 'down', 'enter'))
            elif key == kb.Key.esc:
                self._done_ev.set()
                return False
        except Exception:
            pass

    def _on_release(self, key):
        t = time.perf_counter()
        try:
            ch = getattr(key, 'char', None)
            if ch:
                ch = ch.lower()
                with self._lock:
                    self.events.append((t, 'up', ch))
            elif key == kb.Key.enter:
                with self._lock:
                    self.events.append((t, 'up', 'enter'))
                print()
                self._done_ev.set()
                return False
        except Exception:
            pass

    def capture(self, timeout: float = 60.0) -> list:
        self._done_ev.clear()
        self.events = []
        with kb.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            self._done_ev.wait(timeout=timeout)
        return list(self.events)

def get_distance(k1, k2):
    if k1 not in KEY_COORDS or k2 not in KEY_COORDS:
        return 1.0 # Default distance
    x1, y1 = KEY_COORDS[k1]
    x2, y2 = KEY_COORDS[k2]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def extract_features(events: list) -> np.ndarray | None:
    if not events: return None
    
    presses = defaultdict(list)
    releases = defaultdict(list)
    full_sequence = [] # List of (char, press_time, release_time)

    # Simple matching of press/release
    temp_press = {}
    for t, action, tok in events:
        if action == 'down':
            temp_press[tok] = t
        elif action == 'up' and tok in temp_press:
            p_time = temp_press.pop(tok)
            presses[tok].append(p_time)
            releases[tok].append(t)
            full_sequence.append((tok, p_time, t))

    if len(full_sequence) < 2:
        return None

    # Sort full sequence by press time
    full_sequence.sort(key=lambda x: x[1])

    # 1. Hold Times (HT)
    hts = [r - p for _, p, r in full_sequence]
    mean_ht = np.mean(hts)
    std_ht = np.std(hts)

    # 2. Flight Times (FT) - press[i+1] - release[i]
    fts = []
    dist_weighted_fts = []
    for i in range(len(full_sequence) - 1):
        ft = full_sequence[i+1][1] - full_sequence[i][2]
        fts.append(ft)
        
        # Distance weighting
        dist = get_distance(full_sequence[i][0], full_sequence[i+1][0])
        dist_weighted_fts.append(ft / (dist + 0.1)) # Avoid div by zero

    mean_ft = np.mean(fts) if fts else 0.0
    std_ft = np.std(fts) if fts else 0.0
    mean_dw_ft = np.mean(dist_weighted_fts) if dist_weighted_fts else 0.0

    # 3. CPS (Chars Per Second)
    total_time = full_sequence[-1][2] - full_sequence[0][1]
    cps = len(full_sequence) / total_time if total_time > 0 else 0.0

    # 4. Trigraphs (p3 - p1)
    trigraph_timings = []
    p1_p2_digraph = []
    p2_p3_digraph = []
    for i in range(len(full_sequence) - 2):
        trigraph_timings.append(full_sequence[i+2][1] - full_sequence[i][1])
        p1_p2_digraph.append(full_sequence[i+1][1] - full_sequence[i][1])
        p2_p3_digraph.append(full_sequence[i+2][1] - full_sequence[i+1][1])

    mean_tri = np.mean(trigraph_timings) if trigraph_timings else 0.0
    mean_p1p2 = np.mean(p1_p2_digraph) if p1_p2_digraph else 0.0
    mean_p2p3 = np.mean(p2_p3_digraph) if p2_p3_digraph else 0.0

    # 5. Key-Specific Hold Times
    key_hts = {char: 0.0 for char in ALPHANUM}
    for char, p, r in full_sequence:
        if char in key_hts:
            # If multiple occurrences, take average
            if key_hts[char] == 0.0:
                key_hts[char] = (r - p)
            else:
                key_hts[char] = (key_hts[char] + (r - p)) / 2.0

    # Construct feature vector
    features = [
        mean_ht, std_ht, mean_ft, std_ft, cps, mean_dw_ft, mean_tri, mean_p1p2, mean_p2p3
    ]
    for char in ALPHANUM:
        features.append(key_hts[char])

    return np.array(features, dtype=np.float64)

def collect_samples(n_reps: int = 5) -> np.ndarray:
    rows = []
    capturer = KeyCapture()
    enroll_text = "the quick brown fox jumps over the lazy dog 1234567890"

    rep = 1
    while rep <= n_reps:
        print(f"\n{'='*55}")
        print(f"  Repetition {rep}/{n_reps}")
        print(f"  Type: {enroll_text}")
        print(f"{'='*55}")
        print("  → Type now:\n  > ", end='', flush=True)

        events = capturer.capture(timeout=60)
        feats = extract_features(events)
        if feats is None:
            print("\n  ✗  Capture failed or too short. Let's try again.")
            continue

        rows.append(feats)
        print(f"\n  ✓  Captured ({len(events)} events)")
        time.sleep(0.5)
        rep += 1

    return np.array(rows)

def load_models(model_dir: str):
    def _ld(f):
        p = os.path.join(model_dir, f)
        if not os.path.exists(p):
            sys.exit(f"[Error] Missing model file: {p}\n  Run train.py first.")
        return joblib.load(p)

    scaler = _ld("scaler.pkl")
    le = _ld("label_encoder.pkl")
    # feature_cols is usually loaded for reference, but we use FEATURE_COLS global
    
    sys.path.insert(0, os.path.dirname(__file__))
    from ensemble_voting import EnsembleVoter
    voter = EnsembleVoter()
    voter.le_ = le
    voter.load_all(model_dir)
    return voter, scaler, le, FEATURE_COLS

def predict_me(samples: np.ndarray, voter, scaler, le, feature_cols: list):
    X_scaled = scaler.transform(samples)

    print(f"\n{'='*55}")
    print(f"  Predicting from {len(samples)} samples …")
    print(f"{'='*55}")

    all_votes = {}
    for i, x in enumerate(X_scaled):
        print(f"\n  ── Sample {i+1} ──────────────────────────────────")
        name = voter.predict_single(x, verbose=True)
        all_votes[name] = all_votes.get(name, 0) + 1

    best = max(all_votes, key=all_votes.get)
    print(f"\n{'='*55}")
    print(f"  AGGREGATE VOTE: {best}")
    print(f"{'='*55}\n")
    return best

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--reps", type=int, default=3)
    args.add_argument("--model-dir", default="models")
    parsed = args.parse_args()

    print("=" * 56)
    print("   Free-Text Keystroke Dynamics Authentication")
    print("=" * 56)
    input("  Press ENTER to begin …")

    samples = collect_samples(n_reps=parsed.reps)
    if len(samples) == 0:
        return

    voter, scaler, le, _ = load_models(parsed.model_dir)
    predict_me(samples, voter, scaler, le, FEATURE_COLS)

if __name__ == "__main__":
    main()
