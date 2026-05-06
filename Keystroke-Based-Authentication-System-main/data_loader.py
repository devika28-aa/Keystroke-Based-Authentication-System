import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os

# Import FEATURE_COLS from collect_keystrokes to keep them synchronized
sys.path.append(os.path.dirname(__file__))
from collect_keystrokes import FEATURE_COLS

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    required_cols = ["subject"] + FEATURE_COLS
    
    # Check if all required columns are in the dataframe
    # If not, we might be loading an old dataset. We'll handle it by selecting only available ones if necessary,
    # but for a "perfect model" we should ensure data consistency.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[Warning] Missing columns in {csv_path}: {len(missing)} columns missing.")
        # Only take what is available
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = df[available_features].values
        actual_cols = available_features
    else:
        X_raw = df[FEATURE_COLS].values
        actual_cols = FEATURE_COLS

    y_raw = df["subject"].values
    return X_raw, y_raw, actual_cols

def prepare_data(csv_path: str, test_size=0.20, random_state=42):
    X_raw, y_raw, actual_cols = load_dataset(csv_path)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    # Use stratify only if we have enough samples for each class
    counts = pd.Series(y_encoded).value_counts()
    if counts.min() > 1:
        stratify = y_encoded
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[DataLoader] Loaded {len(X_raw)} rows, {len(le.classes_)} subjects.")
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler, actual_cols
