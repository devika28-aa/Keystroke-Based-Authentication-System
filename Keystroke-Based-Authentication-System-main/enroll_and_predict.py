import sys
import os
import numpy as np
import pandas as pd
import joblib

# Import from updated modules
sys.path.append(os.path.dirname(__file__))
from collect_keystrokes import collect_samples, predict_me, load_models, FEATURE_COLS
from data_loader import prepare_data
from ensemble_voting import EnsembleVoter

def synthesize_samples(base_samples: np.ndarray, target_count: int = 400) -> np.ndarray:
    """
    Synthesize more samples from a small base set by adding Gaussian noise.
    """
    n_existing = len(base_samples)
    if n_existing >= target_count:
        return base_samples

    n_needed = target_count - n_existing
    
    # Calculate mean and std for each feature
    mean_vec = np.mean(base_samples, axis=0)
    std_vec  = np.std(base_samples, axis=0)
    
    # Ensure at least some noise even if std is 0 (one sample or identical samples)
    std_vec = np.clip(std_vec, 0.001, None)
    
    # For rhythm features (first 9), use 5% noise
    # For key-specific features, use same
    noise_rel = 0.05 
    noise_std = np.maximum(std_vec, mean_vec * noise_rel)

    print(f"[Synthesis] Generating {n_needed} synthetic samples ...")
    synthetic = np.random.normal(loc=mean_vec, scale=noise_std, size=(n_needed, base_samples.shape[1]))
    
    # Clip to sensible values (timing can't be negative)
    synthetic = np.clip(synthetic, 0.0001, None)
    
    all_samples = np.vstack([base_samples, synthetic])
    return all_samples

def get_dataset_csv() -> str:
    path = "free_text_keystroke_data.csv"
    if not os.path.exists(path):
        # Create empty dataframe with correct columns
        df = pd.DataFrame(columns=["subject", "sessionIndex", "rep"] + FEATURE_COLS)
        df.to_csv(path, index=False)
        print(f"[Dataset] Created new dataset file: {path}")
    return path

def append_to_dataset(username: str, samples: np.ndarray, csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    
    # Check current subjects
    unique_subjects = df["subject"].unique().tolist()
    
    # We need at least 2 classes for multiclass voting.
    # If we have < 2 total subjects, and 'Background_Noise' isn't one of them, add it.
    if "Background_Noise" not in unique_subjects and len(unique_subjects) < 2:
        print("[Dataset] Adding 'Background_Noise' subject to satisfy 2-class requirement.")
        bg_samples = []
        for i in range(400):
            # First 9: mean_ht, std_ht, mean_ft, std_ft, cps, mean_dw_ft, mean_tri, mean_p1p2, mean_p2p3
            rhythm = [0.12 + np.random.normal(0,0.01), 0.05, 0.15 + np.random.normal(0,0.02), 0.08, 4.0, 0.2, 0.3, 0.15, 0.15]
            # Alphanumeric hold times (36)
            keys = [0.11 + np.random.normal(0, 0.02) for _ in range(36)]
            bg_samples.append(rhythm + keys)
            
        bg_df = pd.DataFrame(bg_samples, columns=FEATURE_COLS)
        bg_df["subject"] = "Background_Noise"
        bg_df["sessionIndex"] = 1
        bg_df["rep"] = range(1, 401)
        df = pd.concat([df, bg_df], ignore_index=True)

    new_rows = []
    for i, row in enumerate(samples):
        record = {
            "subject": username,
            "sessionIndex": 1,
            "rep": i + 1
        }
        for j, col in enumerate(FEATURE_COLS):
            record[col] = row[j]
        new_rows.append(record)
        
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    
    print(f"[Dataset] Saving {len(samples)} samples for '{username}' to {csv_path} ...")
    df_combined.to_csv(csv_path, index=False)
    return csv_path

def main():
    print("=" * 60)
    print("  Free-Text Enrollment & System Training")
    print("=" * 60)
    
    username = input("\n  Enter Username to Enroll: ").strip()
    if not username: return

    print(f"\n[1/4] Collecting 5 Enrollment Samples (Rhythm Baseline)")
    raw_samples = collect_samples(n_reps=5)
    
    print("\n[2/4] Synthesizing 400 Samples ...")
    all_user_samples = synthesize_samples(raw_samples, target_count=400)

    print("\n[3/4] Updating Dataset ...")
    csv_path = get_dataset_csv()
    append_to_dataset(username, all_user_samples, csv_path)

    print("\n[4/4] Retraining 10-Model Ensemble ...")
    X_train, X_test, y_train, y_test, le, scaler, actual_cols = prepare_data(csv_path)
    
    voter = EnsembleVoter()
    voter.train(X_train, y_train, le=le)
    
    model_dir = "models"
    voter.save_all(model_dir)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    
    print("\n  [Success] System trained with 10 algorithms.")
    print("  Run gui_app.py to test live verification.")

if __name__ == "__main__":
    main()
