# Project BBA: Biometric Behavior Authentication

Project **BBA** is a state-of-the-art **Continuous Biometric Authentication** system. Unlike traditional login systems that only verify identity once at the start of a session, BBA monitors the user's **typing rhythm** (keystroke dynamics) throughout their entire session to ensure the authorized person is still the one at the keyboard.

---

## 1. System Architecture Overview

The system is built on a "Zero-Trust" architecture using **Python** and **Machine Learning**. It consists of three primary layers:

1.  **Data Acquisition Layer**: Captures raw keyboard events (key-down and key-up) in real-time.
2.  **Feature Engineering Layer**: Processes raw events into mathematical representations of typing behavior.
3.  **Ensemble Intelligence Layer**: A 10-algorithm voting system that decides whether the current typist matches the enrolled profile.

---

## 2. Core Technology: Keystroke Dynamics

The system analyzes how you type, not what you type. It focuses on several key metric categories:

-   **Hold Time (HT)**: How long a key is held down (measures muscle speed/rhythm).
-   **Flight Time (FT)**: The interval between releasing one key and pressing the next (measures coordination).
-   **Trigraph Rhythm**: The timing across sequences of three keys.
-   **Wait-Time (WT)**: The delay between two consecutive key presses.
-   **CPS (Chars Per Second)**: Overall typing velocity.
-   **Distance-Weighted Flight**: Adjusts flight time based on the physical distance between keys on a QWERTY layout.

---

## 3. The 10-Algorithm Ensemble Engine

To ensure an extremely high level of security (the "Biometric Moat"), BBA uses an **Ensemble Voter** that aggregates predictions from 10 diverse machine learning models:

| Algorithm | Role in BBA |
| :--- | :--- |
| **ExtraTrees** | Excellent at handling high-dimensional biometric data. |
| **HistGB** | Fast gradient boosting for large-scale pattern recognition. |
| **SVM (Support Vector Machine)** | Finds the optimal boundary for user identity. |
| **Random Forest** | Robust to outliers and provides stable predictions. |
| **MLP (Neural Network)** | Captures deep, non-linear relationships in typing habits. |
| **KNN (K-Nearest Neighbors)** | Identifies similarities to previous typing samples. |
| **Isolation Forest** | Specifically looks for "anomaly" typing (intruders). |
| **Scaled Manhattan** | Traditional biometric distance metric for high precision. |
| **AdaBoost** | Combines weak patterns into a strong identifier. |
| **Gradient Boosting** | Iteratively optimizes identification accuracy. |

**Voting Mechanism**: Every model "votes" on the user's identity. If there's a tie, the system uses the raw confidence scores (probabilities) from each model to break it.

---

## 4. The Enrollment & Synthesis Workflow

Enrollment is designed to be quick but generates a massive dataset for the AI:

1.  **Baseline Collection**: The user types a standard sentence (e.g., "the quick brown fox...") 5-10 times.
2.  **Synthetic Data Generation**: Since 10 samples aren't enough for deep learning, the system uses **Gaussian Noise Synthesis** to generate **400 unique synthetic samples** that follow the user's natural rhythm with slight variations. 
3.  **Background Noise Calibration**: The system also generates "Background Noise" (non-matching data) to ensure the AI knows what an *incorrect* user looks like.

---

## 5. Live Continuous Monitoring

Once logged in, the **ContinuousAuthMonitor** runs in the background:

-   **Event Window**: It waits for a window of **30 keystroke events** (~15 characters).
-   **Real-time Scrutiny**: As soon as the window is full, the data is pushed to the Ensemble Engine.
-   **Security Enforcement**: If the ensemble determines the typing characteristics don't match (e.g., an unauthorized person takes over the keyboard), the system triggers a **Security Alert** and **Forced Logout**, protecting sensitive enterprise data.

---

## 6. Key Files Summary

-   [gui_app.py](file:///g:/Project/FTAK/gui_app.py): The main User Interface (Titan ERP).
-   [collect_keystrokes.py](file:///g:/Project/FTAK/collect_keystrokes.py): The "Sensor" that tracks your fingers.
-   [ensemble_voting.py](file:///g:/Project/FTAK/ensemble_voting.py): The "Brain" containing all 10 algorithms.
-   [enroll_and_predict.py](file:///g:/Project/FTAK/enroll_and_predict.py): The "Teacher" that trains the AI on your behavior.
