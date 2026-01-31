import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_risk_over_time():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Convert to numpy
    X_test = np.array(X_test)

    # Load trained model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Select a representative patient (first test sample)
    patient = X_test[0].copy()

    time_steps = 10
    risk_scores = []

    for t in range(time_steps):
        # Simulate temporal variation (small physiological drift)
        noise = np.random.normal(0, 0.02, size=patient.shape)
        patient_t = patient + noise

        prob = model.predict_proba(patient_t.reshape(1, -1))[0, 1]
        risk_scores.append(prob)

    # Plot temporal risk
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, time_steps + 1), risk_scores, marker='o')
    plt.xlabel("Time Step (Clinical Follow-up)")
    plt.ylabel("Predicted Heart Disease Risk")
    plt.title("Temporal Risk Progression for a Patient")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "risk_over_time.png")
    plt.close()

    print("Temporal risk progression plot generated successfully.")
