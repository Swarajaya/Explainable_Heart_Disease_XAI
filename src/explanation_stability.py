import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_explanation_stability_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_test = np.array(X_test)

    # Load trained model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Select a representative patient
    patient = X_test[0]

    perturbation_steps = 15
    risk_scores = []

    for i in range(perturbation_steps):
        # Small controlled perturbation
        noise = np.random.normal(0, 0.01, size=patient.shape)
        perturbed_patient = patient + noise

        prob = model.predict_proba(
            perturbed_patient.reshape(1, -1)
        )[0, 1]

        risk_scores.append(prob)

    # Plot stability
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, perturbation_steps + 1),
        risk_scores,
        marker="o"
    )
    plt.xlabel("Perturbation Step")
    plt.ylabel("Predicted Risk Probability")
    plt.title("Prediction Stability Under Input Perturbations")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "explanation_stability.png")
    plt.close()

    print("Explanation stability analysis generated successfully.")
