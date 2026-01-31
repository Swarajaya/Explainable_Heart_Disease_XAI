import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_counterfactual_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_test = np.array(X_test)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    patient = X_test[0].copy()
    base_risk = model.predict_proba(patient.reshape(1, -1))[0, 1]

    deltas = np.linspace(-0.5, 0.5, 20)
    risks = []

    for d in deltas:
        modified = patient.copy()
        modified[0] += d  # minimal change in first feature
        risks.append(model.predict_proba(modified.reshape(1, -1))[0, 1])

    plt.figure()
    plt.plot(deltas, risks)
    plt.axhline(base_risk, linestyle="--", label="Original Risk")
    plt.xlabel("Feature Change")
    plt.ylabel("Predicted Risk")
    plt.title("Counterfactual Risk Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "counterfactual_changes.png")
    plt.close()
