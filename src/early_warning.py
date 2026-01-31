import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_early_warning_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_test = np.array(X_test)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    patient = X_test[0]
    risks = []

    for step in range(10):
        drift = patient + np.random.normal(0, 0.03, patient.shape)
        risks.append(model.predict_proba(drift.reshape(1, -1))[0, 1])

    plt.figure()
    plt.plot(range(1, 11), risks, marker="o")
    plt.xlabel("Time Step")
    plt.ylabel("Risk")
    plt.title("Early Warning Risk Trend")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "early_warning_curve.png")
    plt.close()
