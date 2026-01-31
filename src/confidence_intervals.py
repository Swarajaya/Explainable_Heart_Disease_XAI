import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_confidence_interval_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    patient = X_test[0]
    samples = []

    for _ in range(100):
        noise = np.random.normal(0, 0.02, patient.shape)
        samples.append(
            model.predict_proba((patient + noise).reshape(1, -1))[0, 1]
        )

    mean = np.mean(samples)
    std = np.std(samples)

    plt.figure()
    plt.bar(["Risk"], [mean], yerr=[std])
    plt.ylabel("Predicted Risk")
    plt.title("Risk Confidence Interval")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "risk_confidence_intervals.png")
    plt.close()
