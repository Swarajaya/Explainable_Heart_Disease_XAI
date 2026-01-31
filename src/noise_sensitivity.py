import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_noise_sensitivity_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    noise_levels = np.linspace(0, 0.1, 10)
    aucs = []

    for n in noise_levels:
        noisy_X = X_test + np.random.normal(0, n, X_test.shape)
        aucs.append(roc_auc_score(y_test, model.predict_proba(noisy_X)[:, 1]))

    plt.figure()
    plt.plot(noise_levels, aucs, marker="o")
    plt.xlabel("Noise Level")
    plt.ylabel("AUC")
    plt.title("Noise Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "noise_sensitivity.png")
    plt.close()
