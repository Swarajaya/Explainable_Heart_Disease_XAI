import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_model_agreement_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_test = np.array(X_test)

    # Load models
    xgb_model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    lstm_model = load_model(OUTPUT_MODEL_PATH + "lstm_model.h5")

    # Predictions
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    lstm_probs = lstm_model.predict(
        np.expand_dims(X_test, axis=1)
    ).ravel()

    # Convert to risk labels
    def risk_label(p):
        if p < 0.3:
            return 0  # Low
        elif p <= 0.6:
            return 1  # Medium
        else:
            return 2  # High

    xgb_labels = np.array([risk_label(p) for p in xgb_probs])
    lstm_labels = np.array([risk_label(p) for p in lstm_probs])

    # Agreement matrix
    agreement = np.zeros((3, 3))
    for x, l in zip(xgb_labels, lstm_labels):
        agreement[x, l] += 1

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        agreement,
        annot=True,
        fmt=".0f",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        cmap="Blues"
    )

    plt.xlabel("LSTM Risk Prediction")
    plt.ylabel("XGBoost Risk Prediction")
    plt.title("Model Agreement & Uncertainty Matrix")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "model_agreement_matrix.png")
    plt.close()

    print("Model agreement matrix generated successfully.")
