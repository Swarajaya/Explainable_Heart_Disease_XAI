import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_threshold_analysis():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load trained XGBoost model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 20)
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision_scores.append(
            precision_score(y_test, preds, zero_division=0)
        )
        recall_scores.append(
            recall_score(y_test, preds, zero_division=0)
        )

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_scores, label="Precision", marker="o")
    plt.plot(thresholds, recall_scores, label="Recall", marker="o")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision–Recall Tradeoff Across Thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "threshold_analysis.png")
    plt.close()

    print("Decision threshold analysis plot generated successfully.")
