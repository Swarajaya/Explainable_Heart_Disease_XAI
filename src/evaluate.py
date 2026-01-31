import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)

from tensorflow.keras.models import load_model
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def evaluate_and_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # ---------------- LSTM ----------------
    lstm = load_model(OUTPUT_MODEL_PATH + "lstm_model.h5")

    # LSTM expects sequences → use mean aggregation for eval
    lstm_preds = lstm.predict(
        np.expand_dims(X_test, axis=1)
    ).ravel()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, lstm_preds)
    plt.figure()
    plt.plot(fpr, tpr, label="LSTM ROC")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(OUTPUT_PLOT_PATH + "roc_curve.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, (lstm_preds > 0.5).astype(int))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(OUTPUT_PLOT_PATH + "confusion_matrix.png")
    plt.close()

    # ---------------- XGBOOST ----------------
    xgb = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # Feature Importance
    plt.figure()
    plt.bar(range(X_test.shape[1]), xgb.feature_importances_)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(OUTPUT_PLOT_PATH + "feature_importance.png")
    plt.close()

    print("Evaluation & plots generated successfully.")
