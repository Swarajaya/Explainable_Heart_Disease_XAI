import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_rejection_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_test = np.array(X_test)

    xgb = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    lstm = load_model(OUTPUT_MODEL_PATH + "lstm_model.h5")

    xgb_p = xgb.predict_proba(X_test)[:,1]
    lstm_p = lstm.predict(X_test[:,None,:]).ravel()

    disagreement = np.abs(xgb_p - lstm_p)

    plt.figure()
    plt.hist(disagreement, bins=20)
    plt.xlabel("Prediction Disagreement")
    plt.ylabel("Count")
    plt.title("Confidence-Based Prediction Rejection")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "rejection_analysis.png")
    plt.close()
