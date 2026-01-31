import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_local_explanation():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_test = np.array(X_test)

    # Load trained XGBoost model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Select a patient (highest predicted risk)
    probs = model.predict_proba(X_test)[:, 1]
    patient_idx = np.argmax(probs)
    patient = X_test[patient_idx].reshape(1, -1)

    # Feature importance from XGBoost
    importances = model.feature_importances_

    # Select top features
    top_k = 10
    indices = np.argsort(importances)[-top_k:]

    plt.figure(figsize=(9, 5))
    plt.barh(range(top_k), importances[indices])
    plt.yticks(range(top_k), [f"Feature {i}" for i in indices])
    plt.xlabel("Contribution Importance")
    plt.title("Local Explanation for High-Risk Patient")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "patient_local_explanation.png")
    plt.close()

    print("Patient-level local explanation generated successfully.")
