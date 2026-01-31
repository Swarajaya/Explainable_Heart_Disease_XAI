import os
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_shap_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    # Load data
    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Load trained XGBoost model
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    # Background for KernelExplainer
    background = shap.sample(X_train, 50)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    # SHAP values for positive class
    shap_values = explainer.shap_values(X_test[:30])[1]

    # 🔥 TIER-1 FIX: Mean |SHAP| Feature Importance
    mean_shap = np.mean(np.abs(shap_values), axis=0)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mean_shap)), mean_shap)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean |SHAP Value|")
    plt.title("SHAP Feature Importance (KernelExplainer)")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH + "shap_summary.png")
    plt.close()

    print("SHAP feature-importance plot generated successfully.")
