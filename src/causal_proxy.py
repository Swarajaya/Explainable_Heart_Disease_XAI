import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_causal_proxy_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    X_test_proxy_removed = X_test.copy()
    X_test_proxy_removed[:, :2] = 0  # assume age & sex proxies

    proxy_auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test_proxy_removed)[:, 1]
    )

    plt.figure()
    plt.bar(["Original", "Proxy Removed"], [base_auc, proxy_auc])
    plt.ylabel("AUC")
    plt.title("Causal Proxy Dependence")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "causal_proxy_analysis.png")
    plt.close()
