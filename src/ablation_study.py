import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_ablation_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    base_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    drops = []

    for i in range(X_test.shape[1]):
        X_test_ablated = X_test.copy()
        X_test_ablated[:, i] = 0
        auc = roc_auc_score(y_test, model.predict_proba(X_test_ablated)[:, 1])
        drops.append(base_auc - auc)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(drops)), drops)
    plt.xlabel("Feature Index Removed")
    plt.ylabel("AUC Drop")
    plt.title("Feature Ablation Study")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "ablation_study.png")
    plt.close()
