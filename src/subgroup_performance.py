import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_subgroup_performance_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    probs = model.predict_proba(X_test)[:, 1]

    age = df.loc[y_test.index, "age"].values

    auc_young = roc_auc_score(y_test[age < 50], probs[age < 50])
    auc_old = roc_auc_score(y_test[age >= 50], probs[age >= 50])

    plt.figure()
    plt.bar(["Age < 50", "Age ≥ 50"], [auc_young, auc_old])
    plt.ylabel("AUC")
    plt.title("Subgroup Performance Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "subgroup_performance.png")
    plt.close()
