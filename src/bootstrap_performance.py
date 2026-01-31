import os, joblib, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_bootstrap_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    aucs=[]
    for _ in range(100):
        idx = np.random.choice(len(X_test), len(X_test), replace=True)
        aucs.append(roc_auc_score(y_test.iloc[idx], model.predict_proba(X_test[idx])[:,1]))

    plt.hist(aucs)
    plt.title("Bootstrapped AUC Stability")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"bootstrap_performance.png")
    plt.close()
