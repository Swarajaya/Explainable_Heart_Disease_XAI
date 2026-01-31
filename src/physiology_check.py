import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_physiology_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, _ = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    chol_idx = 4
    values = np.linspace(-1,1,30)
    risks = []

    base = X_test.mean(axis=0)
    for v in values:
        temp = base.copy()
        temp[chol_idx]+=v
        risks.append(model.predict_proba(temp.reshape(1,-1))[0,1])

    plt.plot(values, risks)
    plt.xlabel("Cholesterol Change")
    plt.ylabel("Risk")
    plt.title("Physiological Consistency Check")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"physiology_consistency.png")
    plt.close()
