import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_cost_sensitive_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    probs = model.predict_proba(X_test)[:,1]
    thresholds = np.linspace(0.1,0.9,20)
    costs = []

    for t in thresholds:
        preds = (probs>t).astype(int)
        fn = ((preds==0)&(y_test==1)).sum()
        fp = ((preds==1)&(y_test==0)).sum()
        costs.append(fn*5 + fp*1)   # FN cost > FP

    plt.plot(thresholds,costs)
    plt.xlabel("Threshold")
    plt.ylabel("Clinical Cost")
    plt.title("Cost-Sensitive Decision Tradeoff")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"cost_sensitive_tradeoff.png")
    plt.close()
