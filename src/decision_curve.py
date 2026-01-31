import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_decision_curve():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    probs = model.predict_proba(X_test)[:,1]
    thresholds = np.linspace(0.05,0.95,20)
    net_benefit = []

    for t in thresholds:
        preds = probs>t
        tp = ((preds==1)&(y_test==1)).sum()
        fp = ((preds==1)&(y_test==0)).sum()
        nb = tp/len(y_test) - fp/len(y_test)*(t/(1-t))
        net_benefit.append(nb)

    plt.plot(thresholds, net_benefit)
    plt.xlabel("Threshold")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"decision_curve.png")
    plt.close()
