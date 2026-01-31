import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_ethical_risk_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, _ = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    probs = model.predict_proba(X_test)[:,1]
    uncertainty = np.abs(probs-0.5)

    plt.scatter(probs, uncertainty)
    plt.xlabel("Risk")
    plt.ylabel("Uncertainty")
    plt.title("Ethical Risk Map")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"ethical_risk_map.png")
    plt.close()
