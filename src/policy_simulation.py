import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_policy_simulation_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, _ = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    thresholds=[0.4,0.5,0.6]
    treated=[(model.predict_proba(X_test)[:,1]>t).sum() for t in thresholds]

    plt.bar(["Policy A","Policy B","Policy C"], treated)
    plt.ylabel("Patients Treated")
    plt.title("Policy What-If Simulation")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"policy_simulation.png")
    plt.close()
