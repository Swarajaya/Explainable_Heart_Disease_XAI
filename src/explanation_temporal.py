import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_temporal_explanation_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, _ = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH+"xgboost_model.pkl")

    patient = X_test[0]
    scores=[]
    for _ in range(10):
        noise = np.random.normal(0,0.02,patient.shape)
        scores.append(model.predict_proba((patient+noise).reshape(1,-1))[0,1])

    plt.plot(scores, marker="o")
    plt.title("Temporal Explanation Consistency")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"explanation_temporal_consistency.png")
    plt.close()
