import os, joblib, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_human_ai_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    _, X_test, _, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    probs = model.predict_proba(X_test)[:,1]

    auto_preds = (probs > 0.5).astype(int)
    human_override = np.where((probs > 0.4) & (probs < 0.6), y_test, auto_preds)

    auto_acc = (auto_preds == y_test).mean()
    human_acc = (human_override == y_test).mean()

    plt.bar(["AI Only", "AI + Doctor"], [auto_acc, human_acc])
    plt.ylabel("Accuracy")
    plt.title("Human-in-the-Loop Performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"human_ai_collaboration.png")
    plt.close()
