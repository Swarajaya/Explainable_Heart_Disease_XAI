import os, numpy as np, matplotlib.pyplot as plt
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *

def generate_distribution_shift_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    X_train, X_test, _, _ = preprocess_data(df)

    plt.hist(X_train[:,0], alpha=0.5, label="Train")
    plt.hist(X_test[:,0], alpha=0.5, label="Test")
    plt.legend()
    plt.title("Representation Shift (Feature 0)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"distribution_shift.png")
    plt.close()
