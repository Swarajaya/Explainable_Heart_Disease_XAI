import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_feature_interaction_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    f1, f2 = 0, 1  # two features
    grid = np.linspace(-2, 2, 20)
    Z = np.zeros((20, 20))

    base = X_test.mean(axis=0)

    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            sample = base.copy()
            sample[f1] += x
            sample[f2] += y
            Z[i, j] = model.predict_proba(sample.reshape(1, -1))[0, 1]

    plt.figure()
    plt.imshow(Z, origin="lower")
    plt.colorbar(label="Risk")
    plt.title("Feature Interaction Surface")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "feature_interactions.png")
    plt.close()
