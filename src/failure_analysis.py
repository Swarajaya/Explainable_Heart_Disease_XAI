import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_failure_analysis_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    preds = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Failure Case Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "failure_case_analysis.png")
    plt.close()
