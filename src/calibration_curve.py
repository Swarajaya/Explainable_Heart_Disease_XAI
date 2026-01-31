import os
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_calibration_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")

    probs = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "calibration_curve.png")
    plt.close()
