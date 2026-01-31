import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_fairness_plots():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    probs = model.predict_proba(X_test)[:, 1]

    # Gender fairness (sex: 0=female, 1=male)
    sex = df.loc[y_test.index, "sex"].values
    male_risk = probs[sex == 1].mean()
    female_risk = probs[sex == 0].mean()

    plt.figure()
    plt.bar(["Female", "Male"], [female_risk, male_risk])
    plt.ylabel("Average Predicted Risk")
    plt.title("Gender Fairness Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "fairness_gender.png")
    plt.close()

    # Age fairness
    age = df.loc[y_test.index, "age"].values
    young = probs[age < 50].mean()
    old = probs[age >= 50].mean()

    plt.figure()
    plt.bar(["Age < 50", "Age ≥ 50"], [young, old])
    plt.ylabel("Average Predicted Risk")
    plt.title("Age-Based Fairness Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "fairness_age.png")
    plt.close()
