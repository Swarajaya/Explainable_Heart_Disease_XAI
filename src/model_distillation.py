import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.config import *


def generate_distillation_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    teacher = joblib.load(OUTPUT_MODEL_PATH + "xgboost_model.pkl")
    soft_labels = teacher.predict_proba(X_train)[:, 1]

    student = LogisticRegression(max_iter=1000)
    student.fit(X_train, soft_labels > 0.5)

    teacher_auc = roc_auc_score(y_test, teacher.predict_proba(X_test)[:, 1])
    student_auc = roc_auc_score(y_test, student.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.bar(["Teacher (XGB)", "Student (LR)"], [teacher_auc, student_auc])
    plt.ylabel("AUC")
    plt.title("Model Distillation Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH + "model_distillation.png")
    plt.close()
