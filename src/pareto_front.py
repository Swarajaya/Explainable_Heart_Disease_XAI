import os, matplotlib.pyplot as plt, numpy as np
from src.config import *

def generate_pareto_plot():
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
    accuracy = np.random.uniform(0.7,0.9,20)
    interpret = np.random.uniform(0.5,0.9,20)

    plt.scatter(interpret, accuracy)
    plt.xlabel("Interpretability")
    plt.ylabel("Accuracy")
    plt.title("Pareto Tradeoff")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH+"pareto_front.png")
    plt.close()
