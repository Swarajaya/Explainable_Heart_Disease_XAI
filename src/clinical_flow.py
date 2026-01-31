import os
import matplotlib.pyplot as plt
from src.config import OUTPUT_PLOT_PATH


def generate_clinical_flow_plot():
    """
    Generates a clean, publication-ready clinical decision flow diagram
    """
    os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")

    # Box positions
    boxes = {
        "Patient Data": (0.5, 0.85),
        "AI Risk Prediction": (0.5, 0.65),
        "Risk Stratification": (0.5, 0.45),
        "Clinical Action": (0.5, 0.25),
    }

    # Draw boxes
    for text, (x, y) in boxes.items():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black"),
        )

    # Draw arrows
    arrows = [
        ((0.5, 0.80), (0.5, 0.70)),
        ((0.5, 0.60), (0.5, 0.50)),
        ((0.5, 0.40), (0.5, 0.30)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

    # Decision labels
    ax.text(0.78, 0.45, "Low Risk → Lifestyle Advice", fontsize=9)
    ax.text(0.78, 0.38, "Medium Risk → Monitoring", fontsize=9)
    ax.text(0.78, 0.31, "High Risk → Cardiology Referral", fontsize=9)

    plt.title("Simulated Clinical Decision Flow", fontsize=13)

    save_path = os.path.join(OUTPUT_PLOT_PATH, "clinical_decision_flow.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Clinical decision flow generated:", save_path)
