Explainable AI for Early Heart Disease Risk Prediction

Temporal • Interpretable • Fair • Clinically Grounded

📌 Overview

This project presents a comprehensive Explainable Artificial Intelligence (XAI) framework for early heart disease risk prediction.
Unlike traditional black-box machine learning models, this system emphasizes interpretability, temporal risk evolution, fairness, uncertainty awareness, and clinical decision support.

The framework is designed to bridge the gap between high predictive performance and real-world clinical usability, making it suitable for academic research, clinical decision-support prototyping, and PhD-level experimentation.

🎯 Key Objectives

Predict early heart disease risk using structured clinical data

Model temporal risk progression over time

Provide global, local, and counterfactual explanations

Evaluate fairness and bias across demographic subgroups

Quantify uncertainty and confidence intervals

Simulate clinical decision workflows and healthcare policies

Ensure ethical and responsible AI deployment

🧠 Core Contributions

✅ Unified framework combining prediction + explainability + ethics

✅ Temporal modeling of cardiovascular risk

✅ Multi-level explainability (global, local, temporal, counterfactual)

✅ Cost-sensitive and decision-curve–based clinical evaluation

✅ Human-in-the-loop and rejection-based safety mechanisms

✅ Policy and causal proxy analysis for real-world deployment

📂 Project Structure
.
├── README.md
├── run_pipeline.py
├── requirements.txt
├── data/
│   └── heart_disease.csv
├── outputs/
│   ├── models/
│   └── plots/
│       ├── *.png
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── clinical_flow.py
│   ├── clinical_risk.py
│   ├── temporal_risk.py
│   ├── fairness_analysis.py
│   ├── counterfactual_analysis.py
│   ├── policy_simulation.py
│   ├── ethical_risk.py
│   ├── causal_proxy.py
│   ├── bootstrap_performance.py
│   ├── ablation_study.py
│   ├── confidence_intervals.py
│   ├── explanation_stability.py
│   ├── explanation_temporal.py
│   └── ...

📊 Generated Outputs

The pipeline generates 35+ publication-ready figures, including:

Predictive performance (ROC, calibration, confidence intervals)

Temporal risk evolution and early warning signals

Global & local explainability (SHAP, feature importance)

Counterfactual explanations

Fairness and subgroup analysis

Cost-sensitive learning effects

Decision curve analysis

Human-AI collaboration analysis

Ethical risk mapping and rejection analysis

Clinical decision flow and policy simulations

All outputs are saved in:

outputs/plots/

⚙️ Installation
1️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Run the Full Pipeline

From the project root directory:

python run_pipeline.py


This will:

Train the predictive model

Evaluate performance metrics

Generate all explainability, fairness, temporal, and policy plots

Save all outputs automatically

You should see console logs confirming successful execution of each module.

📐 Methodology Summary
🔹 Problem Formulation

Binary classification with temporal extensions

Objective: risk prediction + interpretability

🔹 Modeling

Supervised ML for structured clinical data

Temporal risk modeling across simulated time steps

🔹 Explainability

Global explanations (feature importance, SHAP)

Local explanations (patient-level attribution)

Counterfactual reasoning

Temporal explanation consistency

🔹 Evaluation

Accuracy, ROC-AUC, calibration

Decision Curve Analysis (clinical benefit)

Bootstrap stability and ablation studies

🔹 Ethics & Safety

Fairness analysis across age and gender

Uncertainty quantification and rejection analysis

Ethical risk visualization

🧪 Reproducibility

All figures are generated from a single trained model

Deterministic pipeline execution

Modular design for easy extension

Suitable for conference, journal, and PhD evaluation

🏥 Clinical Relevance

The framework is designed to reflect real clinical workflows, including:

Risk stratification

Human oversight under uncertainty

Policy-level decision simulations

Ethical and regulatory considerations

This makes it well-suited for clinical decision-support research and translational AI studies.

🚀 Future Extensions

Integration with real Electronic Health Records (EHRs)

Multimodal data (imaging, wearables, clinical notes)

Prospective clinical validation

Deployment as a decision-support dashboard

📄 Research Paper

A full research paper has been written alongside this project, including:

Structured Introduction with citations

Literature Review table

Extensive Results & Explainability analysis

Ethical, causal, and policy discussions

🙏 Acknowledgements

This work builds upon open-source machine learning libraries and publicly available clinical datasets. The authors acknowledge the broader research community for advancing reproducible and ethical AI in healthcare.

📬 Contact

For academic collaboration, questions, or extensions:

Author: Swarajaya Singh Sawant
Email: swarajayasawant19@gmail.com