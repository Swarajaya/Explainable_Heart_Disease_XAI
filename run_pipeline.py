# ===============================
# EXPLAINABLE HEART DISEASE XAI
# FINAL TIER-1++ PIPELINE
# ===============================

print("Training models...")
import src.train

print("Generating core evaluation & explainability...")
from src.evaluate import evaluate_and_plot
from src.explainability import generate_shap_plot

# ===============================
# CORE CLINICAL & MODEL PLOTS
# ===============================
from src.clinical_risk import generate_clinical_risk_plot
from src.temporal_risk import generate_risk_over_time
from src.local_explanation import generate_local_explanation
from src.model_agreement import generate_model_agreement_plot

# ===============================
# HUMAN-AI & DECISION ANALYSIS
# ===============================
from src.human_in_loop import generate_human_ai_plot
from src.cost_sensitive import generate_cost_sensitive_plot
from src.decision_curve import generate_decision_curve
from src.physiology_check import generate_physiology_plot
from src.distribution_shift import generate_distribution_shift_plot
from src.bootstrap_performance import generate_bootstrap_plot
from src.pareto_front import generate_pareto_plot
from src.ethical_risk import generate_ethical_risk_plot
from src.explanation_temporal import generate_temporal_explanation_plot
from src.policy_simulation import generate_policy_simulation_plot

# ===============================
# THRESHOLD, FAIRNESS & STABILITY
# ===============================
from src.threshold_analysis import generate_threshold_analysis
from src.explanation_stability import generate_explanation_stability_plot
from src.fairness_analysis import generate_fairness_plots
from src.ablation_study import generate_ablation_plot
from src.noise_sensitivity import generate_noise_sensitivity_plot
from src.clinical_flow import generate_clinical_flow_plot
from src.calibration_curve import generate_calibration_plot
from src.rejection_analysis import generate_rejection_plot

# ===============================
# ADVANCED / RARE TIER-0 OUTPUTS
# ===============================
from src.counterfactual_analysis import generate_counterfactual_plot
from src.feature_interactions import generate_feature_interaction_plot
from src.subgroup_performance import generate_subgroup_performance_plot
from src.early_warning import generate_early_warning_plot
from src.causal_proxy import generate_causal_proxy_plot
from src.confidence_intervals import generate_confidence_interval_plot
from src.model_distillation import generate_distillation_plot
from src.failure_analysis import generate_failure_analysis_plot

# ===============================
# EXECUTION ORDER
# ===============================

evaluate_and_plot()
generate_shap_plot()

generate_clinical_risk_plot()
generate_risk_over_time()
generate_local_explanation()
generate_model_agreement_plot()

generate_human_ai_plot()
generate_cost_sensitive_plot()
generate_decision_curve()
generate_physiology_plot()
generate_distribution_shift_plot()
generate_bootstrap_plot()
generate_pareto_plot()
generate_ethical_risk_plot()
generate_temporal_explanation_plot()
generate_policy_simulation_plot()

print("UPGRADES COMPLETE")

generate_threshold_analysis()
generate_explanation_stability_plot()
generate_fairness_plots()
generate_ablation_plot()
generate_noise_sensitivity_plot()
generate_clinical_flow_plot()
generate_calibration_plot()
generate_rejection_plot()

generate_counterfactual_plot()
generate_feature_interaction_plot()
generate_subgroup_performance_plot()
generate_early_warning_plot()
generate_causal_proxy_plot()
generate_confidence_interval_plot()
generate_distillation_plot()
generate_failure_analysis_plot()

print("ALL ")
print("PIPELINE COMPLETE. ALL OUTPUTS SAVED.")
