import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🟢 Read f1_score dataframe for each model and classes")
individual_4_pi_data = pd.read_csv('./paper/1_individual/4_classes/metrics_pi.csv')
individual_4_m_data = pd.read_csv('./paper/1_individual/4_classes/metrics_m.csv')
individual_8_pi_data = pd.read_csv('./paper/1_individual/8_classes/metrics_pi.csv')
individual_8_m_data = pd.read_csv('./paper/1_individual/8_classes/metrics_m.csv')
individual_15_pi_data = pd.read_csv('./paper/1_individual/15_classes/metrics_pi.csv')
individual_15_m_data = pd.read_csv('./paper/1_individual/15_classes/metrics_m.csv')
concatenate_4_data = pd.read_csv('./paper/2_concatenate/4_classes/metrics.csv')
concatenate_8_data = pd.read_csv('./paper/2_concatenate/8_classes/metrics.csv')
concatenate_15_data = pd.read_csv('./paper/2_concatenate/15_classes/metrics.csv')
stacking_rf_4_data = pd.read_csv('./paper/3_statcking_ae/4_classes/metrics.csv')
stacking_rf_8_data = pd.read_csv('./paper/3_statcking_ae/8_classes/metrics.csv')
stacking_rf_15_data = pd.read_csv('./paper/3_statcking_ae/15_classes/metrics.csv')
stacking_ae_4_data = pd.read_csv('./paper/3_statcking_rf/4_classes/metrics.csv')
stacking_ae_8_data = pd.read_csv('./paper/3_statcking_rf/8_classes/metrics.csv')
stacking_ae_15_data = pd.read_csv('./paper/3_statcking_rf/15_classes/metrics.csv')
moe_rf_4_data = pd.read_csv('./paper/4_moe_rf/4_classes/metrics.csv')
moe_rf_8_data = pd.read_csv('./paper/4_moe_rf/8_classes/metrics.csv')
moe_rf_15_data = pd.read_csv('./paper/4_moe_rf/15_classes/metrics.csv')
moe_ae_4_data = pd.read_csv('./paper/4_moe_ae/4_classes/metrics.csv', decimal=",")
moe_ae_8_data = pd.read_csv('./paper/4_moe_ae/8_classes/metrics.csv', decimal=",")
moe_ae_15_data = pd.read_csv('./paper/4_moe_ae/15_classes/metrics.csv', decimal=",")
moe_vae_4_data = pd.read_csv('./paper/4_moe_vae/4_classes/metrics.csv', decimal=",")
moe_vae_8_data = pd.read_csv('./paper/4_moe_vae/8_classes/metrics.csv', decimal=",")
moe_vae_15_data = pd.read_csv('./paper/4_moe_vae/15_classes/metrics.csv', decimal=",")

print("🟢 Create f1_score dataframes")
df_f1_scores = pd.DataFrame({
    "f1_score_individual_pi_4": individual_4_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_individual_m_4": individual_4_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_individual_pi_8": individual_8_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_individual_m_8": individual_8_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_individual_pi_15": individual_15_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_individual_m_15": individual_15_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_concatenated_4": concatenate_4_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_concatenated_8": concatenate_8_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_concatenated_15": concatenate_15_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_stacking_rf_4": stacking_rf_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_rf_8": stacking_rf_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_rf_15": stacking_rf_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_4": stacking_ae_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_8": stacking_ae_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_15": stacking_ae_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_moe_rf_4": moe_rf_4_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_rf_8": moe_rf_8_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_rf_15": moe_rf_15_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_ae_4": moe_ae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_ae_8": moe_ae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_ae_15": moe_ae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_4": moe_vae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_8": moe_vae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_15": moe_vae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

print("🟢 Save f1_score dataframe metrics")
df_f1_scores.to_csv(str(Path.cwd()) + "/results/metrics_f1_scores.csv", index=False)   

print("🟢 Create box-and-whisker plot")
plt.figure(figsize=(8, 5))

plt.boxplot(
    df_f1_scores.values,
    labels=df_f1_scores.columns,
    showmeans=True  # optional: show mean marker
)

plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.grid(axis = 'y')
plt.tight_layout()

print("🟢 Save box-and-whisker plot")
plt.savefig(str(Path.cwd()) + "/paper/whisker_box_plot.png", dpi=300, bbox_inches="tight")
plt.show()