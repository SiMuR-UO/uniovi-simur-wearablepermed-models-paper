import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🟢 Read f1_score dataframe for each model and classes")
concatenate_4_data = pd.read_csv('/home/miguel/temp/paper/1_concatenate/4_classes/concatenate_rf_metrics.csv')
concatenate_8_data = pd.read_csv('/home/miguel/temp/paper/1_concatenate/8_classes/concatenate_rf_metrics.csv')
concatenate_15_data = pd.read_csv('/home/miguel/temp/paper/1_concatenate/15_classes/concatenate_rf_metrics.csv')
stacking_rf_4_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_ae/4_classes/moe_ae_metrics.csv')
stacking_rf_8_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_ae/8_classes/moe_ae_metrics.csv')
stacking_rf_15_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_ae/15_classes/moe_ae_metrics.csv')
stacking_ae_4_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_rf/4_classes/stacking_rf_metrics.csv')
stacking_ae_8_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_rf/8_classes/stacking_rf_metrics.csv')
stacking_ae_15_data = pd.read_csv('/home/miguel/temp/paper/2_statcking_rf/15_classes/stacking_rf_metrics.csv')
moe_rf_4_data = pd.read_csv('/home/miguel/temp/paper/3_moe_rf/4_classes/moe_rf_metrics.csv')
moe_rf_8_data = pd.read_csv('/home/miguel/temp/paper/3_moe_rf/8_classes/moe_rf_metrics.csv')
moe_rf_15_data = pd.read_csv('/home/miguel/temp/paper/3_moe_rf/15_classes/moe_rf_metrics.csv')
moe_ae_4_data = pd.read_csv('/home/miguel/temp/paper/3_moe_ae/4_classes/moe_ae_metrics.csv', decimal=",")
moe_ae_8_data = pd.read_csv('/home/miguel/temp/paper/3_moe_ae/8_classes/moe_ae_metrics.csv', decimal=",")
moe_ae_15_data = pd.read_csv('/home/miguel/temp/paper/3_moe_ae/15_classes/moe_ae_metrics.csv', decimal=",")
moe_vae_4_data = pd.read_csv('/home/miguel/temp/paper/3_moe_vae/4_classes/moe_vae_metrics.csv', decimal=",")
moe_vae_8_data = pd.read_csv('/home/miguel/temp/paper/3_moe_vae/8_classes/moe_vae_metrics.csv', decimal=",")
moe_vae_15_data = pd.read_csv('/home/miguel/temp/paper/3_moe_vae/15_classes/moe_vae_metrics.csv', decimal=",")

print("🟢 Create f1_score dataframes")
df_f1_scores = pd.DataFrame({
    "f1_score_concatenated_4": concatenate_4_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_concatenated_8": concatenate_8_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_concatenated_15": concatenate_15_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_stacking_rf_4": stacking_rf_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_rf_8": stacking_rf_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_rf_15": stacking_rf_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_4": stacking_ae_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_8": stacking_ae_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_15": stacking_ae_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_moe_rf_4_data": moe_rf_4_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_rf_8_data": moe_rf_8_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_rf_15_data": moe_rf_15_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_ae_4_data": moe_ae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_ae_8_data": moe_ae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_ae_15_data": moe_ae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_4_data": moe_vae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_8_data": moe_vae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_15_data": moe_vae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
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
plt.show()

print("🟢 Save box-and-whisker plot")
plt.savefig(str(Path.cwd()) + "/images/whisker_box_plot.png", dpi=300, bbox_inches="tight")