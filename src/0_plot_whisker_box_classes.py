import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🟢 Read f1_score dataframe for each model and classes")
concatenate_4_data = pd.read_csv('./paper/2_concatenate/4_classes/concatenate_rf_metrics.csv')
concatenate_8_data = pd.read_csv('./paper/2_concatenate/8_classes/concatenate_rf_metrics.csv')
concatenate_15_data = pd.read_csv('./paper/2_concatenate/15_classes/concatenate_rf_metrics.csv')
stacking_rf_4_data = pd.read_csv('./paper/3_statcking_ae/4_classes/moe_ae_metrics.csv')
stacking_rf_8_data = pd.read_csv('./paper/3_statcking_ae/8_classes/moe_ae_metrics.csv')
stacking_rf_15_data = pd.read_csv('./paper/3_statcking_ae/15_classes/moe_ae_metrics.csv')
stacking_ae_4_data = pd.read_csv('./paper/3_statcking_rf/4_classes/stacking_rf_metrics.csv')
stacking_ae_8_data = pd.read_csv('./paper/3_statcking_rf/8_classes/stacking_rf_metrics.csv')
stacking_ae_15_data = pd.read_csv('./paper/3_statcking_rf/15_classes/stacking_rf_metrics.csv')
moe_rf_4_data = pd.read_csv('./paper/4_moe_rf/4_classes/moe_rf_metrics.csv')
moe_rf_8_data = pd.read_csv('./paper/4_moe_rf/8_classes/moe_rf_metrics.csv')
moe_rf_15_data = pd.read_csv('./paper/4_moe_rf/15_classes/moe_rf_metrics.csv')
moe_ae_4_data = pd.read_csv('./paper/4_moe_ae/4_classes/moe_ae_metrics.csv', decimal=",")
moe_ae_8_data = pd.read_csv('./paper/4_moe_ae/8_classes/moe_ae_metrics.csv', decimal=",")
moe_ae_15_data = pd.read_csv('./paper/4_moe_ae/15_classes/moe_ae_metrics.csv', decimal=",")
moe_vae_4_data = pd.read_csv('./paper/4_moe_vae/4_classes/moe_vae_metrics.csv', decimal=",")
moe_vae_8_data = pd.read_csv('./paper/4_moe_vae/8_classes/moe_vae_metrics.csv', decimal=",")
moe_vae_15_data = pd.read_csv('./paper/4_moe_vae/15_classes/moe_vae_metrics.csv', decimal=",")

print("🟢 Create f1_score dataframes grouped by classes: 4, 8, 15")
df_4 = pd.DataFrame({
    "f1_score_concatenated_4": concatenate_4_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_stacking_rf_4": stacking_rf_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_4": stacking_ae_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_moe_rf_4_data": moe_rf_4_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_ae_4_data": moe_ae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_4_data": moe_vae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

df_8 = pd.DataFrame({
    "f1_score_concatenated_8": concatenate_8_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_stacking_rf_8": stacking_rf_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_8": stacking_ae_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_moe_rf_8_data": moe_rf_8_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_ae_8_data": moe_ae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_8_data": moe_vae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

df_15 = pd.DataFrame({
    "f1_score_concatenated_15": concatenate_15_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "f1_score_stacking_rf_15": stacking_rf_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_stacking_ae_15": stacking_ae_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "f1_score_moe_rf_15_data": moe_rf_15_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "f1_score_moe_ae_15_data": moe_ae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
    "f1_score_moe_vae_15_data": moe_vae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

print("🟢 Create box-and-whisker plot")
fig, (ax_4, ax_8, ax_15) = plt.subplots(3, 1, figsize=(8, 10))

ax_4.boxplot(
    df_4.values,
    labels=df_4.columns,
    showmeans=True
)
ax_4.set_title('Model F1 Score Comparison for 4 Classes')
ax_4.set_ylabel('F1 Score')
ax_4.grid(axis='y')
ax_4.tick_params(axis='x', labelrotation=45)

ax_8.boxplot(
    df_8.values,
    labels=df_8.columns,
    showmeans=True
)

ax_8.set_title('Model F1 Score Comparison for 8 Classes')
ax_8.set_ylabel('F1 Score')
ax_8.grid(axis='y')
ax_8.tick_params(axis='x', labelrotation=45)

ax_15.boxplot(
    df_15.values,
    labels=df_15.columns,
    showmeans=True
)

ax_15.set_title('Model F1 Score Comparison for 25 Classes')
ax_15.set_ylabel('F1 Score')
ax_15.grid(axis='y')
ax_15.tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()

print("🟢 Save box-and-whisker plot")
plt.savefig(str(Path.cwd()) + "/images/whisker_box_plot_classes.png", dpi=300, bbox_inches="tight")