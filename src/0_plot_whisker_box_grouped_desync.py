import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("🟢 Read f1_score dataframe for each model and classes")
stacking_rf_8_data = pd.read_csv('./paper/3_statcking_rf/8_classes/metrics.csv')
stacking_rf_8_desync_1s_data = pd.read_csv('./paper/3_statcking_rf/8_classes/metrics_desync_1s.csv')
stacking_rf_8_desync_3s_data = pd.read_csv('./paper/3_statcking_rf/8_classes/metrics_desync_3s.csv')
stacking_rf_8_desync_5s_data = pd.read_csv('./paper/3_statcking_rf/8_classes/metrics_desync_5s.csv')

print("🟢 Create f1_score dataframes grouped by classes: 4, 8, 15")
df_desync_f1_scores = pd.DataFrame({
    "stacking_rf": stacking_rf_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_rf_desync_1s": stacking_rf_8_desync_1s_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_rf_desync_3s": stacking_rf_8_desync_3s_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_rf_desync_5s": stacking_rf_8_desync_5s_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
})

print("🟢 Save f1_score dataframe metrics and mean ± std")
df_desync_f1_scores.to_csv(str(Path.cwd()) + "/results/metrics_desync_f1_scores.csv", index=False)   

df_metrics = df_desync_f1_scores.describe().loc[['mean', 'std']]
df_metrics.to_csv(str(Path.cwd()) + "/results/metrics_desync_mean_f1_scores.csv", index=False)   

print("🟢 Create box-and-whisker desync plot")
plt.figure(figsize=(8, 5))

ax_boxplot = plt.boxplot(
                df_desync_f1_scores.values,
                labels=df_desync_f1_scores.columns,
                meanline=True,
                notch=True,
                patch_artist=True
            )

plt.title("Model desync F1 Score")
plt.ylabel("F1 Score")
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.grid(axis = 'y')
plt.tight_layout()

print("🟢 Save box-and-whisker desync plot")
plt.savefig(str(Path.cwd()) + "/paper/whisker_box_plot_desync.png", dpi=300, bbox_inches="tight")
plt.show()