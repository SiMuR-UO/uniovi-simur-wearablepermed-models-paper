import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

METRIC_PI_FILENAME = 'metrics_loocv_pi_all.csv'
METRIC_M_FILENAM = 'metrics_loocv_m_all.csv'
METRIC_C_FILENAM = 'metrics_loocv_c_all.csv'
METRIC_FILENAME = 'metrics_pi_m_c_loocv_all.csv'

print("🟢 Read f1_score dataframe for each model and classes")
individual_4_pi_data = pd.read_csv('./paper/1_individual/4_classes/' + METRIC_PI_FILENAME)
individual_4_m_data = pd.read_csv('./paper/1_individual/4_classes/' + METRIC_M_FILENAM)
individual_4_c_data = pd.read_csv('./paper/1_individual/4_classes/' + METRIC_C_FILENAM)
individual_8_pi_data = pd.read_csv('./paper/1_individual/8_classes/' + METRIC_PI_FILENAME)
individual_8_m_data = pd.read_csv('./paper/1_individual/8_classes/' + METRIC_M_FILENAM)
individual_8_c_data = pd.read_csv('./paper/1_individual/8_classes/' + METRIC_C_FILENAM)
individual_15_pi_data = pd.read_csv('./paper/1_individual/15_classes/' + METRIC_PI_FILENAME)
individual_15_m_data = pd.read_csv('./paper/1_individual/15_classes/' + METRIC_M_FILENAM)
individual_15_c_data = pd.read_csv('./paper/1_individual/15_classes/' + METRIC_C_FILENAM)
concatenate_4_data = pd.read_csv('./paper/2_concatenate/4_classes/' + METRIC_FILENAME)
concatenate_8_data = pd.read_csv('./paper/2_concatenate/8_classes/' + METRIC_FILENAME)
concatenate_15_data = pd.read_csv('./paper/2_concatenate/15_classes/' + METRIC_FILENAME)
stacking_rf_4_data = pd.read_csv('./paper/3_statcking_rf/4_classes/' + METRIC_FILENAME)
stacking_rf_8_data = pd.read_csv('./paper/3_statcking_rf/8_classes/' + METRIC_FILENAME)
stacking_rf_15_data = pd.read_csv('./paper/3_statcking_rf/15_classes/' + METRIC_FILENAME)
stacking_ae_4_data = pd.read_csv('./paper/3_statcking_ae/4_classes/' + METRIC_FILENAME)
stacking_ae_8_data = pd.read_csv('./paper/3_statcking_ae/8_classes/' + METRIC_FILENAME)
stacking_ae_15_data = pd.read_csv('./paper/3_statcking_ae/15_classes/' + METRIC_FILENAME)
moe_rf_4_data = pd.read_csv('./paper/4_moe_rf/4_classes/' + METRIC_FILENAME)
moe_rf_8_data = pd.read_csv('./paper/4_moe_rf/8_classes/' + METRIC_FILENAME)
moe_rf_15_data = pd.read_csv('./paper/4_moe_rf/15_classes/' + METRIC_FILENAME)
moe_ae_4_data = pd.read_csv('./paper/4_moe_ae/4_classes/' + METRIC_FILENAME, decimal=",")
moe_ae_8_data = pd.read_csv('./paper/4_moe_ae/8_classes/' + METRIC_FILENAME, decimal=",")
moe_ae_15_data = pd.read_csv('./paper/4_moe_ae/15_classes/' + METRIC_FILENAME, decimal=",")

print("🟢 Create f1_score dataframes grouped by classes: 4, 8, 15")
df_4 = pd.DataFrame({
    "individual_pi": individual_4_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "individual_m": individual_4_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "individual_c": individual_4_c_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "concatenated": concatenate_4_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "stacking_rf": stacking_rf_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_ae": stacking_ae_4_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "moe_rf": moe_rf_4_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "moe_ae": moe_ae_4_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

df_8 = pd.DataFrame({
    "individual_pi": individual_8_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "individual_m": individual_8_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "individual_c": individual_8_c_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "concatenated": concatenate_8_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "stacking_rf": stacking_rf_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_ae": stacking_ae_8_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "moe_rf": moe_rf_8_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "moe_ae": moe_ae_8_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

df_15 = pd.DataFrame({
    "individual_pi": individual_15_pi_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "individual_m": individual_15_m_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "individual_c": individual_15_c_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),    
    "concatenated": concatenate_15_data.loc[:29,"model_f1_score_test"].to_numpy().astype(float),
    "stacking_rf": stacking_rf_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "stacking_ae": stacking_ae_15_data.loc[:29,"meta_model_f1_score"].to_numpy().astype(float),
    "moe_rf": moe_rf_15_data.loc[:29,"moe_f1_weight_soft"].to_numpy().astype(float),
    "moe_ae": moe_ae_15_data.loc[:29,"moe_model_test_hard_accuracy"].to_numpy().astype(float),
})

print("🟢 Create box-and-whisker plot")
fig, (ax_4, ax_8, ax_15) = plt.subplots(3, 1, figsize=(8, 15), sharey=True)
fig.suptitle('Fusion Strategies for Thigh, Wrist, Hip', fontsize=16, fontweight='bold', y=1)
             
ax_4_boxplot = ax_4.boxplot(
                    df_4.values,
                    labels=df_4.columns,
                    meanline=True,
                    notch=True,
                    patch_artist=True
                )
ax_4.set_title('Model F1 Score for 4 Classes')
ax_4.set_ylabel('F1 Score')
ax_4.grid(axis='y')
ax_4.tick_params(axis='x', labelrotation=45)

# Add numerical medium value
for i, median in enumerate(ax_4_boxplot['medians']):
    # Get the x and y coordinates of the median line
    xdata = median.get_xdata()
    ydata = median.get_ydata()
        
    # The x-position is the center of the line; the y-position is the value itself
    x_pos = sum(xdata) / 2
    median_val = ydata[0]
        
    # Add the text label right above the median line
    ax_4.text(
        x_pos, 
        median_val + 0.01,          # Shift slightly upward so it doesn't overlap the line
        f"{median_val:.2f}",        # Round to 2 decimal places
        ha='center',                # Center horizontally
        va='bottom',                # Align bottom of text
        fontsize=9,                 # Keep it neat and readable
        fontweight='bold',
        color='black'               # Change color if needed (e.g., 'white' if inside the box)
    )

for label in ax_4.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

ax_8_boxplot = ax_8.boxplot(
                    df_8.values,
                    labels=df_8.columns,                    
                    meanline=True,
                    notch=True,
                    patch_artist=True
                )


# Add numerical medium value
for i, median in enumerate(ax_8_boxplot['medians']):
    # Get the x and y coordinates of the median line
    xdata = median.get_xdata()
    ydata = median.get_ydata()
        
    # The x-position is the center of the line; the y-position is the value itself
    x_pos = sum(xdata) / 2
    median_val = ydata[0]
        
    # Add the text label right above the median line
    ax_8.text(
        x_pos, 
        median_val + 0.01,          # Shift slightly upward so it doesn't overlap the line
        f"{median_val:.2f}",        # Round to 2 decimal places
        ha='center',                # Center horizontally
        va='bottom',                # Align bottom of text
        fontsize=9,                 # Keep it neat and readable
        fontweight='bold',
        color='black'               # Change color if needed (e.g., 'white' if inside the box)
    )

ax_8.set_title('Model F1 Score for 8 Classes')
ax_8.set_ylabel('F1 Score')
ax_8.grid(axis='y')
ax_8.tick_params(axis='x', labelrotation=45)

for label in ax_8.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

ax_15_boxplot = ax_15.boxplot(
                    df_15.values,
                    labels=df_15.columns,
                    meanline=True,
                    notch=True,
                    patch_artist=True
                )

# Add numerical medium value
for i, median in enumerate(ax_15_boxplot['medians']):
    # Get the x and y coordinates of the median line
    xdata = median.get_xdata()
    ydata = median.get_ydata()
        
    # The x-position is the center of the line; the y-position is the value itself
    x_pos = sum(xdata) / 2
    median_val = ydata[0]
        
    # Add the text label right above the median line
    ax_15.text(
        x_pos, 
        median_val + 0.01,          # Shift slightly upward so it doesn't overlap the line
        f"{median_val:.2f}",        # Round to 2 decimal places
        ha='center',                # Center horizontally
        va='bottom',                # Align bottom of text
        fontsize=9,                 # Keep it neat and readable
        fontweight='bold',
        color='black'               # Change color if needed (e.g., 'white' if inside the box)
    )

ax_15.set_title('Model F1 Score for 15 Classes')
ax_15.set_ylabel('F1 Score')
ax_15.grid(axis='y')
ax_15.tick_params(axis='x', labelrotation=45)

for label in ax_15.get_xticklabels():
    label.set_horizontalalignment('right')
    label.set_rotation_mode('anchor')

# fill with colors
colors = ['#e9162d', '#f28200', '#ffdb28', '#1fb819', '#00e1da', '#007bd8', '#8f2be7', '#fb4fd9', "#d86f80"]
for patch, color in zip(ax_4_boxplot['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(ax_8_boxplot['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(ax_15_boxplot['boxes'], colors):
    patch.set_facecolor(color)
    
plt.tight_layout()

print("🟢 Save box-and-whisker plot")
plt.savefig(str(Path.cwd()) + "/paper/whisker_box_plot_classes_pi_m_c_loocv_all.png", dpi=300, bbox_inches="tight")
plt.show()