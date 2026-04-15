import numpy as np
import pandas as pd

METRIC_FILENAME = 'metrics_loocv.csv'

print("🟢 Read f1_score dataframe for each model and classes")
individual_4_pi_data = pd.read_csv('./paper/1_individual/4_classes/metrics_pi.csv')
individual_4_m_data = pd.read_csv('./paper/1_individual/4_classes/metrics_m.csv')
individual_8_pi_data = pd.read_csv('./paper/1_individual/8_classes/metrics_pi.csv')
individual_8_m_data = pd.read_csv('./paper/1_individual/8_classes/metrics_m.csv')
individual_15_pi_data = pd.read_csv('./paper/1_individual/15_classes/metrics_pi.csv')
individual_15_m_data = pd.read_csv('./paper/1_individual/15_classes/metrics_m.csv')
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

print("🟢 Filter and rename dataframe and get only the mean and standard deviation")
individual_4_pi_data = individual_4_pi_data[individual_4_pi_data["loop"].isin(["mean", "std"])]
individual_8_pi_data = individual_8_pi_data[individual_8_pi_data["loop"].isin(["mean", "std"])]
individual_15_pi_data = individual_15_pi_data[individual_15_pi_data["loop"].isin(["mean", "std"])]
individual_4_m_data = individual_4_m_data[individual_4_m_data["loop"].isin(["mean", "std"])]
individual_8_m_data = individual_8_m_data[individual_8_m_data["loop"].isin(["mean", "std"])]
individual_15_m_data = individual_15_m_data[individual_15_m_data["loop"].isin(["mean", "std"])]
concatenate_4_data = concatenate_4_data[concatenate_4_data["loop"].isin(["mean", "std"])]
concatenate_8_data = concatenate_8_data[concatenate_8_data["loop"].isin(["mean", "std"])]
concatenate_15_data = concatenate_15_data[concatenate_15_data["loop"].isin(["mean", "std"])]
stacking_rf_4_data = stacking_rf_4_data[stacking_rf_4_data["loop"].isin(["mean", "std"])]
stacking_rf_4_data = stacking_rf_4_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_rf_4_data = stacking_rf_4_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
stacking_rf_8_data = stacking_rf_8_data[stacking_rf_8_data["loop"].isin(["mean", "std"])]
stacking_rf_8_data = stacking_rf_8_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_rf_8_data = stacking_rf_8_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
stacking_rf_15_data = stacking_rf_15_data[stacking_rf_15_data["loop"].isin(["mean", "std"])]
stacking_rf_15_data = stacking_rf_15_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_rf_15_data = stacking_rf_15_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
stacking_ae_4_data = stacking_ae_4_data[stacking_ae_4_data["loop"].isin(["mean", "std"])]
stacking_ae_4_data = stacking_ae_4_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_ae_4_data = stacking_ae_4_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
stacking_ae_8_data = stacking_ae_8_data[stacking_ae_8_data["loop"].isin(["mean", "std"])]
stacking_ae_8_data = stacking_ae_8_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_ae_8_data = stacking_ae_8_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
stacking_ae_15_data = stacking_ae_15_data[stacking_ae_15_data["loop"].isin(["mean", "std"])]
stacking_ae_15_data = stacking_ae_15_data.loc[:, ["loop", "meta_model_accuracy", "meta_model_f1_score"]]
stacking_ae_15_data = stacking_ae_15_data.rename(columns={"meta_model_accuracy": "model_accuracy_test", "meta_model_f1_score": "model_f1_score_test"})
moe_rf_4_data = moe_rf_4_data[moe_rf_4_data["loop"].isin(["mean", "std"])]
moe_rf_4_data = moe_rf_4_data.loc[:, ["loop", "moe_acc_soft", "moe_f1_weight_soft"]]
moe_rf_4_data = moe_rf_4_data.rename(columns={"model_accuracy_test": "model_accuracy_test", "moe_f1_weight_soft": "model_f1_score_test"})
moe_rf_8_data = moe_rf_8_data[moe_rf_8_data["loop"].isin(["mean", "std"])]
moe_rf_8_data = moe_rf_8_data.loc[:, ["loop", "moe_acc_soft", "moe_f1_weight_soft"]]
moe_rf_8_data = moe_rf_8_data.rename(columns={"model_accuracy_test": "model_accuracy_test", "moe_f1_weight_soft": "model_f1_score_test"})
moe_rf_15_data = moe_rf_15_data[moe_rf_15_data["loop"].isin(["mean", "std"])]
moe_rf_15_data = moe_rf_15_data.loc[:, ["loop", "moe_acc_soft", "moe_f1_weight_soft"]]
moe_rf_15_data = moe_rf_15_data.rename(columns={"model_accuracy_test": "model_accuracy_test", "moe_f1_weight_soft": "model_f1_score_test"})
moe_ae_4_data = moe_ae_4_data[moe_ae_4_data["loop"].isin(["mean", "std"])]
moe_ae_4_data = moe_ae_4_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_ae_4_data = moe_ae_4_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})
moe_ae_8_data = moe_ae_8_data[moe_ae_8_data["loop"].isin(["mean", "std"])]
moe_ae_8_data = moe_ae_8_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_ae_8_data = moe_ae_8_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})
moe_ae_15_data = moe_ae_15_data[moe_ae_15_data["loop"].isin(["mean", "std"])]
moe_ae_15_data = moe_ae_15_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_ae_15_data = moe_ae_15_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})

print("🟢 Transform dataframe to be concatenated")
individual_4_pi_data = (
    individual_4_pi_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_4_pi_data.insert(1, "granularity", 4)
individual_4_pi_data.insert(2, "fusion_strategy", "individual_pi")
individual_8_pi_data = (
    individual_8_pi_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_8_pi_data.insert(1, "granularity", 8)
individual_8_pi_data.insert(2, "fusion_strategy", "individual_pi")
individual_15_pi_data = (
    individual_15_pi_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_15_pi_data.insert(1, "granularity", 15)
individual_15_pi_data.insert(2, "fusion_strategy", "individual_pi")
individual_4_m_data = (
    individual_4_m_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_4_m_data.insert(1, "granularity", 4)
individual_4_m_data.insert(2, "fusion_strategy", "individual_m")
individual_8_m_data = (
    individual_8_m_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_8_m_data.insert(1, "granularity", 8)
individual_8_m_data.insert(2, "fusion_strategy", "individual_m")
individual_15_m_data = (
    individual_15_m_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
individual_15_m_data.insert(1, "granularity", 15)
individual_15_m_data.insert(2, "fusion_strategy", "individual_m")
concatenate_4_data = (
    concatenate_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
concatenate_4_data.insert(1, "granularity", 4)
concatenate_4_data.insert(2, "fusion_strategy", "concatenated")
concatenate_8_data = (
    concatenate_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
concatenate_8_data.insert(1, "granularity", 8)
concatenate_8_data.insert(2, "fusion_strategy", "concatenated")
concatenate_15_data = (
    concatenate_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
concatenate_15_data.insert(1, "granularity", 15)
concatenate_15_data.insert(2, "fusion_strategy", "concatenated")
stacking_rf_4_data = (
    stacking_rf_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_rf_4_data.insert(1, "granularity", 4)
stacking_rf_4_data.insert(2, "fusion_strategy", "stacking_rf")
stacking_rf_8_data = (
    stacking_rf_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_rf_8_data.insert(1, "granularity", 8)
stacking_rf_8_data.insert(2, "fusion_strategy", "stacking_rf")
stacking_rf_15_data = (
    stacking_rf_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_rf_15_data.insert(1, "granularity", 15)
stacking_rf_15_data.insert(2, "fusion_strategy", "stacking_rf")
stacking_ae_4_data = (
    stacking_ae_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_ae_4_data.insert(1, "granularity", 4)
stacking_ae_4_data.insert(2, "fusion_strategy", "stacking_ae")
stacking_ae_8_data = (
    stacking_ae_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_ae_8_data.insert(1, "granularity", 8)
stacking_ae_8_data.insert(2, "fusion_strategy", "stacking_ae")
stacking_ae_15_data = (
    stacking_ae_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
stacking_ae_15_data.insert(1, "granularity", 15)
stacking_ae_15_data.insert(2, "fusion_strategy", "stacking_ae")
moe_rf_4_data = (
    moe_rf_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_rf_4_data.insert(1, "granularity", 4)
moe_rf_4_data.insert(2, "fusion_strategy", "moe_rf")
moe_rf_8_data = (
    moe_rf_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_rf_8_data.insert(1, "granularity", 8)
moe_rf_8_data.insert(2, "fusion_strategy", "moe_rf")
moe_rf_15_data = (
    moe_rf_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_rf_15_data.insert(1, "granularity", 15)
moe_rf_15_data.insert(2, "fusion_strategy", "moe_rf")
moe_ae_4_data = (
    moe_ae_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_ae_4_data.insert(1, "granularity", 4)
moe_ae_4_data.insert(2, "fusion_strategy", "moe_ae")
moe_ae_8_data = (
    moe_ae_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_ae_8_data.insert(1, "granularity", 8)
moe_ae_8_data.insert(2, "fusion_strategy", "moe_ae")
moe_ae_15_data = (
    moe_ae_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_ae_15_data.insert(1, "granularity", 15)
moe_ae_15_data.insert(2, "fusion_strategy", "moe_ae")

print("🟢 Concatenated dataframe")
table_whisker_box = pd.concat([
    individual_4_pi_data,
    individual_8_pi_data,
    individual_15_pi_data,    
    individual_4_m_data,
    individual_8_m_data,    
    individual_15_m_data,
    concatenate_4_data,
    concatenate_8_data,
    concatenate_15_data,
    stacking_rf_4_data,
    stacking_rf_8_data,
    stacking_rf_15_data,
    stacking_ae_4_data,
    stacking_ae_8_data,
    stacking_ae_15_data,
    moe_rf_4_data,
    moe_rf_8_data,
    moe_rf_15_data,
    moe_ae_4_data,
    moe_ae_8_data,
    moe_ae_15_data,
], axis=0).reset_index(drop=True)

# Convert mean and std to percent and round
table_whisker_box["mean"] = table_whisker_box["mean"].astype(float)
table_whisker_box["std"] = table_whisker_box["std"].astype(float)
table_whisker_box["f1_score_percent"] = table_whisker_box.apply(
    lambda row: f"{round(row['mean']*100, 2):.2f} ± {round(row['std']*100, 2):.2f}", axis=1
)
table_whisker_box = table_whisker_box.drop(columns=["mean", "std"])

table_whisker_box = table_whisker_box.query('metric == "model_f1_score_test"')

print("🟢 Table Metric Models")
print(table_whisker_box)

print("\n")

print("🟢 F1_Score Ranking")
table_whisker_box_compare = table_whisker_box
table_whisker_box_compare['f1_mean'] = table_whisker_box_compare['f1_score_percent'].apply(lambda x: float(x.split(' ± ')[0]))

ranked_models = table_whisker_box_compare.sort_values(by='f1_mean', ascending=False)

print("--- Top 5 Performing Models ---")
print(ranked_models[['fusion_strategy', 'granularity', 'f1_score_percent']].head(5))

best = ranked_models.iloc[0]
print(f"\nBEST MODEL: {best['fusion_strategy']} (Granularity: {best['granularity']}) with F1: {best['f1_score_percent']}%")
print("\n")

print("🟢 Weighted Ranking: Sharpe-like ratio like: Coefficient of Variation (CV) with Stability Score")
table_whisker_box_compare = table_whisker_box

def extract_stats(val):
    parts = val.split(' ± ')
    return float(parts[0]), float(parts[1])

table_whisker_box_compare[['mean', 'std']] = table_whisker_box_compare['f1_score_percent'].apply(lambda x: pd.Series(extract_stats(x)))

# 3. Calculate Sharpe-like ratio: Coefficient of Variation (CV) - (Lower is more consistent)
table_whisker_box_compare['cv_percent'] = (table_whisker_box_compare['std'] / table_whisker_box_compare['mean']) * 100

# 4. Calculate Stability Score (Mean - Std) - (Higher is better, represents the "safe" performance floor)
table_whisker_box_compare['stability_score'] = table_whisker_box_compare['mean'] - table_whisker_box_compare['std']

# 1. Normalize the metrics so they are on the same scale (0 to 1)
# For Stability: Higher is better
table_whisker_box_compare['norm_stability'] = (table_whisker_box_compare['stability_score'] - table_whisker_box_compare['stability_score'].min()) / (table_whisker_box_compare['stability_score'].max() - table_whisker_box_compare['stability_score'].min())

# For CV: Lower is better (so we subtract from 1)
table_whisker_box_compare['norm_consistency'] = 1 - (table_whisker_box_compare['cv_percent'] - table_whisker_box_compare['cv_percent'].min()) / (table_whisker_box_compare['cv_percent'].max() - table_whisker_box_compare['cv_percent'].min())

# 2. Create a Final Score (e.g., 70% weight on stability, 30% on consistency)
table_whisker_box_compare['final_rank_score'] = (table_whisker_box_compare['norm_stability'] * 0.7) + (table_whisker_box_compare['norm_consistency'] * 0.3)

# 3. Now the ranking uses EVERYTHING you calculated
table_whisker_box_compare = table_whisker_box_compare.sort_values(by='final_rank_score', ascending=False)

print("--- Top 5 Performing Models ---")
print(ranked_models[['fusion_strategy', 'granularity', 'f1_score_percent']].head(5))

best = ranked_models.iloc[0]
print(f"\nBEST MODEL: {best['fusion_strategy']} (Granularity: {best['granularity']}) with F1: {best['f1_score_percent']}%")
print("\n")

print("🟢 Z-Test Ranking")
table_whisker_box_compare[['mean', 'std']] = table_whisker_box_compare['f1_score_percent'].apply(lambda x: pd.Series([float(i) for i in x.split(' ± ')]))

# 2. Identify the "Champion" (Highest Mean)
table_whisker_box_compare = table_whisker_box

# 2. Extract numeric Mean and Std from the string column
table_whisker_box_compare['mean'] = table_whisker_box_compare['f1_score_percent'].apply(lambda x: float(x.split(' ± ')[0]))
table_whisker_box_compare['std'] = table_whisker_box_compare['f1_score_percent'].apply(lambda x: float(x.split(' ± ')[1]))

# 3. Sort by mean to find the Champion
table_whisker_box_compare = table_whisker_box_compare.sort_values('mean', ascending=False).reset_index(drop=True)
champion = table_whisker_box_compare.iloc[0]

# 4. Refactored Z-test Function (Returns ONLY a float)
def calculate_z_score(row, champ):
    if row.name == 0:  # This is the champion itself
        return 0.0
    
    # Formula: (Mean1 - Mean2) / sqrt(std1^2 + std2^2)
    diff = champ['mean'] - row['mean']
    pooled_std = np.sqrt(champ['std']**2 + row['std']**2)
    
    return float(diff / pooled_std) if pooled_std != 0 else 0.0

# 5. Apply the function to create the new column
table_whisker_box_compare['z_score_vs_champ'] = table_whisker_box_compare.apply(
    lambda row: calculate_z_score(row, champion), axis=1
)

# 6. Now the comparison works perfectly
table_whisker_box_compare['is_significantly_worse'] = table_whisker_box_compare['z_score_vs_champ'] > 1.96

# Display results
#print(table_whisker_box_compare[['fusion_strategy', 'granularity', 'mean', 'z_score_vs_champ', 'is_significantly_worse']])

print("--- Top 5 Performing Models (Ranked by Stability) ---")
# We include CV and Z-Score to show why they were ranked this way
print(table_whisker_box_compare[[
    'fusion_strategy', 
    'granularity', 
    'f1_score_percent', 
    'cv_percent', 
    'stability_score'
]].head(5))

# Identify the absolute best based on the stability floor
best = table_whisker_box_compare.iloc[0]

print(f"\nBEST MODEL: {best['fusion_strategy']} (Granularity: {best['granularity']}) with F1: {best['f1_score_percent']}%")

if best['z_score_vs_champ'] < 1.96 and best.name != 0:
    print(f"NOTE: This model is statistically tied with the highest mean provider.")
print("\n")    