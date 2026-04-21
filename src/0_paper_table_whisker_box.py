import pandas as pd

METRIC_PI_FILENAME = 'metrics_pi.csv'
METRIC_M_FILENAM = 'metrics_m.csv'
METRIC_FILENAME = 'metrics.csv'

print("🟢 Read f1_score dataframe for each model and classes")
individual_4_pi_data = pd.read_csv('./paper/1_individual/4_classes/' + METRIC_PI_FILENAME)
individual_4_m_data = pd.read_csv('./paper/1_individual/4_classes/' + METRIC_M_FILENAM)
individual_8_pi_data = pd.read_csv('./paper/1_individual/8_classes/' + METRIC_PI_FILENAME)
individual_8_m_data = pd.read_csv('./paper/1_individual/8_classes/' + METRIC_M_FILENAM)
individual_15_pi_data = pd.read_csv('./paper/1_individual/15_classes/' + METRIC_PI_FILENAME)
individual_15_m_data = pd.read_csv('./paper/1_individual/15_classes/' + METRIC_M_FILENAM)
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
moe_vae_4_data = pd.read_csv('./paper/4_moe_vae/4_classes/' + METRIC_FILENAME, decimal=",")
moe_vae_8_data = pd.read_csv('./paper/4_moe_vae/8_classes/' + METRIC_FILENAME, decimal=",")
moe_vae_15_data = pd.read_csv('./paper/4_moe_vae/15_classes/' + METRIC_FILENAME, decimal=",")

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
moe_vae_4_data = moe_vae_4_data[moe_vae_4_data["loop"].isin(["mean", "std"])]
moe_vae_4_data = moe_vae_4_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_vae_4_data = moe_vae_4_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})
moe_vae_8_data = moe_vae_8_data[moe_vae_8_data["loop"].isin(["mean", "std"])]
moe_vae_8_data = moe_vae_8_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_vae_8_data = moe_vae_8_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})
moe_vae_15_data = moe_vae_15_data[moe_vae_15_data["loop"].isin(["mean", "std"])]
moe_vae_15_data = moe_vae_15_data.loc[:, ["loop", "moe_model_test_soft_accuracy", "moe_model_test_soft_f1_score"]]
moe_vae_15_data = moe_vae_15_data.rename(columns={"moe_model_test_soft_accuracy": "model_accuracy_test", "moe_model_test_soft_f1_score": "model_f1_score_test"})

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
moe_vae_4_data = (
    moe_vae_4_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_vae_4_data.insert(1, "granularity", 4)
moe_vae_4_data.insert(2, "fusion_strategy", "moe_vae")
moe_vae_8_data = (
    moe_vae_8_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_vae_8_data.insert(1, "granularity", 8)
moe_vae_8_data.insert(2, "fusion_strategy", "moe_vae")
moe_vae_15_data = (
    moe_vae_15_data.set_index("loop")   # mean/std become columns after transpose
      .T                                     # transpose metrics to rows
      .reset_index()
      .rename(columns={"index": "metric"})
)
moe_vae_15_data.insert(1, "granularity", 15)
moe_vae_15_data.insert(2, "fusion_strategy", "moe_vae")

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
    moe_vae_4_data,
    moe_vae_8_data,
    moe_vae_15_data
], axis=0).reset_index(drop=True)

# Convert mean and std to percent and round
table_whisker_box["mean"] = table_whisker_box["mean"].astype(float)
table_whisker_box["std"] = table_whisker_box["std"].astype(float)
table_whisker_box["f1_score_percent"] = table_whisker_box.apply(
    lambda row: f"{round(row['mean']*100, 2):.2f} ± {round(row['std']*100, 2):.2f}", axis=1
)
table_whisker_box = table_whisker_box.drop(columns=["mean", "std"])

table_whisker_box = table_whisker_box.query('metric == "model_f1_score_test"')

print("🟢 Print results")
print(table_whisker_box)