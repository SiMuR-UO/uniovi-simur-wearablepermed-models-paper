import pandas as pd
from pathlib import Path

# Read the CSV file
df_metrics = pd.read_csv("/home/miguel/temp/paper/moe_rf/4_classes/moe_rf_metrics.csv")

# Compute mean and std (numeric columns only)
mean_row = df_metrics.mean(numeric_only=True)
std_row = df_metrics.std(numeric_only=True)

# Add a label for the index column
mean_row["loop"] = "mean"
std_row["loop"] = "std"

# Append to dataframe
df_metrics = pd.concat(
    [df_metrics, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

df_metrics.to_csv("/home/miguel/temp/paper/moe_rf/4_classes/moe_rf_metrics_test.csv", index=False, decimal=",") 