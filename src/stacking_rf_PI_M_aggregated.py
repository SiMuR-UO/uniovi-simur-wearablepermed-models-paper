from pathlib import Path
import pandas as pd

df = pd.read_csv(str(Path.cwd()) + "/results/metrics_output.csv")

stats = (
    df
    .drop(columns="loop")          # exclude loop from aggregation
    .groupby("participants")
    .agg(["mean", "var"])
    .reset_index()                 # brings participant back as a column
)

stats.columns = ["participants" if col == ("participants", "") else f"{col[0]}_{col[1]}" for col in stats.columns]

stats.to_csv(str(Path.cwd()) + "/results/metrics_output_aggregated.csv", index=False)
