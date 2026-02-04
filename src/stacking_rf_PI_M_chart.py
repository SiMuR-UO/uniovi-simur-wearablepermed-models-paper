from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(str(Path.cwd()) +  "/results/metrics_output.csv")

participants = df['participants'].values

base_model_accuracy_train_mean_PI = df['base_model_accuracy_train_mean_PI'].values
base_model_accuracy_train_mean_M = df['base_model_accuracy_train_mean_M'].values

base_model_accuracy_validate_mean_PI = df['base_model_accuracy_validate_mean_PI'].values
base_model_accuracy_validate_mean_M = df['base_model_accuracy_validate_mean_M'].values

meta_model_accuracy_train = df['meta_model_accuracy_train'].values 
meta_model_accuracy_test = df['meta_model_accuracy_test'].values 

participants = df['participants'].values

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# --------- Plot 1: Base models PI ---------
axes[0].plot(participants, base_model_accuracy_train_mean_PI, marker='.', label='Training Accuracy PI')
axes[0].plot(participants, base_model_accuracy_validate_mean_PI, marker='.', label='Validation Accuracy PI')

axes[0].set_ylabel('Accuracy')
axes[0].set_title('Base Model Accuracy PI')
axes[0].legend()
axes[0].grid(True)

# --------- Plot 2: Base models M ---------
axes[1].plot(participants, base_model_accuracy_train_mean_M, marker='.', label='Training Accuracy M')
axes[1].plot(participants, base_model_accuracy_validate_mean_M, marker='.', label='Validation Accuracy M')

axes[1].set_ylabel('Accuracy')
axes[1].set_title('Base Model Accuracy M')
axes[1].legend()
axes[1].grid(True)

# --------- Plot 4: Meta models ---------
axes[2].plot(participants, meta_model_accuracy_train, marker='.', markersize=4, label='Training Accuracy')
axes[2].plot(participants, meta_model_accuracy_test, marker='.', markersize=4, label='Test Accuracy')

axes[2].set_xlabel('Participants')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Meta Model Accuracy')
axes[2].legend()
axes[2].grid(True)

plt.savefig(str(Path.cwd()) + "/results/rf_chart.png", dpi=300, bbox_inches="tight")