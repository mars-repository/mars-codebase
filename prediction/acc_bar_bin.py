"""
Code for calculating Acc-5 and Acc-15 per class and plotting the results as a bar plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

save_dir = "evaluation_results_50_10_epoch20"
predictions_path = os.path.join(save_dir, "all_predictions.npy")
labels_path = os.path.join(save_dir, "all_labels.npy")

if not os.path.exists(predictions_path) or not os.path.exists(labels_path):
    raise FileNotFoundError(
        f"Predictions or labels not found in {save_dir}. "
        "Please run the evaluation script first."
    )

# Load the predictions and labels
all_predictions = np.load(predictions_path)
all_labels = np.load(labels_path)

num_bins = 50

# if labels > 50, then make them 50
all_labels = np.minimum(all_labels, num_bins - 1)

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions, labels=range(num_bins), normalize="true")

main_diag = np.diag(cm)

# Padding the off-diagonals to match the length of the main diagonal (For Acc-15)
upper_diag = np.pad(np.diag(cm, k=1), (0, 1), mode='constant')
lower_diag = np.pad(np.diag(cm, k=-1), (1, 0), mode='constant')

# Sum the main diagonal and the upper and lower diagonals
class_accuracies = main_diag + upper_diag + lower_diag # For Acc-15
# class_accuracies = main_diag # For Acc-5

print("Class Accuracies: ", class_accuracies[:22])

# remove the first class accuracy (empty class)
class_accuracies = class_accuracies[1:]
num_bins -= 1

# Plot the accuracy for each class as a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x=np.arange(1, 11), y=class_accuracies[:10], palette="Blues_d")

plt.title("Acc-5 per Class", fontsize=16)
# plt.title("Acc-15 per Class", fontsize=16)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)  # Set y-axis from 0 to 1

# Annotate each bar with the accuracy value
for i, v in enumerate(class_accuracies[:20]):
    if i == 0:
        continue
    plt.text(i, v + 0.02, f'{v:.2f}', color='black', ha='center', fontsize=10)

plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, "class_accuracy_barplot_50_10_epoch20_acc5_top10.pdf")

plt.savefig(plot_path, dpi=300)

print(f"Bar plot saved to {plot_path}")
