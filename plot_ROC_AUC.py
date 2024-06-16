"""
Script to generate ROC curves for centrality measures vs. homophily for different networks.
Author: Fernando Barraza
Date: March 2024

This script loads network data, applies logistic regression, and generates ROC curves for various classes.
The networks analyzed include StackOverflow, DBLP, Amazon, LiveJournal, YouTube, Orkut, Protein-Protein Interaction, Drug-Drug Interaction, and C. Elegans.

Libraries:
- pandas for data manipulation
- numpy for numerical operations
- matplotlib.pyplot for plotting
- sklearn for machine learning and evaluation
- scipy for interpolation

Usage:
Modify the base_path variable to point to the directory containing your data files.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Datasets
datasets = {
    'so': 'StackOverflow',
    'dblp': 'DBLP',
    'amazon': 'Amazon',
    'livejournal': 'LiveJournal',
    'youtube': 'YouTube',
    'orkut': 'Orkut',
    'ppi': 'Protein-Protein Interaction',
    'ddi': 'Drug-Drug Interaction',
    'celegans': 'C. Elegans'
}

# List of networks
networks = list(datasets.keys())
network_legends = list(datasets.values())

# Modify this to the base directory of your files
base_path = '/Volumes/Doctorado/experiments/ext/node_level_analysis/'

# Configure subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True)
ax = axes.flatten()

# Classification labels
class_labels = {0: 'Assortative', 1: 'Non-Assortative', 2: 'Disassortative'}

# Process each network
for idx, network in enumerate(networks):
    print(f'Processing network: {network}')
    data_path = f"{base_path}{network}_node-level_metrics_3000.csv"
    data = pd.read_csv(data_path)
    X = data[['degree', 'closeness', 'eigenvector']]
    y = data['label']

    # Verify if classes are present and filter classes with very few examples
    class_counts = y.value_counts()
    print(f'Class counts before filtering: {class_counts.to_dict()}')
    valid_classes = class_counts[class_counts >= 5].index
    data = data[data['label'].isin(valid_classes)]
    X = data[['degree', 'closeness', 'eigenvector']]
    y = data['label']

    class_counts_after = y.value_counts()
    print(f'Class counts after filtering: {class_counts_after.to_dict()}')

    if len(valid_classes) < 2:
        print(f'Not enough valid classes for network: {network}, skipping...')
        continue

    y_bin = label_binarize(y, classes=valid_classes)
    n_classes = y_bin.shape[1]
    print(f'Number of classes after binarization: {n_classes}')

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = OneVsRestClassifier(LogisticRegression(random_state=123))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        if np.sum(y_train[:, i]) == 0:
            print(f'Class {i} has no samples in training set, skipping...')
            continue

        fpr[i] = []
        tpr[i] = []
        roc_auc[i] = []

        for train, test in skf.split(X_train_scaled, y_train[:, i]):
            probas_ = model.fit(X_train_scaled[train], y_train[train][:, i]).decision_function(X_train_scaled[test])
            fpr_tmp, tpr_tmp, _ = roc_curve(y_train[test][:, i], probas_)
            fpr[i].append(fpr_tmp)
            tpr[i].append(tpr_tmp)
            roc_auc[i].append(auc(fpr_tmp, tpr_tmp))

        # Interpolate ROC curves to have the same length
        all_fpr = np.unique(np.concatenate([fpr_ for fpr_ in fpr[i]]))
        mean_tpr = np.zeros_like(all_fpr)

        for fpr_, tpr_ in zip(fpr[i], tpr[i]):
            mean_tpr += interp1d(fpr_, tpr_)(all_fpr)

        mean_tpr /= len(fpr[i])

        fpr[i] = all_fpr
        tpr[i] = mean_tpr
        roc_auc[i] = np.mean(roc_auc[i])

    # Plot ROC curve for each class in its respective subplot
    colors = ['blue', 'green', 'red']
    for i, color in zip(range(n_classes), colors):
        if i in roc_auc:
            ax[idx].plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_labels[i]} (area = {roc_auc[i]:.2f})')
    ax[idx].plot([0, 1], [0, 1], 'k--', lw=2)
    ax[idx].set_xlim([0.0, 1.0])
    ax[idx].set_ylim([0.0, 1.05])
    ax[idx].set_title(f'{network_legends[idx]}')

    if idx > 5:
        ax[idx].set_xlabel('False Positive Rate')
    if idx in (0, 3, 6):
        ax[idx].set_ylabel('True Positive Rate')

    # Show legend
    ax[idx].legend(loc="lower right")

plt.tight_layout()
plt.show()
