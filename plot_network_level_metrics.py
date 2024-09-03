"""
Script to plot Pluralistic Homophily across different networks at varying thresholds.
Author: Fernando Barraza
Date: March 2024

This script reads network data, computes Pluralistic Homophily metrics, and generates plots for various networks
on a shared y-axis scale.

Libraries:
- pandas for data manipulation
- matplotlib.pyplot for plotting
- numpy for numerical operations

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Set the names of results file with a prefix according to the dataset key
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Dictionary of networks with their respective sampling factors and readable names
networks = {
    'so': 'StackOverflow',
    'dblp': 'DBLP',
    'amazon':'Amazon',
    'livejournal': 'LiveJournal',
    'youtube': 'YouTube',
    'orkut': 'Orkut',
    'ppi': 'Protein-Protein Interaction',
    'ddi': 'Drug-Drug Interaction',
    'celegans': 'C. Elegans'
}

# Determine the global y-axis limits
global_min = np.inf
global_max = -np.inf
for network_key, _ in networks.items():
    results_file = f'{network_key}_networkLevel_results.csv'
    data = pd.read_csv(results_file)
    data = data[data['Threshold'] != 'def']
    global_min = min(global_min, data['Pluralistic Homophily'].min())
    global_max = max(global_max, data['Pluralistic Homophily'].max())

# Create a 3x3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs = axs.flatten()  # Flatten the array of axes to simplify indexing


# Function to plot relationship on given axes
def plot_relationship(ax, data, x_var, y_var, network_name, xlabel, ylabel, show_xlabels=True, show_ylabels=True):
    ax.plot(data[x_var], data[y_var], marker='o')
    ax.set_title(network_name)
    ax.set_ylim([global_min, global_max])  # Apply the global y-axis limits
    if show_ylabels:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.grid(True)
    if show_xlabels:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])


# Plot each network in the subplot
for i, (network_key, readable_name) in enumerate(networks.items()):
    results_file = f'{network_key}_networkLevel_results.csv'
    data = pd.read_csv(results_file)
    data = data[data['Threshold'] != 'def']
    data['Threshold'] = data['Threshold'].astype(float)
    show_xlabels = i >= 6  # Only show x-labels for the bottom row subplots
    show_ylabels = i % 3 == 0  # Only show y-labels for the left column subplots
    plot_relationship(axs[i], data, 'Threshold', 'Pluralistic Homophily', readable_name, 'Threshold',
                      'Pluralistic Homophily, $h$', show_xlabels, show_ylabels)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Adjust the top margin to make space for the main title
plt.show()
