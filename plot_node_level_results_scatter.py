"""
Script to create scatter plots of centrality measures vs. homophily for different networks.
Author: Fernando Barraza
Date: March 2024

This script loads network data, calculates statistical properties, and generates scatter plots.
The networks analyzed include StackOverflow, DBLP, Amazon, LiveJournal, YouTube, Orkut, Protein-Protein Interaction, Drug-Drug Interaction, and C. Elegans.

Libraries:
- matplotlib.pyplot for plotting
- pandas for data manipulation

Usage:
Modify the base_dir variable to point to the directory containing your data files.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Dictionary of networks
all_networks = {
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

# Modify this to the base directory of your files
base_dir = '/Volumes/Doctorado/experiments/ext/'
k = 10  # Number of standard deviations to adjust the limits

# Plot configuration
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=False)
fig.suptitle('Scatter Plots of Centrality Measures vs. Homophily for Different Networks')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Colors by centrality measure
colors = {'degree': 'red', 'closeness': 'green', 'eigenvector': 'blue'}

# Iterate through each network and load the corresponding data
for i, (network_key, network_name) in enumerate(all_networks.items()):
    if i >= 9:  # Prevent error if there are more networks than subplots
        break
    file_path = f'{base_dir}node_level_analysis/{network_key}_node-level_metrics_3000.csv'
    try:
        data = pd.read_csv(file_path)
        # Load complete homophily data to calculate mean and standard deviation
        homophily_data = pd.read_csv(f'{base_dir}node_level_analysis/{network_key}_homophilies.csv')
        mean_h = homophily_data['h_v'].mean()
        std_h = homophily_data['h_v'].std()

        # Add lines for mean +/- 1 standard deviation
        axes[i].axhline(y=mean_h + std_h, color='gray', linestyle='--', linewidth=1.5)
        axes[i].axhline(y=mean_h - std_h, color='gray', linestyle='--', linewidth=1.5)

        # Scatter plot for each centrality measure
        for centrality_measure in ['degree', 'closeness', 'eigenvector']:
            axes[i].scatter(data[centrality_measure], data['homophily_value'],
                            alpha=0.3, marker='.', color=colors[centrality_measure],
                            label=f'{centrality_measure.capitalize()}')

        axes[i].set_title(network_name)
        axes[i].set_xlim(0, 1)  # X-axis limits
        axes[i].set_ylim(mean_h - k * std_h, mean_h + k * std_h)  # Y-axis limits adjusted
        if i > 6:
            axes[i].set_xlabel('Normalized Centrality Measures')
        if i in (0, 3, 6):
            axes[i].set_ylabel('Homophily Value')

        # Show legend only in the first subplot to avoid repetition
        if i == 0:
            axes[i].legend()
    except FileNotFoundError:
        axes[i].text(0.5, 0.5, f'File not found for {network_name}', horizontalalignment='center',
                     verticalalignment='center')
        continue

# Adjust subplots and show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
