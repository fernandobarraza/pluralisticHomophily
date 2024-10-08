"""
Script to plot Pluralistic Homophily across different networks at varying thresholds.
Author: Fernando Barraza
Date: March 2024

This script reads network data, computes Pluralistic Homophily metrics, and generates plots for various networks.

Libraries:
- pandas for data manipulation
- matplotlib.pyplot for plotting
- matplotlib.markers for custom markers

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Set the names of network files with a prefix according to the dataset key

"""

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import pandas as pd

# Define a list of markers to differentiate the networks
markers = list(mmarkers.MarkerStyle.markers.keys())

# Dictionary of networks with their respective sampling factors and readable names
networks = {
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

# Prepare the plot
plt.figure(figsize=(14, 8))

# Collect all data first to ensure we have consistent coloring and markers
all_data = []

for network_key, readable_name in networks.items():
    results_file = f'{network_key}_networkLevel_results.csv'
    network_data = pd.read_csv(results_file)
    network_data = network_data[network_data['Threshold'] != 'def']
    all_data.append((network_data, readable_name))

# Now plot all networks on the same axes
for i, (data, name) in enumerate(all_data):
    plt.plot(data['Threshold'], data['Pluralistic Homophily'],
             marker=markers[i % len(markers)], label=name)

# Add labels and legend inside the plot area
plt.xlabel('Threshold')
plt.ylabel('Pluralistic Homophily, $h$')
plt.legend(loc='upper left')
plt.grid(True)

plt.ylim(-0.45, 0.45)

# Show the plot
plt.show()
