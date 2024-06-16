"""
Script for Network Analysis and Correlation Calculation
Author: Fernando Barraza
Date: Jan 2024

This script processes network data to calculate Spearman correlations between
pluralistic homophily, overlap coverage, and average scaled link-density. The results are output to the console.

Libraries:
- pandas for data manipulation
- scipy for statistical analysis

Usage:
Modify the base_dir variable to point to the appropriate directory containing your data files.
Run the script to calculate and print Spearman correlations for each network.
"""

import pandas as pd
from scipy.stats import spearmanr

# Define the base directory for experiments
base_dir = '/Volumes/Doctorado/experiments/'

# Define networks and their respective sampling factors and readable names
networks = {
    'so': (None, 'StackOverflow'),
    'dblp': (None, 'DBLP'),
    'amazon': (None, 'Amazon'),
    'livejournal': (0.05, 'LiveJournal'),
    'youtube': (0.1, 'YouTube'),
    'orkut': (0.01, 'Orkut'),
    'ppi': (None, 'Protein-Protein Interaction'),
    'ddi': (None, 'Drug-Drug Interaction'),
    'celegans': (None, 'C. Elegans')
}

# Process each network and calculate correlations
for network_key, (factor, readable_name) in networks.items():
    factor_part = f'_RWR_{factor}' if factor else ''
    results_file = f'{base_dir}ext/results/{network_key}/{network_key}_networkLevel_results{factor_part}_new.csv'

    # Load data
    data = pd.read_csv(results_file)
    data = data.query("Threshold != 'def'")

    print(f"Analysis for: {readable_name}")

    # Calculate Spearman correlation between pluralistic homophily and overlap coverage
    corr_h_oc, p_value_h_oc = spearmanr(data['Pluralistic Homophily'], data['Overlap Coverage'])
    print(f"Spearman correlation between h and OC: {corr_h_oc}, p-value: {p_value_h_oc}")

    # Calculate Spearman correlation between pluralistic homophily and average scaled link-density
    corr_h_asdl, p_value_h_asdl = spearmanr(data['Pluralistic Homophily'], data['Avg Scaled Link-Density'])
    print(f"Spearman correlation between h and asld: {corr_h_asdl}, p-value: {p_value_h_asdl}")

# Uncomment the following code if want to visualize the results
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(data['Threshold'], data['Pluralistic Homophily'], marker='o')
# plt.title('Network-Level pluralistic homophily vs Threshold for ' + readable_name)
# plt.xlabel('Threshold')
# plt.ylabel('Network-Level pluralistic homophily, $h$')
# plt.grid(True)
# plt.show()
