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

import argparse

parser = argparse.ArgumentParser(description='Script for Spearman\'s coefficients calculation.')
parser.add_argument('--network_metrics_file', type=str, help='Network-Level metrics file.', default='networkLevel_results.csv')

args = parser.parse_args()

network_metrics_file = args.network_metrics_file
# Process each network and calculate correlations

# Load data
data = pd.read_csv(network_metrics_file)
data = data.query("Threshold != 'def'")

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
# plt.xlabel('Threshold')
# plt.ylabel('Network-Level pluralistic homophily, $h$')
# plt.grid(True)
# plt.show()
