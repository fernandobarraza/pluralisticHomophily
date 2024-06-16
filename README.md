# Pluralistic Homophily Analysis

This project implements a comprehensive pipeline for analyzing pluralistic homophily in various network datasets. The analysis includes data processing, community detection, metric calculation, and correlation analysis using Python.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction

Pluralistic homophily refers to the tendency of individuals in a network to associate with similar others based on multiple attributes. This project provides tools to analyze and visualize these tendencies in large network datasets. The implemented pipeline follows a structured approach to process data, detect communities, calculate various metrics, and analyze correlations.

## Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/pluralistic-homophily-analysis.git

# Change to the project directory
cd pluralistic-homophily-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Dataset Reading and Network Reconstruction:**
   - Datasets are read to reconstruct network graphs. For larger networks such as LiveJournal, YouTube, and Orkut, sampling techniques are applied to effectively manage execution times during subsequent calculations.

2. **Community Detection:**
   - Using the HLC Algorithm: In networks like SO, PPI, DDI, and C. elegans, communities are detected using the HLC algorithm.
   - Using Ground-Truth Data: In networks such as DBLP, Amazon, LiveJournal, YouTube, and Orkut, communities are identified based on ground-truth data.

3. **Metric Calculation:**
   - Network Level: Overall pluralistic homophily \( h \), OC, and \( \tilde{\rho} \) across communities are computed.
   - Node Level: Metric \( h_v \) and centrality measures such as \( C_d \),  \( C_c \),  \( C_b \), \( C_e \) are calculated for individual nodes.

4. **Correlation Analysis and Visualization:**
   - The linear and monotonic relationships between centrality metrics and pluralistic homophily at both the network and the node levels are initially examined using the Spearman correlation index. Logistic regression models are then utilized to predict the categories of pluralistic homophily (assortative, non-assortative, disassortative) based on the centrality measures calculated for each node.

Some general consideratios:

- You must set the defaults directories for datasets and experiments result. Set the name of the directories within each script depending of your preferences.
- All network datasets files must be in NCol format.
- A dataset dictionary named 'netowrk' inside the scripts defines the network(s) to be processed. Change on your preferences.
- Set the variables flags at the top of the script to control if you want to save the results and plot images.

Run Pipeline as following:
1. Run hlc_run.py . This script calculates the overlapping communities for your network. If you have a ground-truth community file, this step is not necessary.
2. Run calc_pluralistic_homophily.py . This script generates local pluralistic homophily for each node within your network.
3. Run calc_networks_metrics.py. This scripts calculates pluralistic homophily for the network.
4. Run calc_spearman.py if you want to calculate Spearman correlation betweeen pluralistic homophily and network-level metrics.
5. Run calc_centrality_metrics.py to calculate centrality measures for the network.

All plot_*.py files can be executed any time after the pipeline is completed.

##  Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

##  License

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

