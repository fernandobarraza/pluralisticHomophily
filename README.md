# Pluralistic Homophily Analysis

This project implements a comprehensive pipeline for analyzing pluralistic homophily in various network datasets. The analysis includes data processing, community detection, metric calculation, and correlation analysis using Python.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Acknowledgements](#acknowledgements)
5. [Contributing](#contributing)
6. [License](#license)

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

Run Pipeline as following:
1. Run hlc_run.py . This script calculates the overlapping communities for your network. If you have a ground-truth community file, this step is not necessary.
2. Run calc_pluralistic_homophily_metrics.py . This scripts calculates pluralistic homophily for the network and local pluralistic homophily for each node within your network.
3. Run calc_spearman.py if you want to calculate Spearman correlation betweeen pluralistic homophily and network-level metrics.
4. Run calc_centrality_metrics.py to calculate centrality measures for the network.

Some general considerations:

- You must set the defaults directories for datasets and experiments result. Set the name of the directories within each script depending of your preferences.
- A dataset dictionary named 'network' inside the scripts defines the network(s) to be processed. Change on your preferences.

1. **Dataset Reading and Network Reconstruction:**

   Download the network and community files for each dataset and put on your desired directories. All datasets used in this project are publicly available. 

Links to Public Datasets:

**General Networks (DBLP, Amazon, Livejournal, Youtube, Orkut)**: 
   - [SNAP Datasets](https://snap.stanford.edu/data/)

**Biological Networks (PPI, DDI, and C.elegans)**:
   - [BioSNAP Datasets](https://snap.stanford.edu/biodata/index.html)

Preprocessed Dataset:

   - The preprocessed datasets are also available at https://doi.org/10.5281/zenodo.13647528 .

2. **Community Detection:**

   Run the script to calculate the overlapping communities for your network. If you have a ground-truth community file, this step is not necessary.

```bash
python3 hlc_run.py -dataset_file 'dataset_file'
```

   Using the HLC Algorithm:

   - Default network file is the the text for Bee definition. Use the right dataset_file as an argument in command line execution.
   - All network datasets files must be in NCol format.
   - Set the variable weight_option to True or False according to your preferences.
   - In the code, change the threshold_runs list if you want to change for what thresholds you will cut the dendrogram in the HLC algorithm.
   - At the end, the script generates a file 'communities.txt' that is used in the next step.

3. **Metric Calculation:**

   Run the scripts for both network and local level pluralistic homophily calculation:

   - Network Level: Metric \( h \) and , OC and rho are calculated for the network.
   - Node Level: Metric \( h_v \) and centrality measures such as \( C_d \),  \( C_c \),  \( C_b \), \( C_e \) are calculated for individual nodes.

```bash
python3 calc_pluralistic_homophily_metrics.py -dataset_file 'dataset_file'
```

   Run the scripts for centrality metrics calculation, \( C_d \),  \( C_c \),  \( C_b \), \( C_e \):

```bash
python3 calc_centrality_metrics.py -community_file 'community_file' -homophily_file 'homophily_file'
```

4. **Visualization:**

All plot_*.py files can be executed any time after the pipeline is completed. No arguments are needed but you must rename the result files, using a prefix with the dataset key for each dataset.

- Set the variables flags at the top of the script to control if you want to save the results and plot images.


## Acknowledgements

The implementation of the Hierarchical Link Clustering (HLC) algorithm used in this project was originally developed by Tamás Nepusz. His implementation is available at ntamas/hlc. Special thanks to Tamás Nepusz for making the code publicly available.

##  Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

