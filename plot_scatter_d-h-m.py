"""
Script to generate scatter plots for centrality measures vs. local pluralistic homophily.
Author: Fernando Barraza
Date: March 2024

This script reads network data, computes metrics, and generates scatter plots of centrality measures against local pluralistic homophily for various networks.

Libraries:
- pandas for data manipulation
- numpy for numerical operations
- matplotlib.pyplot for plotting

Usage:
Modify the input_directory and output_directory variables to point to the appropriate directories containing your data files.
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np

import funcs_common


def read_file(file, separator, skip_header=False):
    """
    Reads a file and processes each line to remove brackets and extra spaces.

    Args:
        file (str): The path to the file to be read.
        separator (str): The delimiter to split each line.
        skip_header (bool): Whether to skip the first line of the file.

    Returns:
        list: A list of processed lines.
    """
    result_list = []
    with open(file, "r") as f:
        if skip_header:
            next(f)
        for line in f:
            line2 = ''.join(c for c in line if c not in '[]').strip()
            list_line = [element.strip() for element in line2.split(separator)]
            result_list.append(list_line)
    return result_list


alg = 'alg3'
scalar_method = 'm1'

verbose = True
plot_show = True
plot_save = False

dataset_dict = {
    "so": ("StackOverflow", 8, 10),
    "dblp": ("DBLP", 8, 10),
    "amazon": ("Amazon", 8, 10),
    "livejournal": ("LiveJournal", 12, 10),
    "youtube": ("Youtube", 12, 10),
    "orkut": ("Orkut", 12, 10),
    "ppi": ("Protein-Protein Interaction", 8, 6),
    "ddi": ("Drug-Drug Interaction", 8, 5),
    "celegans": ("C.elegans", 3, 5)
}

input_directory = '/Volumes/Doctorado/experiments/'
output_directory = '/Volumes/Doctorado/experiments/'
data_dir = 'datasets/'
plot_dir = 'plots/'
weights_option = False

HOMOPHILY_AXIS = 1
DEGREE_AXIS = 2
MEMBERSHIP_AXIS = 3

y_axis = 1

memberships_sample = list()
homophily_sample = list()
degree_sample = list()

max_cols = 3

num_plots = len(dataset_dict)
num_rows = math.ceil(num_plots / max_cols)
num_cols = min(max_cols, num_plots)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(3 * num_cols, 3 * num_rows))

if num_plots == 1:
    iter_axes = iter([axes])
else:
    iter_axes = iter(axes.flatten())

max_dot_lim_upper = -np.inf

t0 = time.time()

for i_subplot, dataset in enumerate(dataset_dict.keys()):
    print("i_subplot=", i_subplot)

    input_file = dataset + "_network"
    homophilies_file = f'/Volumes/Doctorado/experiments/ext/{dataset}_homophilies_{alg}_{scalar_method}.csv'

    delay = time.time() - t0
    print("duration: %.2f s." % delay)

    if verbose:
        print("running for ", dataset)
        print("reading ", homophilies_file)

    homofilies = read_file(homophilies_file, ",", skip_header=True)

    homophily = [float(i[HOMOPHILY_AXIS]) for i in homofilies]
    degrees = [float(i[DEGREE_AXIS]) for i in homofilies]
    memberships = [float(i[MEMBERSHIP_AXIS]) for i in homofilies]

    x_sample = degree_sample
    dot_sample = memberships_sample
    total_dot_sample = memberships

    y_sample = homophily_sample
    dataset_legend, n_std_dev, zoom_factor = dataset_dict[dataset]

    sample_size_plot = funcs_common.calculate_sample_size_to_plot(len(homophily))
    idx = np.random.choice(np.arange(len(homophily)), sample_size_plot, replace=False)

    homophily_sample[:] = np.array(homophily)[idx]
    degree_sample[:] = np.array(degrees)[idx]
    memberships_sample[:] = np.array(memberships)[idx]

    ax = next(iter_axes)

    x_inf = 1

    media_x = np.mean(x_sample)
    std_x = np.std(x_sample)

    if zoom_factor:
        x_upper = media_x + zoom_factor * std_x

    x_lower = 1

    ax.set_xlim(x_lower, x_upper)

    dot_media_total = np.mean(total_dot_sample)
    dot_std_dev_total = np.std(total_dot_sample)

    dot_lim_lower = 1
    dot_lim_upper = dot_media_total + 2 * dot_std_dev_total

    print("dot_lim_lower:", dot_lim_lower)
    print("dot_lim_upper:", dot_lim_upper)

    max_dot_lim_upper = max(max_dot_lim_upper, np.max(dot_lim_upper))
    print("max_dot_lim_upper:", max_dot_lim_upper)

    ax.set_title(dataset_legend, fontweight='bold')

    scatter_size = np.piecewise(sample_size_plot,
                                [sample_size_plot <= 10, sample_size_plot <= 100, sample_size_plot <= 1000,
                                 sample_size_plot <= 10000,
                                 sample_size_plot > 10000],
                                [50, 40, 20, 10, 5])

    scatter = ax.scatter(x_sample, y_sample, s=scatter_size, c=dot_sample, cmap='coolwarm', vmin=dot_lim_lower,
                         vmax=max_dot_lim_upper)

    if (x_sample is memberships_sample):
        label_x = "memberships per node, $m_v$"
        x_tag = 'm'
    else:
        label_x = "node degree, $d_v$"
        x_tag = 'd'

    y_tag = 'h'
    label_y = "local pluralistic homophily, $h_v$"

    subplots_x = [(num_rows - 1) * num_cols + i for i in range(num_cols)]
    subplots_y = [i * num_cols for i in range(num_rows)]

    if i_subplot in subplots_x or num_plots == 1:
        ax.set_xlabel(label_x)

    if i_subplot in subplots_y or num_plots == 1:
        ax.set_ylabel(label_y)

    homophily_mean = np.mean(homophily)
    homophily_std_dev = np.std(homophily)

    y_lower = homophily_mean - n_std_dev * homophily_std_dev
    y_upper = homophily_mean + n_std_dev * homophily_std_dev

    ax.set_ylim(y_lower, y_upper)

    ax.axhline(y=homophily_mean, color='grey', linestyle='--', linewidth=1, label=f'Mean h = {homophily_mean:.4f}')

    if dataset in ["ppi", "ddi", "celegans"]:
        epsilon = homophily_std_dev / 4
    elif dataset in ["so", "dblp", "amazon"]:
        epsilon = homophily_std_dev / 2
    else:
        epsilon = homophily_std_dev

    upper_limit = homophily_mean + epsilon
    lower_limit = homophily_mean - epsilon

    ax.axhline(y=upper_limit, color='green', linestyle='--', linewidth=1)
    ax.axhline(y=lower_limit, color='green', linestyle='--', linewidth=1)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    total_within_range = np.sum((homophily >= lower_limit) & (homophily <= upper_limit))
    proportion_total = total_within_range / len(homophily)

    print()
    print("sample length = ", len(memberships))
    print("sample_size_plot =", sample_size_plot)
    print("mean(degree)=", np.mean(degrees))
    print("mean(memberships)=", np.mean(memberships))
    print("mean(homophily)=", np.mean(homophily))
    print("max(degree)=", np.max(degrees))
    print("max(memberships)=", np.max(memberships))
    print("y_lower=", y_lower)
    print("y_upper=", y_upper)
    print("x_lower, x_upper", x_lower, x_upper)
    print(f"Proportion within non-assortativity range (total): {proportion_total:.4f}")

if True:
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(scatter, cax=cbar_ax)

    if (x_sample is memberships_sample):
        cbar.set_label('node degree, $d_v$')
    else:
        cbar.set_label('number of community memberships per node, $m_v$')

if plot_save:
    if num_plots != 6:
        dataset_names = "".join([str(val)[0] for val in dataset_dict.keys()])
    else:
        dataset_names = 'All'

    plt.savefig(output_directory + plot_dir + dataset_names + "-scatter.png")
if plot_show:
    print("Plotting....")
    plt.show()

delay = time.time() - t0
print("duration: %.2f s." % delay)
