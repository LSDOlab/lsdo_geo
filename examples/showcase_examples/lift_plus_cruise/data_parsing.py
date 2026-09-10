import pickle
import numpy as np
import os

import matplotlib.pyplot as plt

# Make all plot text significantly larger for readability.
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

# Define file paths - attempt to find the second file based on name pattern
file_path = 'examples/showcase_examples/lift_plus_cruise/'
file1 = file_path+'lift_plus_cruise_lhs_timing_results_no_sweep_without_inequalities.pkl'
file2 = file_path+'lift_plus_cruise_lhs_timing_results_with_sweep_with_inequalities.pkl'
file3_temp = file_path+'lift_plus_cruise_lhs_timing_results_with_sweep_sharp_activation.pkl'

# Load data from pickle files
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Successfully loaded {filename}")
    return data

# Load the data
data1 = load_pickle(file1)
data2 = load_pickle(file2)
data3 = load_pickle(file3_temp)

# sample_results1 = data1['sample_results']
sample_results2 = data2['sample_results']

for i in range(len(sample_results2)):
    # constraint_values1 = np.concatenate(sample_results1[i][-6:-3])
    constraint_values2 = np.concatenate(sample_results2[i][-6:-3])
    # print(f"Sample {i}: No Inequalities: {constraint_values1}, With Inequalities: {constraint_values2}")
    constraint_values3 = np.concatenate(data3['sample_results'][i][:3])
    # print(f"Sample {i}: With Inequalities: {constraint_values2}, Sharp Activation: {constraint_values3}")
    differences = (constraint_values2 - constraint_values3)/np.maximum(1e-5, np.abs(constraint_values3))
    print(f"Sample {i}: Differences (With Inequalities - Sharp Activation): {differences}")

# exit()

# Extract timing data (convert to numpy arrays once)
timings1 = np.asarray(data1['sample_timings'], dtype=float)
timings2 = np.asarray(data2['sample_timings'], dtype=float)
all_timings = np.concatenate([timings1, timings2])

print(f"Dataset 1: {len(timings1)} values, range: [{timings1.min():.4f}, {timings1.max():.4f}]")
print(f"Dataset 2: {len(timings2)} values, range: [{timings2.min():.4f}, {timings2.max():.4f}]")

# Added descriptive statistics
def describe(name: str, arr: np.ndarray):
    print(f"{name} -> mean: {arr.mean():.4f}, median: {np.median(arr):.4f}, std: {arr.std(ddof=1):.4f}")

describe("Dataset 1", timings1)
describe("Dataset 2", timings2)
describe("Combined ", all_timings)

# ------------------ Adaptive / Clean Binning Section ------------------
# (Updated: fewer bins ~half previous, clearer y-axis labeling)

BIN_MODE = 'dense'
DENSE_TARGET_BINS = 20          # was 40; roughly half for clearer grouping
FIXED_WIDTH = 0.25
KMEANS_K = 5
QUANTILE_Q = 20

FRACTION_DENOMINATOR = 300      # keeps comparability with earlier plots

def kmeans_1d_edges(data: np.ndarray, k: int = 4, max_iter: int = 50) -> np.ndarray:
    data = np.asarray(data).ravel()
    data_sorted = np.sort(data)
    # Initialize centers at quantiles
    centers = np.percentile(data_sorted, np.linspace(0, 100, k, endpoint=False)[1:])
    centers = np.concatenate(([data_sorted.min()], centers))
    for _ in range(max_iter):
        # Assign
        dists = np.abs(data_sorted[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([data_sorted[labels == i].mean() if np.any(labels == i) else centers[i]
                                for i in range(len(centers))])
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    centers = np.unique(centers)
    centers.sort()
    # Build edges as midpoints
    edges = [data_sorted.min()]
    for a, b in zip(centers[:-1], centers[1:]):
        edges.append((a + b) / 2.0)
    edges.append(data_sorted.max())
    edges = np.array(edges)
    # Optional rounding for cleaner labels
    step = 0.1
    edges = np.unique(np.round(edges / step) * step)
    if edges[0] > data_sorted.min():
        edges[0] = np.floor(data_sorted.min() / step) * step
    if edges[-1] < data_sorted.max():
        edges[-1] = np.ceil(data_sorted.max() / step) * step
    return edges

def fd_edges(data: np.ndarray) -> np.ndarray:
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return np.linspace(data.min(), data.max(), 10)
    h = 2 * iqr / (len(data) ** (1 / 3))
    if h <= 0:
        return np.linspace(data.min(), data.max(), 10)
    n_bins = int(np.ceil((data.max() - data.min()) / h))
    n_bins = max(5, min(80, n_bins))
    return np.linspace(data.min(), data.max(), n_bins + 1)

def quantile_edges(data: np.ndarray, q: int = 10) -> np.ndarray:
    return np.unique(np.percentile(data, np.linspace(0, 100, q + 1)))

def fixed_edges(data: np.ndarray, width: float = 0.5) -> np.ndarray:
    lo = np.floor(data.min() / width) * width
    hi = np.ceil(data.max() / width) * width
    return np.arange(lo, hi + 1e-12, width)

def dense_edges(data: np.ndarray, target_bins: int = 40) -> np.ndarray:
    lo, hi = data.min(), data.max()
    width = (hi - lo) / target_bins if target_bins > 0 else (hi - lo) / 20
    width = max(width, 1e-6)
    return fixed_edges(data, width)

# Select bin edges
if BIN_MODE == 'cluster':
    bins = kmeans_1d_edges(all_timings, k=KMEANS_K)
elif BIN_MODE == 'fd':
    bins = fd_edges(all_timings)
elif BIN_MODE == 'quantile':
    bins = quantile_edges(all_timings, q=QUANTILE_Q)
elif BIN_MODE == 'fixed':
    bins = fixed_edges(all_timings, width=FIXED_WIDTH)
elif BIN_MODE == 'dense':
    bins = dense_edges(all_timings, target_bins=DENSE_TARGET_BINS)
else:
    bins = fd_edges(all_timings)

print(f"Using {BIN_MODE} binning with {len(bins)-1} bins. Edges: {bins}")

# Raw counts (no density)
hist1, _ = np.histogram(timings1, bins=bins, density=False)
hist2, _ = np.histogram(timings2, bins=bins, density=False)

# Convert to fraction of FRACTION_DENOMINATOR
frac1 = hist1 / FRACTION_DENOMINATOR
frac2 = hist2 / FRACTION_DENOMINATOR

# Plot grouped bar chart
# fig, ax = plt.subplots(figsize=(14, 7))
# bar_width = 0.45
# x = np.arange(len(bins) - 1)

# ax.bar(x - bar_width/2, frac1, bar_width, label="No Inequalities", alpha=0.8, edgecolor='k')
# ax.bar(x + bar_width/2, frac2, bar_width, label="With Inequalities", alpha=0.8, edgecolor='k')

# ax.set_xlabel('Execution Time Bin (s)')
# ax.set_ylabel(f'Relative Frequency (Count / {FRACTION_DENOMINATOR})')  # clearer label
# ax.set_title('Timing Distribution')

# # Tick labels with bin ranges
# bin_labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins) - 1)]
# ax.set_xticks(x)
# ax.set_xticklabels(bin_labels, rotation=55, ha='right')

# # Optional annotations (now show count and fraction)
# for xi, c, f in zip(x - bar_width/2, hist1, frac1):
#     if c > 0:
#         ax.text(xi, f, f'{c}\n({f:.3f})', ha='center', va='bottom', fontsize=7)
# for xi, c, f in zip(x + bar_width/2, hist2, frac2):
#     if c > 0:
#         ax.text(xi, f, f'{c}\n({f:.3f})', ha='center', va='bottom', fontsize=7)

# # Optional secondary y-axis with raw counts
# ax2 = ax.twinx()
# ax2.set_ylim(ax.get_ylim()[0]*FRACTION_DENOMINATOR, ax.get_ylim()[1]*FRACTION_DENOMINATOR)
# ax2.set_ylabel('Sample Count')
# ax2.grid(False)

# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig('timing_results_histogram_fraction_reduced_bins.png', dpi=300)
# plt.show()

# ------------------ Separate Vertically Aligned Plots (PDF overlay removed) ------------------
SHOW_COUNTS_SECOND_AXIS = True

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

def plot_dataset(ax, hist, frac, title):
    # Plot bars centered in each bin
    for i in range(len(bins) - 1):
        center = 0.5 * (bins[i] + bins[i+1])
        width  = (bins[i+1] - bins[i]) * 0.9
        ax.bar(center,
               frac[i],
               width=width,
               align='center',
               color='#4C72B0',
               alpha=0.75,
               edgecolor='k')
    ax.set_ylabel(f'Relative Frequency\n(Count / {FRACTION_DENOMINATOR})')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if SHOW_COUNTS_SECOND_AXIS:
        ax_sec = ax.twinx()
        ax_sec.set_ylabel('Count')
        ymin, ymax = ax.get_ylim()
        ax_sec.set_ylim(ymin * FRACTION_DENOMINATOR, ymax * FRACTION_DENOMINATOR)
        ax_sec.grid(False)

plot_dataset(ax1, hist1, frac1, 'Timing Distribution With No Inequalities')
plot_dataset(ax2, hist2, frac2, 'Timing Distribution With Inequalities')

ax2.set_xlabel('Execution Time (s)')

plt.tight_layout()
plt.savefig('timing_results_two_panel_no_pdf.png', dpi=300)
plt.show()
# ----------------------------------------------------------------------