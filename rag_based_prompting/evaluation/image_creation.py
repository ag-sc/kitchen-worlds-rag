from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm

# Column Mapping
rag_ax = {
    'recipes': 'Recipes',
    'wikihow': 'WikiHow',
    'videos': 'Tutorial Videos',
    'locations': 'CSKG Locations',
    'plans': 'Plans'
}
metr_ax = {
    'avg_cont_sr': 'ConSR',
    'avg_completed_sr': 'ComSR',
    'avg_true_sr': 'TSR',
    'avg_plan_length': 'PL',
    'avg_plan_time': 'TPT',
    'avg_effective_time': 'EPT',
    'avg_wasted_time': 'IPT',
}


def preprocess_correlation_data():
    # Data pre-processing
    file = f'{Path(__file__).parent / ".." / "eval_scenarios" / "correlation_results.csv"}'
    df = pd.read_csv(file)

    # Extract metrics (row labels)
    metrics = [metr_ax.get(m, m) for m in df["metric"].values]

    # Extract correlation (r) and p-values into matrices
    r_columns = [col for col in df.columns if col.endswith("_r")]
    p_columns = [col for col in df.columns if col.endswith("_p")]

    # Clean column names for axis mapping
    rag_labels = [rag_ax.get(c.replace("_r", ""), c.replace("_r", "")) for c in r_columns]
    correlations = df[r_columns].to_numpy()
    p_values = df[p_columns].to_numpy()

    # Transpose so metrics are on x-axis and sources on y-axis
    correlations = correlations.T
    p_values = p_values.T
    return p_values, correlations, metrics, rag_labels


def create_and_save_heatmap(p_values, correlations, metrics, rag_labels):
    # Mask for significance (only paint when p > 0.05)
    mask = p_values > 0.05
    # Prepare a colormap: 3 reds for negative, 3 greens for positive
    bounds = [-1, -0.5, -0.3, 0, 0.3, 0.5, 1]
    cmap = plt.cm.RdYlGn
    norm = BoundaryNorm(bounds, cmap.N)

    # Replace values where p <= 0.05 with a neutral gray background
    plot_data = np.where(mask, 0, correlations)  # placeholder values for coloring

    # Create heatmap
    plt.figure(figsize=(9, 4))
    ax = sns.heatmap(plot_data, annot=correlations, fmt=".3f", cmap=cmap, norm=norm, cbar=True, linewidths=0.5,
                     linecolor='gray', annot_kws={"color": "black"}, xticklabels=metrics, yticklabels=rag_labels)

    # Overlay gray for non-significant cells
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightgray', edgecolor='gray', lw=0.5))

    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)  # save as PNG, high resolution
    plt.show()


if __name__ == '__main__':
    p_values, correlations, metrics, rag_labels = preprocess_correlation_data()
    create_and_save_heatmap(p_values, correlations, metrics, rag_labels)
