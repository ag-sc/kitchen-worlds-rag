from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm

from rag_based_prompting.evaluation.evaluate_experiments import get_exp_meta_with_plans, \
    split_success_rate_string_decimal
from rag_based_prompting.evaluation.run_experiment import EXP_FOLDER_CHICKEN_SOUP, EXP_FOLDER_DISHWASHER, PLAN_FOLDER

# Column Mapping
rag_ax = {
    'recipes': 'Recipes',
    'wikihow': 'WikiHow',
    'videos': 'Tutorial Videos',
    'locations': 'CSKG Locations',
    'plans': 'Plans'
}
avg_metr_ax = {
    'avg_consr': 'ConSR',
    'avg_comsr': 'ComSR',
    'avg_tsr': 'TSR',
    'avg_pl': 'PL',
    'avg_tpt': 'TPT',
    'avg_ept': 'EPT',
    'avg_ipt': 'IPT',
}
metr_ax = {
    'cont_succ_rate': 'ConSR',
    'completed_succ_rate': 'ComSR',
    'true_succ_rate': 'TSR',
    'plan_length': 'PL',
    'plan_time': 'TPT',
    'effective_time': 'EPT',
    'wasted_time': 'IPT',
}
boxplot_exp_map = {
    'baseline': 'None',
    'rec': 'R',
    'wh': 'W',
    'vids': 'V',
    'locs': 'L',
    'rec_wh': 'R+W',
    'rec_vids': 'R+V',
    'rec_locs': 'R+L',
    'wh_vids': 'W+V',
    'wh_locs': 'W+L',
    'vids_locs': 'V+L',
    'rec_wh_vids': 'R+W+V',
    'rec_wh_locs': 'R+W+L',
    'rec_vids_locs': 'R+V+L',
    'wh_vids_locs': 'W+V+L',
    'all': 'All',
    'plans': 'Plans',
}

METRICS_BOXPLOT = ["cont_succ_rate", "completed_succ_rate", "true_succ_rate", "plan_length", "plan_time",
                   "effective_time", "wasted_time"]

EVALUATE_CHICKEN = True


def process_and_visualise_correlation_data(experiment_path: Path):
    # Data pre-processing
    file = f'{experiment_path / "correlation_results.csv"}'
    df = pd.read_csv(file)

    # Extract metrics (row labels)
    metrics = [avg_metr_ax.get(m, m) for m in df["metric"].values]

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
    create_and_save_heatmap(p_values, correlations, metrics, rag_labels, experiment_path)


def create_and_save_heatmap(p_values, correlations, metrics, rag_labels, exp_path):
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
    plt.savefig(f"{exp_path}/plots/correlation_heatmap.png", dpi=300)  # save as PNG, high resolution
    plt.show()


def plot_sr_from_plan_amount(experiment_path: Path):
    with open(experiment_path / "seeds.txt", "r") as f:
        seeds = [int(line.strip()) for line in f]

    file = f'{experiment_path / PLAN_FOLDER / "experiment_summary.csv"}'
    df = pd.read_csv(file)
    res_map = {}
    plan_amount = 3
    for s in seeds:
        ct_sr = split_success_rate_string_decimal(df.loc[df["seed"] == s, "cont_succ_rate"].iloc[0])
        # cm_sr = extract_decimal(df.loc[df["seed"] == s, "completed_succ_rate"].iloc[0])
        t_sr = split_success_rate_string_decimal(df.loc[df["seed"] == s, "true_succ_rate"].iloc[0])
        # res_map[plan_amount] = [ct_sr, cm_sr, t_sr]
        res_map[plan_amount] = [ct_sr, t_sr]
        plan_amount += 1

    x = sorted(res_map.keys())  # X-axis: sorted amounts
    # Prepare y-values for each line
    y_lines = [[], []]  # One list per line
    for amt in x:
        vals = res_map[amt]
        for i in range(2):
            y_lines[i].append(vals[i])

    # Plot each line
    plt.figure(figsize=(10, 6))
    lbls = ['Continuous SR', 'True SR']
    for i, y in enumerate(y_lines):
        plt.plot(x, y, marker='o', label=lbls[i])

    # Set axis limits
    plt.xlim(3, len(seeds)+3)
    plt.ylim(0, 1)

    # Labels, title, legend
    plt.xlabel("Amount of plans")
    plt.ylabel("Success rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{experiment_path}/plots/plan_amount_influence.png", dpi=300)  # save as PNG, high resolution
    plt.show()


def create_boxplots(experiment_path: Path):
    experiment_metadata = get_exp_meta_with_plans()
    df_combined = []
    for name, row in experiment_metadata.iterrows():
        exp_summary = experiment_path / row["subfolder"] / "experiment_summary.csv"
        exp_df = pd.read_csv(exp_summary)
        exp_df["experiment"] = row["subfolder"]

        for col in METRICS_BOXPLOT[0:3]:
            exp_df[col] = exp_df[col].apply(split_success_rate_string_decimal)

        df_combined.append(exp_df)
    df_combined = pd.concat(df_combined, ignore_index=True)

    for metric in METRICS_BOXPLOT:
        experiments = df_combined["experiment"].unique()
        grouped_data = [
            df_combined[df_combined["experiment"] == exp][metric]
            for exp in experiments
        ]
        plt.boxplot(grouped_data)

        labels = [boxplot_exp_map.get(exp, exp) for exp in experiments]
        plt.xticks(range(1, len(experiments) + 1), labels, rotation=65)
        plt.ylabel(metr_ax[metric])
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"{experiment_path}/plots/boxplot_{metric}.png", dpi=300)
        plt.close()


def create_performance_table(experiment_path: Path):
    def fmt(avg, std, precision=2):
        return f"{avg:.{precision}f} ($\\pm$ {std:.{precision}f})"

    df = pd.read_csv(experiment_path / "summary.csv")
    cols = [
        ('avg_consr', 'std_consr'),
        ('avg_comsr', 'std_comsr'),
        ('avg_tsr', 'std_tsr'),
        ('avg_pl', 'std_pl'),
        ('avg_tpt', 'std_tpt'),
        ('avg_ept', 'std_ept'),
        ('avg_ipt', 'std_ipt')
    ]

    table_rows = []
    for _, row in df.iterrows():
        formatted_values = [fmt(row[avg], row[std]) for avg, std in cols]
        # Escape underscores in experiment names
        exp_name = row['exp_name'].replace('_', '\\_')
        table_rows.append(f"\\texttt{{{exp_name}}} & " + " & ".join(formatted_values) + " \\\\")

    latex_table_content = "\n".join(table_rows)
    print(latex_table_content)


if __name__ == '__main__':
    if EVALUATE_CHICKEN:
        path = EXP_FOLDER_CHICKEN_SOUP
    else:
        path = EXP_FOLDER_DISHWASHER

    # Create the 'plots' folder
    plot_dir = path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create correlation diagram
    process_and_visualise_correlation_data(path)

    # Create plan influence diagram
    plot_sr_from_plan_amount(path)

    # Create boxplots
    create_boxplots(path)

    # Create latex table for the performance values
    # create_performance_table(path)
