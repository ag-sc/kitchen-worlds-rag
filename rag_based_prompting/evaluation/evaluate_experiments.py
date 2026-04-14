import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from run_experiment import EXP_PATH, RAG_COLUMNS, PLAN_FOLDER, PLAN_SEEDS, EXP_FOLDER_CHICKEN_SOUP, \
    EXP_FOLDER_DISHWASHER

SUMMARY_COLUMNS = ['seed', 'cont_succ_rate', 'completed_succ_rate', 'true_succ_rate', 'plan_length', 'plan_time',
                   'effective_time', 'wasted_time']
COLUMNS = ['exp_name', 'no_seeds', 'avg_consr', 'std_consr', 'avg_comsr', 'std_comsr', 'avg_tsr', 'std_tsr', 'total_pl',
           'avg_pl', 'std_pl', 'total_tpt', 'avg_tpt', 'std_tpt', 'total_ept', 'avg_ept', 'std_ept', 'total_ipt',
           'avg_ipt', 'std_ipt']
PLAN_COLUMN = ['plans']
EVALUATE_CHICKEN = True


def summarise_all_experiments(experiment_path: Path):
    experiment_metadata = get_exp_meta_with_plans()
    for name, row in tqdm(experiment_metadata.iterrows(), f"Summarising all {len(experiment_metadata)} experiments..."):
        df_exp_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        parent_folder = experiment_path / row["subfolder"]
        # Skip the initial seeds for the planning experiment
        if name.lower() == PLAN_FOLDER:
            matching_folder = []
            for f in parent_folder.iterdir():
                parts = f.name.split("_seed_")
                if len(parts) < 2:
                    continue  # skip unexpected names
                try:
                    seed = int(parts[-1])
                except ValueError:
                    continue  # skip if seed is not a number

                if seed not in PLAN_SEEDS:
                    matching_folder.append(f)
        else:
            matching_folder = parent_folder.iterdir()

        for exp_folder in matching_folder:
            if not exp_folder.is_dir():
                continue
            res_csv = list(exp_folder.glob("seed_*.csv"))
            if len(res_csv) == 1:
                csv_path = res_csv[0]
                summary_row = pd.read_csv(csv_path).tail(1).squeeze()
                # For more information on what column corresponds to what value, see result_columns_explanation.md
                try:
                    new_row = {
                        SUMMARY_COLUMNS[0]: int(csv_path.stem.split("_")[1]),
                        SUMMARY_COLUMNS[1]: summary_row['goal'],
                        SUMMARY_COLUMNS[2]: summary_row['task_idx'],
                        SUMMARY_COLUMNS[3]: summary_row['status'],
                        SUMMARY_COLUMNS[4]: summary_row['plan_len'],
                        SUMMARY_COLUMNS[5]: summary_row['planning_time'],
                        SUMMARY_COLUMNS[6]: summary_row['planning_objects'].split()[0],
                        SUMMARY_COLUMNS[7]: summary_row['object_reducer'].split()[0],
                    }
                    df_exp_summary = pd.concat([df_exp_summary, pd.DataFrame([new_row])], ignore_index=True)
                    df_exp_summary.to_csv(f'{parent_folder / "experiment_summary.csv"}', index=False)
                except:
                    print(f"Error for {exp_folder}: Empty csv file")
            else:
                print(f"Warning: {exp_folder} has {len(res_csv)} matching csv files")


def evaluate_experiment_summaries(exp_path: Path):
    experiment_metadata = get_exp_meta_with_plans()
    df_eval = pd.DataFrame(columns=COLUMNS)
    for name, row in tqdm(experiment_metadata.iterrows(),
                          f"Evaluating all {len(experiment_metadata)} experiment summaries..."):
        new_row = summarise_specific_experiment(exp_path, name, row['subfolder'])
        df_eval = pd.concat([df_eval, new_row], ignore_index=True)
    df_eval.to_csv(exp_path / "summary.csv", index=False)


def summarise_specific_experiment(exp_path: Path, exp_name: str, folder: str) -> pd.DataFrame:
    summary_csv = exp_path / folder / "experiment_summary.csv"
    df = pd.read_csv(summary_csv, index_col=SUMMARY_COLUMNS[0])
    for col in SUMMARY_COLUMNS[1:4]:
        df[col] = df[col].apply(split_success_rate_string_decimal)

    new_row = {
        COLUMNS[0]: exp_name.lower(),
        COLUMNS[1]: len(df),
        COLUMNS[2]: round(df[SUMMARY_COLUMNS[1]].mean(), 3),
        COLUMNS[3]: round(df[SUMMARY_COLUMNS[1]].std(), 3),
        COLUMNS[4]: round(df[SUMMARY_COLUMNS[2]].mean(), 3),
        COLUMNS[5]: round(df[SUMMARY_COLUMNS[2]].std(), 3),
        COLUMNS[6]: round(df[SUMMARY_COLUMNS[3]].mean(), 3),
        COLUMNS[7]: round(df[SUMMARY_COLUMNS[3]].std(), 3),
        COLUMNS[8]: round(df[SUMMARY_COLUMNS[4]].sum(), 3),
        COLUMNS[9]: round(df[SUMMARY_COLUMNS[4]].mean(), 3),
        COLUMNS[10]: round(df[SUMMARY_COLUMNS[4]].std(), 3),
        COLUMNS[11]: round(df[SUMMARY_COLUMNS[5]].sum(), 3),
        COLUMNS[12]: round(df[SUMMARY_COLUMNS[5]].mean(), 3),
        COLUMNS[13]: round(df[SUMMARY_COLUMNS[5]].std(), 3),
        COLUMNS[14]: round(df[SUMMARY_COLUMNS[6]].sum(), 3),
        COLUMNS[15]: round(df[SUMMARY_COLUMNS[6]].mean(), 3),
        COLUMNS[16]: round(df[SUMMARY_COLUMNS[6]].std(), 3),
        COLUMNS[17]: round(df[SUMMARY_COLUMNS[7]].sum(), 3),
        COLUMNS[18]: round(df[SUMMARY_COLUMNS[7]].mean(), 3),
        COLUMNS[19]: round(df[SUMMARY_COLUMNS[7]].std(), 3),
    }
    return pd.DataFrame([new_row])


def split_success_rate_string_ratio(sr: str) -> (int, int):
    # Extract the trailing integers in brackets: "(1, 13)" from "0.20 (1 / 13)"
    match = re.search(r"\((\d+)\s*/\s*(\d+)\)", sr)
    if match:
        result = (int(match.group(1)), int(match.group(2)))
    return result


def split_success_rate_string_decimal(sr: str) -> float:
    # Extract the leading decimal number: "0.20" from "0.20 (1 / 13)"
    match = re.search(r"(\d+\.\d+)", sr)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"No decimal number found in: {sr}")


def get_exp_meta_with_plans():
    experiment_metadata = pd.read_csv(EXP_PATH, index_col="name")
    plan_row = {
        "exp_idx": 16,
        "name": "Plans",
        "recipes": 0.0,
        "wikihow": 0.0,
        "videos": 0.0,
        "locations": 0.0,
        "subfolder": "plans"
    }
    experiment_metadata.loc[plan_row["name"]] = plan_row
    return experiment_metadata


def prepare_correlations_evaluation(exp_path: Path):
    df_corr = pd.DataFrame(
        columns=[COLUMNS[0], COLUMNS[2], COLUMNS[4], COLUMNS[6], COLUMNS[9], COLUMNS[12], COLUMNS[15],
                 COLUMNS[18]] + RAG_COLUMNS + PLAN_COLUMN)
    experiment_metadata = pd.read_csv(EXP_PATH, index_col="name")
    for name, row in tqdm(experiment_metadata.iterrows(),
                          f"Evaluating the correlations across all {len(experiment_metadata)} experiments..."):
        summary_csv = exp_path / row["subfolder"] / "experiment_summary.csv"
        df = pd.read_csv(summary_csv, index_col=SUMMARY_COLUMNS[0])
        for seed, exp_row in df.iterrows():
            count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[1]])
            avg_cont_sr = round(count / total, 3)
            count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[2]])
            avg_completed_sr = round(count / total, 3)
            count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[3]])
            avg_true_sr = round(count / total, 3)
            new_row = {
                COLUMNS[0]: name.lower(),
                COLUMNS[2]: avg_cont_sr,
                COLUMNS[4]: avg_completed_sr,
                COLUMNS[6]: avg_true_sr,
                COLUMNS[9]: round(exp_row[SUMMARY_COLUMNS[4]], 3),
                COLUMNS[12]: round(exp_row[SUMMARY_COLUMNS[5]], 3),
                COLUMNS[15]: round(exp_row[SUMMARY_COLUMNS[6]], 3),
                COLUMNS[18]: round(exp_row[SUMMARY_COLUMNS[7]], 3),
                RAG_COLUMNS[0]: 1.0 if row[RAG_COLUMNS[0]] > 0.0 else 0.0,
                RAG_COLUMNS[1]: 1.0 if row[RAG_COLUMNS[1]] > 0.0 else 0.0,
                RAG_COLUMNS[2]: 1.0 if row[RAG_COLUMNS[2]] > 0.0 else 0.0,
                RAG_COLUMNS[3]: 1.0 if row[RAG_COLUMNS[3]] > 0.0 else 0.0,
                PLAN_COLUMN[0]: 0.0
            }
            df_corr = pd.concat([df_corr, pd.DataFrame([new_row])], ignore_index=True)
            df_corr.to_csv(f'{exp_path / "correlation_data.csv"}', index=True, index_label="no")
    add_plans_to_correlation_preparation(exp_path)


def evaluate_correlations(exp_path: Path):
    res_columns = ["metric"] + [column + suffix for column in RAG_COLUMNS + PLAN_COLUMN for suffix in ["_r", "_p"]]
    metric_columns = [COLUMNS[2], COLUMNS[4], COLUMNS[6], COLUMNS[9], COLUMNS[12], COLUMNS[15], COLUMNS[18]]
    df_corr_res = pd.DataFrame(columns=res_columns)

    df_corr = pd.read_csv(f'{exp_path / "correlation_data.csv"}', index_col="no")
    for m_col in metric_columns:
        new_row = {"metric": m_col}
        for r_col in RAG_COLUMNS + PLAN_COLUMN:
            res = spearmanr(df_corr[r_col], df_corr[m_col])
            corr = round(getattr(res, "correlation", res[0]), 3)
            p_val = round(getattr(res, "pvalue", res[1]), 5)
            print(
                f"Calculate the correlation between {r_col} and {m_col}: {'Significant' if p_val <= 0.05 else 'Not significant'} (p = {p_val}) with r = {corr}")
            new_row[r_col + "_r"] = corr
            new_row[r_col + "_p"] = p_val
        df_corr_res = pd.concat([df_corr_res, pd.DataFrame([new_row])], ignore_index=True)
    df_corr_res.to_csv(f'{exp_path / "correlation_results.csv"}', index=False)


def add_plans_to_correlation_preparation(exp_path: Path):
    # Add to correlation data
    corr_data_file = f'{exp_path / "correlation_data.csv"}'
    df_corr = pd.read_csv(corr_data_file, index_col='no')
    summary_csv = exp_path / PLAN_FOLDER / "experiment_summary.csv"
    df = pd.read_csv(summary_csv, index_col=SUMMARY_COLUMNS[0])
    for seed, exp_row in df.iterrows():
        if seed in PLAN_SEEDS:
            continue
        count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[1]])
        avg_cont_sr = round(count / total, 3)
        count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[2]])
        avg_completed_sr = round(count / total, 3)
        count, total = split_success_rate_string_ratio(exp_row[SUMMARY_COLUMNS[3]])
        avg_true_sr = round(count / total, 3)
        new_row = {
            COLUMNS[0]: 'plans',
            COLUMNS[2]: avg_cont_sr,
            COLUMNS[4]: avg_completed_sr,
            COLUMNS[6]: avg_true_sr,
            COLUMNS[9]: round(exp_row[SUMMARY_COLUMNS[4]], 3),
            COLUMNS[12]: round(exp_row[SUMMARY_COLUMNS[5]], 3),
            COLUMNS[15]: round(exp_row[SUMMARY_COLUMNS[6]], 3),
            COLUMNS[18]: round(exp_row[SUMMARY_COLUMNS[7]], 3),
            RAG_COLUMNS[0]: 0.0,
            RAG_COLUMNS[1]: 0.0,
            RAG_COLUMNS[2]: 0.0,
            RAG_COLUMNS[3]: 0.0,
            PLAN_COLUMN[0]: 1.0
        }
        df_corr = pd.concat([df_corr, pd.DataFrame([new_row])], ignore_index=True)
        df_corr.to_csv(corr_data_file, index=True, index_label="no")
    print("Added the plan experiment results to correlation data file")


if __name__ == "__main__":
    if EVALUATE_CHICKEN:
        path = EXP_FOLDER_CHICKEN_SOUP
        goal = "make chicken soup"
    else:
        path = EXP_FOLDER_DISHWASHER
        goal = "load dishwasher"

    summarise_all_experiments(path)
    print(f'\nFinished summarising all seeds in each experiment setup for the \'{goal}\' experiment')
    evaluate_experiment_summaries(path)
    print(f'\nFinished calculating the complete summary over all seeds for the \'{goal}\' experiment')
    prepare_correlations_evaluation(path)
    print(f'\nFinished preparing the data for the correlation evaluation for the \'{goal}\' experiment')
    evaluate_correlations(path)
    print(f'\nFinished evaluating the correlations between metrics for the \'{goal}\' experiment')
