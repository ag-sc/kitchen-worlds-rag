import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from run_experiment import EXP_PATH, RAG_COLUMNS

SUMMARY_COLUMNS = ['seed', 'cont_succ_rate', 'completed_succ_rate', 'true_succ_rate', 'plan_length', 'plan_time',
                   'effective_time', 'wasted_time']
COLUMNS = ['exp_name', 'no_seeds', 'avg_cont_sr', 'avg_completed_sr', 'avg_true_sr', 'total_plan_length',
           'avg_plan_length', 'total_plan_time', 'avg_plan_time', 'total_effective_time', 'avg_effective_time',
           'total_wasted_time', 'avg_wasted_time']


def summarise_all_experiments():
    experiment_metadata = pd.read_csv(EXP_PATH, index_col="name")
    for name, row in tqdm(experiment_metadata.iterrows(), f"Summarising all {len(experiment_metadata)} experiments..."):
        df_exp_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        parent_folder = Path(__file__).parent / ".." / "eval_scenarios" / row["subfolder"]
        for exp_folder in parent_folder.iterdir():
            if not exp_folder.is_dir():
                continue
            res_csv = list(exp_folder.glob("seed_*.csv"))
            if len(res_csv) == 1:
                csv_path = res_csv[0]
                summary_row = pd.read_csv(csv_path).tail(1).squeeze()
                # For more information on what column corresponds to what value, see result_columns_explanation.md
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
            else:
                print(f"Warning: {exp_folder} has {len(res_csv)} matching csv files")


def evaluate_experiment_summaries():
    experiment_metadata = pd.read_csv(EXP_PATH, index_col="name")
    for name, row in tqdm(experiment_metadata.iterrows(),
                          f"Evaluating all {len(experiment_metadata)} experiment summaries..."):
        df_eval = pd.DataFrame(columns=COLUMNS)
        summary_csv = Path(__file__).parent / ".." / "eval_scenarios" / row["subfolder"] / "experiment_summary.csv"
        df = pd.read_csv(summary_csv, index_col=SUMMARY_COLUMNS[0])
        cont_sr_count, cont_sr_total = 0, 0
        completed_sr_count, completed_sr_total = 0, 0
        true_sr_count, true_sr_total = 0, 0
        plan_length, plan_time = 0, 0.0
        effective_time, wasted_time = 0.0, 0.0
        for seed, exp_row in df.iterrows():
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[1]])
            cont_sr_count += count
            cont_sr_total += total
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[2]])
            completed_sr_count += count
            completed_sr_total += total
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[3]])
            true_sr_count += count
            true_sr_total += total
            plan_length += exp_row[SUMMARY_COLUMNS[4]]
            plan_time += exp_row[SUMMARY_COLUMNS[5]]
            effective_time += exp_row[SUMMARY_COLUMNS[6]]
            wasted_time += exp_row[SUMMARY_COLUMNS[7]]
        new_row = {
            COLUMNS[0]: name.lower(),
            COLUMNS[1]: len(df),
            COLUMNS[2]: round(cont_sr_count / cont_sr_total, 3),
            COLUMNS[3]: round(completed_sr_count / completed_sr_total, 3),
            COLUMNS[4]: round(true_sr_count / true_sr_total, 3),
            COLUMNS[5]: round(plan_length, 3),
            COLUMNS[6]: round(plan_length / len(df), 3),
            COLUMNS[7]: round(plan_time, 3),
            COLUMNS[8]: round(plan_time / len(df), 3),
            COLUMNS[9]: round(effective_time, 3),
            COLUMNS[10]: round(effective_time / len(df), 3),
            COLUMNS[11]: round(wasted_time, 3),
            COLUMNS[12]: round(wasted_time / len(df), 3),
        }
        df_eval = pd.concat([df_eval, pd.DataFrame([new_row])], ignore_index=True)
        df_eval.to_csv(f'{Path(__file__).parent / ".." / "eval_scenarios" / "summary.csv"}', index=False)


def split_success_rate_string(sr: str) -> (int, int):
    # Turn '0.20 (1 / 13)' into (1, 13)
    match = re.search(r"\((\d+)\s*/\s*(\d+)\)", sr)
    if match:
        result = (int(match.group(1)), int(match.group(2)))
    return result


def prepare_correlations_evaluation():
    df_corr = pd.DataFrame(columns=[COLUMNS[0], COLUMNS[2], COLUMNS[3], COLUMNS[4], COLUMNS[6], COLUMNS[8], COLUMNS[10],
                                    COLUMNS[12]] + RAG_COLUMNS)
    experiment_metadata = pd.read_csv(EXP_PATH, index_col="name")
    for name, row in tqdm(experiment_metadata.iterrows(),
                          f"Evaluating the correlations across all {len(experiment_metadata)} experiments..."):
        summary_csv = Path(__file__).parent / ".." / "eval_scenarios" / row["subfolder"] / "experiment_summary.csv"
        df = pd.read_csv(summary_csv, index_col=SUMMARY_COLUMNS[0])
        for seed, exp_row in df.iterrows():
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[1]])
            avg_cont_sr = round(count / total, 3)
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[2]])
            avg_completed_sr = round(count / total, 3)
            count, total = split_success_rate_string(exp_row[SUMMARY_COLUMNS[3]])
            avg_true_sr = round(count / total, 3)
            new_row = {
                COLUMNS[0]: name.lower(),
                COLUMNS[2]: avg_cont_sr,
                COLUMNS[3]: avg_completed_sr,
                COLUMNS[4]: avg_true_sr,
                COLUMNS[6]: round(exp_row[SUMMARY_COLUMNS[4]], 3),
                COLUMNS[8]: round(exp_row[SUMMARY_COLUMNS[5]], 3),
                COLUMNS[10]: round(exp_row[SUMMARY_COLUMNS[6]], 3),
                COLUMNS[12]: round(exp_row[SUMMARY_COLUMNS[7]], 3),
                RAG_COLUMNS[0]: 1.0 if row[RAG_COLUMNS[0]] > 0.0 else 0.0,
                RAG_COLUMNS[1]: 1.0 if row[RAG_COLUMNS[1]] > 0.0 else 0.0,
                RAG_COLUMNS[2]: 1.0 if row[RAG_COLUMNS[2]] > 0.0 else 0.0,
                RAG_COLUMNS[3]: 1.0 if row[RAG_COLUMNS[3]] > 0.0 else 0.0,
            }
            df_corr = pd.concat([df_corr, pd.DataFrame([new_row])], ignore_index=True)
            df_corr.to_csv(f'{Path(__file__).parent / ".." / "eval_scenarios" / "correlation_data.csv"}', index=True,
                           index_label="no")


def evaluate_correlations():
    prepare_correlations_evaluation()
    res_columns = ["metric"] + [column + suffix for column in RAG_COLUMNS for suffix in ["_r", "_p"]]
    metric_columns = [COLUMNS[2], COLUMNS[3], COLUMNS[4], COLUMNS[6], COLUMNS[8], COLUMNS[10], COLUMNS[12]]
    df_corr_res = pd.DataFrame(columns=res_columns)

    df_corr = pd.read_csv(f'{Path(__file__).parent / ".." / "eval_scenarios" / "correlation_data.csv"}', index_col="no")
    for m_col in metric_columns:
        new_row = {"metric": m_col}
        for r_col in RAG_COLUMNS:
            res = spearmanr(df_corr[r_col], df_corr[m_col])
            corr = round(getattr(res, "correlation", res[0]), 3)
            p_val = round(getattr(res, "pvalue", res[1]), 5)
            print(
                f"Calculate the correlation between {r_col} and {m_col}: {'Significant' if p_val <= 0.05 else 'Not significant'} (p = {p_val}) with r = {corr}")
            new_row[r_col + "_r"] = corr
            new_row[r_col + "_p"] = p_val
        df_corr_res = pd.concat([df_corr_res, pd.DataFrame([new_row])], ignore_index=True)
    df_corr_res.to_csv(f'{Path(__file__).parent / ".." / "eval_scenarios" / "correlation_results.csv"}', index=False)


if __name__ == "__main__":
    summarise_all_experiments()
    evaluate_experiment_summaries()
    evaluate_correlations()
