import argparse
import ast
import gc
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from RAG4Robots.src.plan_manager import RAGPlanManager
from pybullet_planning.tutorials.test_vlm_tamp import get_vlm_tamp_agent_parser_given_config
from pybullet_planning.vlm_tools import run_vlm_tamp_with_argparse

SEED_AMOUNT = 100
SEED_PATH = Path(__file__).resolve().parent.parent / "eval_scenarios" / "seeds.txt"
EXP_PATH = Path(__file__).resolve().parent / "experiment_setup.csv"
RAG_COLUMNS = ['recipes', 'wikihow', 'videos', 'locations']

PLAN_FOLDER = "plans"
PLAN_SEEDS = ['414', '992349', '910001']


def update_parser(conf):
    parser = get_vlm_tamp_agent_parser_given_config(conf)
    parser.add_argument("--rag_recipes", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the Recipe1M+ database should be used in RAG?")
    parser.add_argument("--rag_wikihow", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the WikiHow database should be used in RAG?")
    parser.add_argument("--rag_cutting_vids", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the cutting tutorial videos should be used in RAG?")
    parser.add_argument("--rag_cskg_locations", type=float, choices=[Range(  0.0, 1.0)], default=0.5,
                        help="What percentage of the CSKG Locations should be used in RAG?")
    parser.add_argument("--rag_plans", type=bool, default=False, help="Should the dynamic plan database be used?")
    return parser


def run_all_experiments(start_idx: int):
    with open(SEED_PATH, "r") as f:
        seeds = [int(line.strip()) for line in f]

    experiment_data = pd.read_csv(EXP_PATH, index_col="exp_idx")
    for idx, row in tqdm(experiment_data.iterrows(), f"Running all {len(experiment_data)} experiments..."):
        if idx < start_idx:
            continue
        folder = row["subfolder"]
        sys.argv = [
            sys.argv[0],
            "--open_goal", "make chicken soup",
            "--rag_recipes", str(row[RAG_COLUMNS[0]]),
            "--rag_wikihow", str(row[RAG_COLUMNS[1]]),
            "--rag_cutting_vids", str(row[RAG_COLUMNS[2]]),
            "--rag_cskg_locations", str(row[RAG_COLUMNS[3]]),
            "--exp_subdir", folder,
            "--planning_mode", "actions",
            "--dual_arm"
        ]

        if check_experiment_needed(folder):
            for s in tqdm(seeds, f"Running the experiment \'{row['name']}\' with all seeds"):
                if check_seed_needed(folder, s):
                    run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s)
                    gc.collect()
                    print(f'Finished experiment with seed {s}')
                    time.sleep(5)


def run_planning_experiment():
    sys.argv = [
        sys.argv[0],
        "--open_goal", "make chicken soup",
        "--rag_recipes", "0.0",
        "--rag_wikihow", "0.0",
        "--rag_cutting_vids", "0.0",
        "--rag_cskg_locations", "0.0",
        "--exp_subdir", PLAN_FOLDER,
        "--planning_mode", "actions",
        "--dual_arm", "--rag_plans"
    ]

    # Create the 3 foundational plans
    for s in PLAN_SEEDS:
        run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s)
        gc.collect()
        plan = get_plan_from_seed(s)
        RAGPlanManager.add_new_plan(plan)

    # Perform the 100 experiment seeds
    with open(SEED_PATH, "r") as f:
        seeds = [int(line.strip()) for line in f]
    if check_experiment_needed(PLAN_FOLDER):
        for s in tqdm(seeds, f"Running the experiment \'Dynamic plans\' with all seeds"):
            if check_seed_needed(PLAN_FOLDER, s):
                run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s)
                gc.collect()
                print(f'Finished experiment with seed {s}')
                plan = get_plan_from_seed(s)
                RAGPlanManager.add_new_plan(plan)
                time.sleep(5)


def check_experiment_needed(folder: str, seed_amount=SEED_AMOUNT) -> bool:
    # ToDo: Get path to experiment results automatically from experiment config
    full_path = Path(__file__).parent / ".." / "eval_scenarios" / folder
    full_path = full_path.resolve()
    full_path.mkdir(parents=True, exist_ok=True)
    subfolders = [p for p in full_path.iterdir() if p.is_dir()]
    return len(subfolders) < seed_amount


def check_seed_needed(folder: str, seed: str) -> bool:
    # ToDo: Get path to experiment results automatically from experiment config
    full_path = Path(__file__).parent / ".." / "eval_scenarios" / folder
    full_path = full_path.resolve()
    matching_folders = list(full_path.glob(f"*seed_{seed}"))
    return len(matching_folders) == 0


def get_plan_from_seed(seed: str) -> str:
    full_path = Path(__file__).parent / ".." / "eval_scenarios" / PLAN_FOLDER
    full_path = full_path.resolve()
    matching_folder = list(full_path.glob(f"*seed_{seed}"))
    if len(matching_folder) != 1:
        return None
    res_csv = list(matching_folder[0].glob("seed_*.csv"))
    if len(res_csv) != 1:
        return None
    results = pd.read_csv(res_csv[0], index_col="idx")
    plan = []
    for idx, row in results.iterrows():
        action = transform_planning_string(row['planning_node'])
        success = "FAILED" if row["status"] == "failed" else "SUCCESS"
        act_suc = f'{action} [{success}]'
        if act_suc not in plan:
            plan.append(act_suc)
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))


def transform_planning_string(node_desc: str):
    match = re.search(r"\[(.*)\]", node_desc)
    if not match:
        return None
    parts = ast.literal_eval("[" + match.group(1) + "]")
    func = parts[0]
    args = ", ".join(parts[1:])
    return f"{func}({args})"


def generate_seeds_for_experiment(n=SEED_AMOUNT, path=SEED_PATH):
    seeds = []
    for i in range(n):
        seeds.append(random.randint(0, 10 ** 6 - 1))
    with open(path, "w") as f:
        f.writelines(f"{seed}\n" for seed in seeds)
    print(f'Finished generating {n} seeds')

# https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-in-a-range-using-arg
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


if __name__ == "__main__":
    # generate_seeds_for_experiment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_start_idx", type=int, default=0)
    parser.add_argument("--plan_exp", type=bool, default=False)
    args = parser.parse_args()
    if args.plan_exp:
        run_planning_experiment()
    else:
        run_all_experiments(args.exp_start_idx)
