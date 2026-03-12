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

EXP_FOLDER_CHICKEN_SOUP = Path(__file__).resolve().parent.parent / "eval_scenarios" / "make_chicken_soup"
EXP_FOLDER_DISHWASHER = Path(__file__).resolve().parent.parent / "eval_scenarios" / "load_dishwasher"
EXP_PATH = Path(__file__).resolve().parent / "experiment_setup.csv"

SEEDS_CHICKEN_SOUP = 100
SEEDS_DISHWASHER = 10
RAG_COLUMNS = ['recipes', 'wikihow', 'videos', 'locations']

PLAN_FOLDER = "plans"
PLAN_SEEDS = [414, 992349, 910001]


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
    parser.add_argument("--rag_plans", action="store_true", help="Should the dynamic plan database be used?")
    return parser


def run_all_experiments(goal: str, start_idx: int, exp_folder: Path, no_seeds: int):
    with open(exp_folder / "seeds.txt", "r") as f:
        seeds = [int(line.strip()) for line in f]

    experiment_data = pd.read_csv(EXP_PATH, index_col="exp_idx")
    for idx, row in tqdm(experiment_data.iterrows(), f"Running all {len(experiment_data)} \'{goal}\' experiments..."):
        if idx < start_idx:
            continue
        folder = row["subfolder"]
        sys.argv = [
            sys.argv[0],
            "--open_goal", goal,
            "--rag_recipes", str(row[RAG_COLUMNS[0]]),
            "--rag_wikihow", str(row[RAG_COLUMNS[1]]),
            "--rag_cutting_vids", str(row[RAG_COLUMNS[2]]),
            "--rag_cskg_locations", str(row[RAG_COLUMNS[3]]),
            "--exp_dir", exp_folder,
            "--exp_subdir", folder,
            "--planning_mode", "actions",
            "--dual_arm"
        ]

        if check_experiment_needed(folder, exp_folder, no_seeds):
            for s in tqdm(seeds, f"Running the experiment \'{row['name']}\' with all seeds for \'{goal}\'"):
                if check_seed_needed(folder, s, exp_folder):
                    run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s,
                                               problem_name=get_environment_from_goal(goal))
                    gc.collect()
                    print(f'Finished experiment with seed {s}')
                    time.sleep(5)
                else:
                    print(f'Seed {s} was already evaluated')
        else:
            print(f"Experiment \'{row['name']}\' is already finished for \'{goal}\'")


def run_planning_experiment(goal: str, exp_folder: Path, no_seeds: int):
    sys.argv = [
        sys.argv[0],
        "--open_goal", goal,
        "--rag_recipes", "0.0",
        "--rag_wikihow", "0.0",
        "--rag_cutting_vids", "0.0",
        "--rag_cskg_locations", "0.0",
        "--exp_dir", exp_folder,
        "--exp_subdir", PLAN_FOLDER,
        "--planning_mode", "actions",
        "--dual_arm", "--rag_plans"
    ]
    # Setup the Planning database
    RAGPlanManager()

    # Create the 3 foundational plans
    for s in PLAN_SEEDS:
        if check_seed_needed(PLAN_FOLDER, str(s), exp_folder):
            run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s,
                                       problem_name=get_environment_from_goal(goal))
            gc.collect()
        else:
            print(f'Seed {s} was already evaluated')
        plan = get_plan_from_seed(str(s), exp_folder)
        RAGPlanManager.add_new_plan(plan)
        time.sleep(5)

    # Perform the 100 experiment seeds
    with open(exp_folder / "seeds.txt", "r") as f:
        seeds = [int(line.strip()) for line in f]
    if check_experiment_needed(PLAN_FOLDER, exp_folder, no_seeds):
        for s in tqdm(seeds, f"Running the experiment \'Dynamic plans\' with all seeds for \'{goal}\'"):
            if check_seed_needed(PLAN_FOLDER, s, exp_folder):
                run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s,
                                           problem_name=get_environment_from_goal(goal))
                gc.collect()
            else:
                print(f'Seed {s} was already evaluated')
            plan = get_plan_from_seed(s, exp_folder)
            RAGPlanManager.add_new_plan(plan)
            time.sleep(5)
            print(f'Finished experiment with seed {s}')
    else:
        print(f"Experiment \'Dynamic plans\' is already finished for \'{goal}\'")


def get_environment_from_goal(goal: str) -> str:
    if goal == "load dishwasher":
        return "test_kitchen_dishwasher"
    else:
        return "test_kitchen_chicken_soup"


def check_experiment_needed(folder: str, experiment_path=EXP_FOLDER_CHICKEN_SOUP,
                            seed_amount=SEEDS_CHICKEN_SOUP) -> bool:
    # ToDo: Get path to experiment results automatically from experiment config
    full_path = experiment_path / folder
    full_path = full_path.resolve()
    full_path.mkdir(parents=True, exist_ok=True)
    subfolders = [p for p in full_path.iterdir() if p.is_dir()]
    return len(subfolders) < seed_amount


def check_seed_needed(folder: str, seed: str, experiment_path=EXP_FOLDER_CHICKEN_SOUP) -> bool:
    # ToDo: Get path to experiment results automatically from experiment config
    full_path = experiment_path / folder
    full_path = full_path.resolve()
    matching_folders = list(full_path.glob(f"*seed_{seed}"))
    return len(matching_folders) == 0


def get_plan_from_seed(seed: str, experiment_path=EXP_FOLDER_CHICKEN_SOUP) -> str:
    full_path = experiment_path / PLAN_FOLDER
    full_path = full_path.resolve()
    full_path.mkdir(parents=True, exist_ok=True)
    matching_folder = [
        p for p in full_path.iterdir()
        if f"seed_{seed}" in p.name
    ]
    if len(matching_folder) != 1:
        print(f'No matching folder found in {full_path}')
        return None
    res_csv = list(matching_folder[0].glob("seed_*.csv"))
    if len(res_csv) != 1:
        print(f'No CSV file found in {matching_folder}')
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
    match = re.search(r"\[.*?\]", str(node_desc))
    if not match:
        return None
    candidate = match.group(0)
    candidate = re.sub(r">.*", "", candidate)
    try:
        parts = ast.literal_eval(candidate)
    except Exception:
        # Fallback: manually split strings inside the brackets
        parts = [s.strip(" '\"") for s in candidate.strip("[]").split(",")]
    func = parts[0]
    objs = ", ".join(parts[1:])
    return f"{func}({objs})"


def generate_seeds_for_experiment(n=SEEDS_CHICKEN_SOUP, exp_path=EXP_FOLDER_CHICKEN_SOUP):
    path = exp_path / "seeds.txt"
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
    # generate_seeds_for_experiment(SEEDS_DISHWASHER, EXP_FOLDER_DISHWASHER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_start_idx", type=int, default=0)
    parser.add_argument("--chicken_exp", action="store_true")
    parser.add_argument("--plan_exp", action="store_true")
    args = parser.parse_args()

    if args.chicken_exp:
        run_all_experiments("make chicken soup", args.exp_start_idx, EXP_FOLDER_CHICKEN_SOUP, SEEDS_CHICKEN_SOUP)
        if args.plan_exp:
            run_planning_experiment("make chicken soup", EXP_FOLDER_CHICKEN_SOUP, SEEDS_CHICKEN_SOUP)
    else:
        run_all_experiments("load dishwasher", args.exp_start_idx, EXP_FOLDER_DISHWASHER, SEEDS_DISHWASHER)
        if args.plan_exp:
            run_planning_experiment("load dishwasher", EXP_FOLDER_DISHWASHER, SEEDS_DISHWASHER)
