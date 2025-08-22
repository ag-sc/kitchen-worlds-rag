import gc
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pybullet_planning.tutorials.test_vlm_tamp import get_vlm_tamp_agent_parser_given_config
from pybullet_planning.vlm_tools import run_vlm_tamp_with_argparse

SEED_AMOUNT = 100
SEED_PATH = Path(__file__).resolve().parent.parent / "eval_scenarios" / "seeds.txt"
EXP_PATH = Path(__file__).resolve().parent / "experiment_setup.csv"

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
    return parser


def run_all_experiments():
    with open(SEED_PATH, "r") as f:
        seeds = [int(line.strip()) for line in f]

    experiment_data = pd.read_csv(EXP_PATH, index_col="name")
    for name, row in tqdm(experiment_data.iterrows(), f"Running all {len(experiment_data)} experiments..."):
        folder = row["subfolder"]
        sys.argv = [
            sys.argv[0],
            "--open_goal", "make chicken soup",
            "--rag_recipes", str(row["recipes"]),
            "--rag_wikihow", str(row["wikihow"]),
            "--rag_cutting_vids", str(row["videos"]),
            "--rag_cskg_locations", str(row["locations"]),
            "--exp_subdir", folder,
            "--planning_mode", "actions",
            "--dual_arm"
        ]

        if check_experiment_needed(folder):
            for s in tqdm(seeds, f"Running the experiment \'{name}\' with all seeds"):
                if check_seed_needed(folder, s):
                    run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser, seed=s)
                    gc.collect()
                    print(f'Finished experiment with seed {s}')
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

def generate_seeds_for_experiment(n=SEED_AMOUNT):
    seeds = []
    for i in range(n):
        seeds.append(random.randint(0, 10 ** 6 - 1))
    with open(SEED_PATH, "w") as f:
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
    run_all_experiments()
