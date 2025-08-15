import random

from pybullet_planning.tutorials.test_vlm_tamp import get_vlm_tamp_agent_parser_given_config
from pybullet_planning.vlm_tools import run_vlm_tamp_with_argparse


def update_parser(conf):
    parser = get_vlm_tamp_agent_parser_given_config(conf)
    parser.add_argument("--rag_recipes", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the Recipe1M+ database should be used in RAG?")
    parser.add_argument("--rag_wikihow", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the WikiHow database should be used in RAG?")
    parser.add_argument("--rag_cutting_vids", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the cutting tutorial videos should be used in RAG?")
    parser.add_argument("--rag_cskg_locations", type=float, choices=[Range(0.0, 1.0)], default=0.5,
                        help="What percentage of the CSKG Locations should be used in RAG?")
    return parser


def run_experiments():
    run_vlm_tamp_with_argparse(get_agent_parser_given_config=update_parser)


def generate_seeds_for_experiment(n=100):
    seeds = []
    for i in range(n):
        seeds.append(random.randint(0, 10 ** 6 - 1))
    with open("../eval_scenarios/seeds.txt", "w") as f:
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
    run_experiments()
