import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="citeseer")

    # data_cons
    parser.add_argument("--feat_flag", type=bool, default=False)
    parser.add_argument("--datasplit_flag", type=bool, default=False)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # seed
    parser.add_argument("--seed", type=int, default=42)

    # LLM
    parser.add_argument("--LLM", type=str, default='llama')
    parser.add_argument("--max_output", type=int, default=128)


    return parser.parse_args()

args = parse_args()
