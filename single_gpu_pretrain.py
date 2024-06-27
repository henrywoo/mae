import argparse
import os
from pathlib import Path

import maskedautoencoder.main_pretrain as trainer


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("MAE pretrain on single GPU", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of GPUs to use (set to 1 for single GPU)")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to use (set to 1 for single GPU)")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="./job_dir", type=str, help="Job directory")
    return parser.parse_args()


def get_shared_folder() -> Path:
    p = Path("./experiments")
    p.mkdir(exist_ok=True)
    return p


def main():
    args = parse_args()
    args.job_dir = Path(args.job_dir)
    args.job_dir.mkdir(parents=True, exist_ok=True)

    # Set up single GPU settings
    args.dist_url = "tcp://127.0.0.1:29500"
    args.output_dir = args.job_dir
    args.gpu = 0
    args.rank = 0
    args.world_size = 1

    trainer.main(args)


if __name__ == "__main__":
    main()
