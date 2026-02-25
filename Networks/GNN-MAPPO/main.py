import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import random
import numpy as np
import torch

from args import Args
from runner import Runner


def main():
    args = Args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**31))
    random.seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    print(
        f"Agents={args.num_agents}  Goals={args.num_goals}  Grid={args.grid_size}x{args.grid_size}\n"
        f"Threads={args.n_rollout_threads}  EpLen={args.episode_length}  "
        f"TotalSteps={args.num_env_steps:,}  "
        f"Episodes≈{args.num_env_steps // args.episode_length // args.n_rollout_threads:,}"
    )

    runner = Runner(args)
    try:
        runner.run()
    finally:
        runner.envs.close()
        print("Done.")


if __name__ == "__main__":
    main()
