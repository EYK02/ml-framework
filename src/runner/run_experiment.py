import argparse
import yaml

from src.core.experiment import Experiment
from src.core.registry import registry

from src.runner.register_defaults import *


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    exp = Experiment(config, registry)
    exp.run()


if __name__ == "__main__":
    main()