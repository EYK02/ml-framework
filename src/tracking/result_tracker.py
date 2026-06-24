import json
from pathlib import Path

import yaml


class ResultTracker:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_config(self, config):
        path = self.run_dir / "config.yaml"

        with open(path, "w") as f:
            yaml.safe_dump(config, f)

    def save_metrics(self, metrics):
        path = self.run_dir / "metrics.json"

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_summary(self, summary):
        path = self.run_dir / "summary.json"

        with open(path, "w") as f:
            json.dump(summary, f, indent=4)