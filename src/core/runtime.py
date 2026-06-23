from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunContext:
    run_name: str
    run_dir: Path
    config: Dict[str, Any]

    model: Any = None
    dataset: Any = None
    trainer: Any = None
    attack: Any = None

    def summary(self):
        print("\n=== RUN CONTEXT ===")
        print(f"Run: {self.run_name}")
        print(f"Dir: {self.run_dir}")
        print(f"Model: {type(self.model).__name__ if self.model else None}")
        print(f"Dataset: {type(self.dataset).__name__ if self.dataset else None}")
        print(f"Trainer: {type(self.trainer).__name__ if self.trainer else None}")
        print(f"Attack: {type(self.attack).__name__ if self.attack else None}")
        print("===================\n")