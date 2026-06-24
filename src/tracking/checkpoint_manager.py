from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def save_final(self, model):
        path = self.run_dir / "final_model.pt"

        torch.save(
            model.state_dict(),
            path,
        )

    def save_best(self, model):
        path = self.run_dir / "best_model.pt"

        torch.save(
            model.state_dict(),
            path,
        )

    def load(self, model, checkpoint_path):
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(state_dict)

        return model