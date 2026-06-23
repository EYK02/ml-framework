from pathlib import Path
from src.core.runtime import RunContext


class Experiment:
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.ctx = None

    def build(self):
        # dataset
        dataset_cls = self.registry.datasets[self.config["dataset"]]
        dataset = dataset_cls()

        # model
        model_cls = self.registry.models[self.config["model"]]
        model = model_cls()

        # trainer
        trainer_cls = self.registry.trainers[self.config["trainer"]]
        trainer = trainer_cls()

        # attack (optional)
        attack = None
        if "attack" in self.config:
            attack_cls = self.registry.attacks[self.config["attack"]["name"]]
            attack = attack_cls()

        self.ctx = RunContext(
            run_name=self.config["run_name"],
            run_dir=Path(self.config["run_dir"]),
            config=self.config,
            model=model,
            dataset=dataset,
            trainer=trainer,
            attack=attack,
        )

    def run(self):
        self.build()
        self.ctx.summary()
