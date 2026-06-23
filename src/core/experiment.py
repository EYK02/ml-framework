from pathlib import Path

from src.core.runtime import RunContext
from src.core.adversarial_wrapper import AdversarialDataWrapper


class Experiment:
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.ctx = None

    def build(self):
        dataset_cls = self.registry.datasets[self.config["dataset"]]
        model_cls = self.registry.models[self.config["model"]]
        trainer_cls = self.registry.trainers[self.config["trainer"]]

        dataset = dataset_cls()
        model = model_cls()
        trainer = trainer_cls()


        attack = None
        if "attack" in self.config:
            attack_cfg = self.config["attack"]
            attack_cls = self.registry.attacks[attack_cfg["name"]]
            attack = attack_cls(
                **{k: v for k, v in attack_cfg.items() if k != "name"}
            )


        adv_cfg = self.config.get("adversarial", {})
        adv_enabled = adv_cfg.get("enabled", False)

        train_loader = dataset.get_train()

        if adv_enabled:
            train_loader = AdversarialDataWrapper(
                train_loader,
                attack,
                model,
                enabled=True
            )


        self.ctx = RunContext(
            run_name=self.config.get("run_name", "unnamed_run"),
            run_dir=Path(self.config.get("run_dir", "runs/")),
            config=self.config,
            model=model,
            dataset=dataset,
            trainer=trainer,
            attack=attack,
        )

        self.ctx.train_loader = train_loader
    

    def run(self):
        # Build experiment graph
        self.build()

        # Optional: debug print
        self.ctx.summary()

        # Execute training
        self.ctx.trainer.train(self.ctx)