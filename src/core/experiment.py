from pathlib import Path

from src.core.runtime import RunContext
from src.core.adversarial_wrapper import AdversarialDataWrapper
from src.evaluation.evaluator import Evaluator


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
        evaluator = Evaluator()


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
            evaluator=evaluator,
        )

        self.ctx.train_loader = train_loader
    

    def run(self):
        self.build()

        self.ctx.summary()

        self.ctx.trainer.train(self.ctx)

        clean_results = self.ctx.evaluator.evaluate_clean(self.ctx)

        print("\n=== Evaluation ===")
        print(
            f"Clean Accuracy: "
            f"{clean_results['accuracy']:.2f}%"
        )

        if self.ctx.attack is not None:
            adv_results = (
                self.ctx.evaluator.evaluate_adversarial(self.ctx)
            )

            print(
                f"Adversarial Accuracy: "
                f"{adv_results['accuracy']:.2f}%"
            )

        print("==================")