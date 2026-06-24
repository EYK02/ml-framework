from datetime import datetime
from pathlib import Path

from src.core.runtime import RunContext
from src.evaluation.evaluator import Evaluator
from src.tracking.result_tracker import ResultTracker
from src.tracking.checkpoint_manager import CheckpointManager

from src.utils.reproducibility import set_seed


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

        run_root = Path(
            self.config.get("run_dir", "runs")
        )

        run_name = self.config.get(
            "run_name",
            "unnamed_run"
        )
        
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_path = run_root / f"{timestamp}_{run_name}"

        tracker = ResultTracker(run_path)
        checkpoint_manager = CheckpointManager(run_path)

        self.ctx = RunContext(
            run_name=run_name,
            run_dir=run_path,
            config=self.config,
            model=model,
            dataset=dataset,
            trainer=trainer,
            attack=attack,
            evaluator=evaluator,
            tracker=tracker,
            checkpoint_manager=checkpoint_manager,
        )

        self.ctx.attack = attack
        self.ctx.adversarial_enabled = adv_enabled
        self.ctx.train_loader = train_loader
    

    def run(self):
        set_seed(self.config.get("seed", 42))
        self.build()

        self.ctx.tracker.save_config(
            self.config
        )

        self.ctx.summary()

        self.ctx.trainer.train(self.ctx)

        self.ctx.checkpoint_manager.save_final(
            self.ctx.model
        )

        results = {}

        clean_results = (
            self.ctx.evaluator.evaluate_clean(self.ctx)
        )

        results["clean"] = clean_results

        if self.ctx.attack is not None:
            adv_results = (
                self.ctx.evaluator.evaluate_adversarial(self.ctx)
            )

            results["adversarial"] = adv_results

        self.ctx.tracker.save_metrics(results)

        summary = {
            "run_name": self.ctx.run_name,
            "model": type(self.ctx.model).__name__,
            "trainer": type(self.ctx.trainer).__name__,
            "attack": (
                type(self.ctx.attack).__name__
                if self.ctx.attack is not None
                else None
            ),
        }

        self.ctx.tracker.save_summary(summary)

        print("\n=== Evaluation ===")
        print(
            f"Clean Accuracy: "
            f"{results['clean']['accuracy']:.2f}%"
        )

        if "adversarial" in results:
            print(
                f"Adversarial Accuracy: "
                f"{results['adversarial']['accuracy']:.2f}%"
            )