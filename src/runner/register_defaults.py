from src.core.registry import registry


@registry.register_dataset("dummy")
class DummyDataset:
    pass


@registry.register_model("dummy")
class DummyModel:
    pass


@registry.register_trainer("dummy")
class DummyTrainer:
    def train(self, ctx):
        print("Dummy training loop executed")


@registry.register_attack("dummy")
class DummyAttack:
    pass
