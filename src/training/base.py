from src.core.base import BaseTrainer


class DummyTrainer(BaseTrainer):
    def train(self, ctx):
        print("Training would happen here (not implemented yet)")