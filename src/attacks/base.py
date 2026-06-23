from src.core.base import BaseAttack


class DummyAttack(BaseAttack):
    def perturb(self, model, x, y):
        pass
