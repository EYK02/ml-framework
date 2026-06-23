class BaseAttack:
    def perturb(self, model, x, y):
        """Return adversarially perturbed x"""
        return x