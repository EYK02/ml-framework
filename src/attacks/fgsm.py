import torch
import torch.nn.functional as F

from src.core.registry import registry


@registry.register_attack("fgsm")
class FGSM:
    def __init__(self, eps=0.1):
        self.eps = eps

    def perturb(self, model, x, y):
        """
        Fast Gradient Sign Method (FGSM)
        x: input batch
        y: labels
        """

        # ensure we're not tracking gradients on original tensor
        x_adv = x.detach().clone()
        x_adv.requires_grad = True

        # forward pass
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)

        # backward pass to get gradients wrt input
        model.zero_grad()
        loss.backward()

        # FGSM step
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.eps * grad_sign

        # important: detach so graph is not reused
        return x_adv.detach()