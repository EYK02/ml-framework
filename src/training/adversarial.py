import torch
import torch.nn as nn

from src.core.registry import registry


@registry.register_trainer("adversarial")
class AdversarialTrainer:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train(self, ctx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ctx.model.to(device)
        train_loader = ctx.train_loader

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=ctx.config["training"]["lr"]
        )

        epochs = ctx.config["training"]["epochs"]

        model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                adv_data = ctx.attack.perturb(
                    model,
                    data,
                    target,
                )

                optimizer.zero_grad()

                output = model(adv_data)

                loss = self.criterion(output, target)

                loss.backward()

                optimizer.step()

            print(f"[ADV Epoch {epoch}] loss={total_loss / len(train_loader):.4f}")