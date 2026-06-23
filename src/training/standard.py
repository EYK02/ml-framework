import torch
import torch.nn as nn
import torch.optim as optim

from src.core.registry import registry


@registry.register_trainer("standard")
class StandardTrainer:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def train(self, ctx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ctx.model.to(device)
        train_loader = ctx.dataset.get_train()

        optimizer = optim.Adam(
            model.parameters(),
            lr=ctx.config["training"]["lr"]
        )

        epochs = ctx.config["training"]["epochs"]

        model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = model(data)
                loss = self.criterion(output, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch}] loss={avg_loss:.4f}")