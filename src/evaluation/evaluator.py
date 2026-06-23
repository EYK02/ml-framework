import torch

from src.evaluation.metrics import accuracy


class Evaluator:
    def evaluate_clean(self, ctx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ctx.model.to(device)
        test_loader = ctx.dataset.get_val()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == target).sum().item()
                total += target.size(0)

        return {
            "accuracy": accuracy(correct, total)
        }

    def evaluate_adversarial(self, ctx):
        if ctx.attack is None:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ctx.model.to(device)
        test_loader = ctx.dataset.get_val()

        model.eval()

        correct = 0
        total = 0

        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            adv_data = ctx.attack.perturb(
                model,
                data,
                target
            )

            with torch.no_grad():
                outputs = model(adv_data)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == target).sum().item()
            total += target.size(0)

        return {
            "accuracy": accuracy(correct, total)
        }