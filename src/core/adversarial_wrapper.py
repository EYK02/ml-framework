class AdversarialDataWrapper:
    def __init__(self, dataloader, attack, model, enabled=False):
        self.dataloader = dataloader
        self.attack = attack
        self.model = model
        self.enabled = enabled

    def __iter__(self):
        for x, y in self.dataloader:
            if self.enabled and self.attack is not None:
                x = self.attack.perturb(self.model, x, y)
            yield x, y

    def __len__(self):
        return len(self.dataloader)