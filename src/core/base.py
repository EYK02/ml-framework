class BaseDataset:
    def get_train(self):
        raise NotImplementedError

    def get_val(self):
        raise NotImplementedError

    def get_test(self):
        raise NotImplementedError


class BaseModel:
    def forward(self, x):
        raise NotImplementedError


class BaseTrainer:
    def train(self, ctx):
        raise NotImplementedError


class BaseAttack:
    def perturb(self, model, x, y):
        raise NotImplementedError
