class Registry:
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.trainers = {}
        self.attacks = {}

    def register_model(self, name):
        def wrapper(cls):
            self.models[name] = cls
            return cls

        return wrapper

    def register_dataset(self, name):
        def wrapper(cls):
            self.datasets[name] = cls
            return cls

        return wrapper

    def register_trainer(self, name):
        def wrapper(cls):
            self.trainers[name] = cls
            return cls

        return wrapper

    def register_attack(self, name):
        def wrapper(cls):
            self.attacks[name] = cls
            return cls

        return wrapper


registry = Registry()
