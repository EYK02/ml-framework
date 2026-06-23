from src.core.registry import registry

from src.data.mnist import MNISTDataset
from src.models.cnn import SimpleCNN
from src.training.standard import StandardTrainer


# datasets
registry.datasets["mnist"] = MNISTDataset

# models
registry.models["cnn"] = SimpleCNN

# trainers
registry.trainers["standard"] = StandardTrainer