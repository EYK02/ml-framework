from src.core.registry import registry

from src.data.mnist import MNISTDataset

from src.models.cnn import SimpleCNN

from src.training.standard import StandardTrainer
from src.training.adversarial import AdversarialTrainer

from src.attacks.fgsm import FGSM



# datasets
registry.datasets["mnist"] = MNISTDataset

# models
registry.models["cnn"] = SimpleCNN

# trainers
registry.trainers["standard"] = StandardTrainer
registry.trainers["adversarial"] = AdversarialTrainer

# attacks
registry.attacks["fgsm"] = FGSM