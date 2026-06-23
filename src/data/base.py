from src.core.base import BaseDataset


class DummyDataset(BaseDataset):
    def get_train(self): pass
    def get_val(self): pass
    def get_test(self): pass