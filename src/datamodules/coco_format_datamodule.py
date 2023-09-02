# source: https://github.com/dandelin/ViLT
# modified by Noriyuki Kojima
from src.datasets import *
from .datamodule_base import BaseDataModule

datasets = {
    "touchdown_sdr": TouchdownSDRDataset,
    "f30k_refgame": F30kRefGameDataset,
    "tangram_refgame": TangramRefGameDataset,
}

class CocoFormatDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        assert(len(args[0]["datasets"]) == 1)
        self._dataset = args[0]["datasets"][0]
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return datasets[self._dataset]

    @property
    def dataset_name(self):
        return self._dataset
