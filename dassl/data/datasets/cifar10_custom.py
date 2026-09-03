import os.path as osp
from dassl.utils import read_json
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class CIFAR10_Custom(DatasetBase):
    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, "cifar10")
        self.split_path = osp.join(self.dataset_dir, "split_zhou_CIFAR10.json")
        
        items = read_json(self.split_path)
        train = self._read_data(items["train"])
        val = self._read_data(items["val"])
        test = self._read_data(items["test"])

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, items):
        return [Datum(impath=osp.join(self.dataset_dir, x[0]), label=x[1], classname=x[2]) for x in items]
