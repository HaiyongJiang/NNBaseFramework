import os
import numpy as np
import yaml
from torch.utils import data
from libs.logger import get_logger


class DatasetBase(data.Dataset):
    ''' Example database.
    '''
    def __init__(self, split, cfg):
        self.cfg = cfg
        self.is_train = split == "train"

        dataset_folder = cfg["data"]["path"]
        categories = cfg["data"]["classes"]

        # TODO: iterate through a list of files
        self.flist = [1,] * 8
        if cfg["training"]["overfit"]:
            self.flist = self.flist[:5]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.flist)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        fpath = self.flist[idx] # if idxs is out of range, an exception will be triggered and the iteration stops.
        inp = np.ones((1024, 3), dtype=np.float32) * idx
        outp = np.ones((1024,), dtype=np.long) * (idx % 10)
        return {"input": inp, "output": outp}












if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    print("\n".join(sys.path))
    from libs import config
    cfg = config.load_config('configs/default.yaml')
    ds = DatasetBase("train", cfg)
    for ii, data in enumerate(ds):
        print("ii: %d/%d" % (ii, len(ds)) )
        if ii == 0:
            print(data["input"].shape, data["output"].shape)
        continue


