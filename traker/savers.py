from abc import ABC, abstractmethod
import os
from torch import Tensor
import numpy as np
import torch as ch
from typing import Optional, Iterable
from pathlib import Path

class AbstractSaver(ABC):
    """
    Implementations of Saver class must implement getters and setters
    for `grads` and `loss_grads`, as well as the methods `save` and
    `load`.
    """
    @abstractmethod
    def __init__(self,
                 save_dir,
                 device) -> None:
        self.device = device

        self.save_dir = Path(save_dir).resolve() 
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_ids = set()
        # check if there are existing model ids in the save_dir
        self.model_ids_file = self.save_dir.joinpath('ids.txt')
        if self.model_ids_file.is_file():
            with open(self.model_ids_file, 'r') as f:
                existing_ids = [int(id) for id in f.readlines()]
            self.model_ids.update(existing_ids)

    # TODO: create abstract getters and setters


class MmapSaver(AbstractSaver):
    def __init__(self, device, save_dir, grads_shape) -> None:
        super().__init__(device=device, save_dir=save_dir)
    
    def init_tensor(self, shape, device, name='test.mmap') -> None:
        obj = np.memmap(filename=name, shape=shape)
        return obj.to(device)