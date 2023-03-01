from abc import ABC, abstractmethod
import os
from torch import Tensor
import numpy as np
from numpy.lib.format import open_memmap
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
        # init ids metadata
        else:
            with open(self.model_ids_file, 'w+') as f:
                pass


    @abstractmethod
    def register_model_id(self, model_id:int):
        ...


class ModelIDException(Exception):
    pass


class MmapSaver(AbstractSaver):
    def __init__(self, device, save_dir, grads_shape) -> None:
        super().__init__(device=device, save_dir=save_dir)
        self.grad_dim, self.proj_dim = grads_shape
        self.current_model_id = None
        self.current_grads = None
        self.current_out_to_loss = None
        self.current_features = None
    
    def init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        try:
            os.makedirs(prefix)
        except:
            raise ModelIDException(f'model id folder {prefix} already exists')

        self.load_store(model_id, mode='w+')
    
    def load_store(self, model_id, mode='r+') -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        self.current_grads = open_memmap(filename=prefix.joinpath('grads.mmap'),
                                         mode=mode,
                                         shape=(self.grad_dim, self.proj_dim),
                                         dtype=np.float32)

        self.current_out_to_loss = open_memmap(filename=prefix.joinpath('out_to_loss.mmap'),
                                               mode=mode,
                                               shape=(self.grad_dim, 1),
                                               dtype=np.float32)

        self.current_features = open_memmap(filename=prefix.joinpath('features.mmap'),
                                            mode=mode,
                                            shape=(self.grad_dim, self.proj_dim),
                                            dtype=np.float32)
    
    def register_model_id(self, model_id:int):
        self.current_model_id = model_id

        if self.current_model_id in self.model_ids:
            err_msg = f'model id {self.current_model_id} is already registered. Check {self.save_dir}'
            raise ModelIDException(err_msg)
        self.model_ids.add(self.current_model_id)

        self.init_store(self.current_model_id)
        with open(self.model_ids_file, 'a+') as f:
            f.write(str(self.current_model_id) + '\n')