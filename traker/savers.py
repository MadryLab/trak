from abc import ABC, abstractmethod
import os
import json
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

        self.model_ids = {}
        # check if there are existing model ids in the save_dir
        self.model_ids_file = self.save_dir.joinpath('ids.json')
        if self.model_ids_file.is_file():
            with open(self.model_ids_file, 'r') as f:
                existing_ids = json.load(f)
                # ids are converted to strings during serialization
                existing_ids = {int(model_id): value for model_id, value in existing_ids.items()}
            self.model_ids.update(existing_ids)

        # init ids metadata
        else:
            with open(self.model_ids_file, 'w+') as f:
                json.dump(self.model_ids, f)



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
        self.current_target_grads = None

    def register_model_id(self, model_id:int):
        self.current_model_id = model_id

        if self.current_model_id in self.model_ids.keys():
            err_msg = f'model id {self.current_model_id} is already registered. Check {self.save_dir}'
            raise ModelIDException(err_msg)
        self.model_ids[self.current_model_id] = {'featurized': 0,
                                                 'finalized': 0,
                                                 'num_target_grads': 0}

        self.init_store(self.current_model_id)
        with open(self.model_ids_file, 'w+') as f:
            json.dump(self.model_ids, f)
    
    def serialize_model_id_metadata(self):
        # TODO: this will be problematic if we have multiple
        # threads writing the featurized/finalized bits in parallel;
        # we should fix that in the future
        with open(self.model_ids_file, 'w+') as f:
            json.dump(self.model_ids, f)
    
    def init_store(self, model_id) -> None:
        prefix = self.save_dir.joinpath(str(model_id))
        try:
            os.makedirs(prefix)
        except:
            raise ModelIDException(f'model id folder {prefix} already exists')

        self.load_store(model_id, mode='w+')
    
    def load_store(self, model_id, mode='r+') -> None:
        self.current_model_id = model_id
        prefix = self.save_dir.joinpath(str(model_id))

        self.current_grads_path = prefix.joinpath('grads.mmap')
        self.current_grads = open_memmap(filename=self.current_grads_path,
                                         mode=mode,
                                         shape=(self.grad_dim, self.proj_dim),
                                         dtype=np.float32)

        self.current_out_to_loss_path = prefix.joinpath('out_to_loss.mmap')
        self.current_out_to_loss = open_memmap(filename=self.current_out_to_loss_path,
                                               mode=mode,
                                               shape=(self.grad_dim, 1),
                                               dtype=np.float32)

        self.current_features_path = prefix.joinpath('features.mmap')
        self.current_features = open_memmap(filename=self.current_features_path,
                                            mode=mode,
                                            shape=(self.grad_dim, self.proj_dim),
                                            dtype=np.float32)

        self.current_target_grads_dict = {}

        self.current_num_target_grads = self.model_ids[model_id]['num_target_grads']
        if self.current_num_target_grads > 0:
            self.current_target_grads_path = prefix.joinpath('grads_target.mmap')
            self.current_target_grads = open_memmap(filename=self.current_target_grads_path,
                                                    mode=mode,
                                                    shape=(self.current_num_target_grads, self.proj_dim),
                                                    dtype=np.float32)

    def finalize_target_grads(self, model_id):
        """ Go from a indices-to-target-grads dictionary to a torch tensor, and save
        to a mmap
        """
        inds = ch.cat(tuple(self.current_target_grads_dict.keys()))
        _current_target_grads_data = ch.cat(tuple(self.current_target_grads_dict.values()))[inds]

        prefix = self.save_dir.joinpath(str(model_id))
        self.current_target_grads = open_memmap(filename=prefix.joinpath('grads_target.mmap'),
                                                mode='w+',
                                                shape=(inds.shape[0], self.proj_dim),
                                                dtype=np.float32)
        self.current_target_grads[:] = _current_target_grads_data
        self.current_target_grads_path = prefix.joinpath('grads_target.mmap')
        self.model_ids[model_id]['num_target_grads'] = len(self.current_target_grads)
     
    def del_grads(self, model_id, target=False):
        if target:
            grads_file = self.save_dir.joinpath(str(model_id)).joinpath('grads_target.mmap')
        else:
            grads_file = self.save_dir.joinpath(str(model_id)).joinpath('grads.mmap')

        # delete grads memmap
        grads_file.unlink()

    def clear_target_grad_count(self, model_id):
        self.model_ids[model_id]['num_target_grads'] = 0
        if model_id == self.current_model_id:
            self.current_num_target_grads = 0
