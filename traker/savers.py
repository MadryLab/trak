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

    @abstractmethod
    def grad_set(self, grads: Tensor) -> None:
        ...

    @abstractmethod
    def grad_get(self, inds: Optional[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def loss_set(self, loss_grads: Tensor) -> None:
        ...

    @abstractmethod
    def loss_get(self, inds: Optional[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def features_set(self, features: Tensor) -> None:
        ...

    @abstractmethod
    def features_get(self, inds: Optional[Tensor]) -> Tensor:
        ...

    @abstractmethod
    def save(self, features_only=True) -> None:
        ...

    @abstractmethod
    def load(self, path, features_only=False) -> Iterable[Tensor]:
        ...


class KeepInRAMSaver(AbstractSaver):
    """ A basic "saver" that does not serialize anything and
    instead keeps all tensors in RAM.
    """
    def __init__(self, device, save_dir, grads_shape) -> None:
        super().__init__(save_dir, device)
        self.grads_shape = grads_shape
        self.loss_shape = [grads_shape[0], 1]
        self.grads = {}
        self.loss_grads = {}
        self.features = {}
        self.model_ids.add(0)
    
    def init_tensor(self, shape, device):
        return ch.zeros(shape, device=device)
    
    def grad_set(self, grads: Tensor, inds: Tensor, model_id=0) -> None:
        if self.grads.get(model_id) is None:
            self.model_ids.add(model_id)
            self.grads[model_id] = self.init_tensor(shape=self.grads_shape, device=self.device)
        self.grads[model_id][inds] = grads

    def grad_get(self, inds: Optional[Tensor]=None, model_id=0) -> Tensor:
        if inds == None:
            return self.grads[model_id]
        else:
            return self.grads[model_id][inds]

    def features_set(self, features: Tensor, inds: Tensor=None, model_id=0) -> None:
        if self.features.get(model_id) is None:
            self.model_ids.add(model_id)
            self.features[model_id] = self.init_tensor(shape=self.grads_shape, device=self.device)
        if inds is not None:
            self.features[model_id][inds] = features
        else:
            self.features[model_id][:] = features

    def features_get(self, inds: Optional[Tensor]=None, model_id=0) -> Tensor:
        if inds == None:
            if self.features.get(model_id) is not None:
                return self.features[model_id]
            else:
                return ch.tensor([])
        else:
            return self.features[model_id][inds]

    def loss_set(self, loss_grads: Tensor, inds: Tensor, model_id=0) -> None:
        if self.loss_grads.get(model_id) is None:
            self.model_ids.add(model_id)
            self.loss_grads[model_id] = self.init_tensor(shape=self.loss_shape, device=self.device)
        self.loss_grads[model_id][inds] = loss_grads.unsqueeze(-1)

    def loss_get(self, inds: Optional[Tensor]=None, model_id=0) -> Tensor:
        if inds == None:
            return self.loss_grads[model_id]
        else:
            return self.loss_grads[model_id][inds]

    def save(self, features_only) -> None:
        for model_id in self.model_ids:
            if not features_only:
                f_grads = self.save_dir.joinpath(f'gradients_{model_id}.npy')
                np.save(f_grads, self.grad_get(model_id=model_id).cpu())
                f_loss = self.save_dir.joinpath(f'loss_grads_{model_id}.npy')
                np.save(f_loss, self.loss_get(model_id=model_id).cpu())
            f_feat = self.save_dir.joinpath(f'features_{model_id}.npy')
            np.save(f_feat, self.features_get(model_id=model_id).cpu())

    def load(self, load_from=None, features_only=False) -> None:
        if load_from is None:
            load_from = self.save_dir
        else:
            load_from = Path(load_from).resolve()
        if not features_only:
            for f in load_from.rglob('gradients_*.npy'):
                model_id = int(str(f).split('.npy')[0].split('_')[-1])
                self.grad_set(ch.from_numpy(np.load(f)).to(self.device), model_id=model_id)
            for f in load_from.rglob('loss_grads_*.npy'):
                model_id = int(str(f).split('.npy')[0].split('_')[-1])
                self.loss_set(ch.from_numpy(np.load(f)).to(self.device), model_id=model_id)
        for f in load_from.rglob('features_*.npy'):
            model_id = int(str(f).split('.npy')[0].split('_')[-1])
            self.features_set(ch.from_numpy(np.load(f)).to(self.device), model_id=model_id)


class ZarrSaver(AbstractSaver):
    def __init__(self, save_dir, device) -> None:
        super().__init__(save_dir, device)
    
    def grad_set(self, grads: Tensor) -> None:
        return super().grad_set(grads)


class MmapSaver(KeepInRAMSaver):
    def __init__(self, device, save_dir, grads_shape) -> None:
        super().__init__(device=device, save_dir=save_dir, grads_shape=grads_shape)
    
    def init_tensor(self, shape, device, name='test.mmap') -> None:
        obj = np.memmap(filename=name, shape=shape)
        return obj.to(device)