"""
TODO: @Andrew
"""
from .modelout_functions import AbstractModelOutput, TASK_TO_MODELOUT
from .projectors import ProjectionType, AbstractProjector, CudaProjector
from .reweighters import BasicReweighter
from .gradient_computers import FunctionalGradientComputer,\
                                AbstractGradientComputer
from .savers import MmapSaver, ModelIDException
from .utils import get_num_params

from typing import Iterable, Optional, Union
from pathlib import Path
from tqdm import tqdm
from torch import Tensor

import numpy as np
import torch
ch = torch


class TRAKer():
    def __init__(self,
                 model: torch.nn.Module,
                 task: Union[AbstractModelOutput, str],
                 train_set_size: int,
                 save_dir: str = './trak_results',
                 projector: Optional[AbstractProjector] = None,
                 device: Union[str, torch.device] = None,
                 gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
                 proj_dim: int = 2048,
                 ) -> None:
        """ Main class for TRAK. See [TODO: add link] for detailed examples.
        TODO: @Andrew

        Either set proj_dim and a CudaProjector Rademacher projector will be
        used or give a custom Projector class and leave proj_dim to None.

        Args:
            model (torch.nn.Module): _description_
            task (Union[AbstractModelOutput, str]): _description_
            train_set_size (int): _description_
            save_dir (str, optional): _description_. Defaults to
                '/tmp/trak_results'.
            projector (Optional[AbstractProjector], optional): _description_.
                Defaults to None.
            device (Union[str, torch.device], optional): _description_.
                Defaults to None.
            gradient_computer (AbstractGradientComputer, optional):
                _description_. Defaults to FunctionalGradientComputer.
            proj_dim (int, optional): _description_. Defaults to 2048.
        """

        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device

        self.num_params = get_num_params(self.model)
        self.init_projector(projector, proj_dim)  # inits self.projector

        self.save_dir = Path(save_dir).resolve()

        if type(self.task) is str:
            self.modelout_fn = TASK_TO_MODELOUT[(self.task, gradient_computer.is_functional)]

        self.gradient_computer = gradient_computer(model=self.model,
                                                   modelout_fn=self.modelout_fn,
                                                   grad_dim=self.num_params)
        metadata = {
            'JL dimension': self.projector.proj_dim,
            'JL matrix type': self.projector.proj_type,
        }
        self.saver = MmapSaver(grads_shape=[self.train_set_size, self.proj_dim],
                               save_dir=self.save_dir,
                               metadata=metadata)

    def init_projector(self, projector, proj_dim) -> None:
        """ Initialize the projector for a traker class

        Args:
            projector (AbstractProjector): _description_
        """

        self.projector = projector
        if projector is not None:
            self.proj_dim = self.projector.proj_dim
        else:
            self.proj_dim = proj_dim
            self.projector = CudaProjector(grad_dim=self.num_params,
                                           proj_dim=self.proj_dim,
                                           seed=0,
                                           proj_type=ProjectionType.rademacher,
                                           device=self.device)

    def load_checkpoint(self,
                        checkpoint: Iterable[Tensor],
                        model_id: int,
                        _allow_featurizing_already_registered=None) -> None:
        """ Loads state dictionary for the given checkpoint, initializes arrays
        to store TRAK features for that checkpoint, tied to the model id.

        Args:
            checkpoint (Iterable[Tensor]): state_dict to load
            model_id (int): a unique ID for a checkpoint
            _allow_featurizing_already_registered (bool, optional): Only use if
            you want to override the default behaviour that `featurize` is
            forbidden on already registered model ids. Defaults to None.
        """
        if self.saver.model_ids.get(model_id) is None:
            self.saver.register_model_id(model_id,
                                         _allow_featurizing_already_registered)
        else:
            self.saver.load_store(model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.gradient_computer.load_model_params(self.model)

        self._last_ind = 0

    def featurize(self,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]] = None,
                  num_samples: Optional[int] = None
                  ) -> Tensor:
        """ Featurizes a batch (TODO: actual summary here)
        TODO: @Andrew

        Either inds or num_samples must be specified. Using num_samples will
        write sequentially into the internal store of the TRAKer.

        Args:
            batch (Iterable[Tensor]): _description_
            inds (Optional[Iterable[int]], optional): _description_. Defaults to None.
            num_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        assert (inds is None) or (num_samples is None),\
            "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None),\
            "Exactly one of num_samples and inds should be specified"

        if num_samples is not None:
            inds = np.arange(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch,
                                                               batch_size=num_samples)
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        self.saver.current_grads[inds] = grads.cpu().clone().detach()

        loss_grads = self.gradient_computer.compute_loss_grad(batch)
        self.saver.current_out_to_loss[inds] = loss_grads.cpu().clone().detach()

        if self.saver.model_ids[self.saver.current_model_id]['featurized'] == 0:
            # TODO: it might be better to set featurized to 1 only after
            # we've featurized all of the train set (as opposed to at the start like
            # we do now)
            self.saver.model_ids[self.saver.current_model_id]['featurized'] = 1
            self.saver.serialize_model_id_metadata(self.saver.current_model_id)

    def finalize_features(self,
                          model_ids: Iterable[int] = None,
                          del_grads: bool = False) -> None:
        """_summary_

        Args:
            model_ids (Iterable[int], optional): _description_. Defaults to None.
        """
        if model_ids is None:
            model_ids = list(self.saver.model_ids.keys())

        self._last_ind = 0

        self.reweighter = BasicReweighter(device=self.device)

        for model_id in tqdm(model_ids, desc='Finalizing features for all model IDs..'):
            if self.saver.model_ids.get(model_id) is None:
                raise ModelIDException(f'Model ID {model_id} not registered, not ready for finalizing.')
            elif self.saver.model_ids[model_id]['finalized'] == 1:
                print(f'Model ID {model_id} already finalized, skipping .finalize_features for it.')
                continue

            self.saver.load_store(model_id)

            g = ch.as_tensor(self.saver.current_grads)
            xtx = self.reweighter.reweight(g)

            self.saver.current_features[:] = self.reweighter.finalize(g, xtx).cpu()
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]['finalized'] = 1
            self.saver.serialize_model_id_metadata(self.saver.current_model_id)

    def start_scoring_checkpoint(self,
                                 checkpoint: Iterable[Tensor],
                                 model_id: int,
                                 num_targets: int,
                                 ) -> None:
        """_summary_

        Args:
            checkpoint (Iterable[Tensor]): _description_
            model_id (int): _description_
            num_targets (int): _description_
        """
        self.saver.load_target_store(model_id, num_targets, mode='w+')

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self._last_ind_target = 0

    def score(self,
              batch: Iterable[Tensor],
              inds: Optional[Iterable[int]] = None,
              num_samples: Optional[int] = None,
              ) -> Tensor:
        """_summary_
        TODO: @Andrew

        Args:
            batch (Iterable[Tensor]): _description_
            inds (Optional[Iterable[int]], optional): _description_. Defaults to None.
            num_samples (Optional[int], optional): _description_. Defaults to None.
            _serialize_target_grads (bool, optional): _description_. Defaults to True.

        Returns:
            Tensor: _description_
        """
        assert (inds is None) or (num_samples is None),\
            "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None),\
            "Exactly one of num_samples and inds should be specified"

        if self.saver.model_ids[self.saver.current_model_id]['finalized'] == 0:
            print(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
            return None

        if num_samples is not None:
            inds = np.arange(self._last_ind_target, self._last_ind_target + num_samples)
            self._last_ind_target += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch,
                                                               batch_size=num_samples)

        grads = self.projector.project(grads, model_id=self.saver.current_model_id)

        self.saver.current_target_grads[inds] = grads.cpu().clone().detach()

    def finalize_scores(self,
                        model_ids: Iterable[int] = None,
                        del_grads: bool = True,
                        exp_name: str = None) -> Tensor:
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids

        _avg_out_to_losses = ch.zeros(self.saver.grad_dim, 1, device=self.device)
        _completed = [False] * len(model_ids)

        for j, model_id in enumerate(tqdm(model_ids, desc='Finalizing scores for all model IDs..')):
            self.saver.load_store(model_id)
            self.saver.load_target_store(model_id, self.saver.num_targets)
            if j == 0:  # during the first pass, create _scores array where avg will be accumulated
                targets_size = self.saver.current_target_grads.shape[0]
                _scores = ch.empty(len(model_ids),
                                   self.train_set_size,
                                   targets_size,
                                   device=self.device)

            if self.saver.model_ids[self.saver.current_model_id]['finalized'] == 0:
                print(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
                continue

            g = ch.as_tensor(self.saver.current_features, device=self.device)
            g_target = ch.as_tensor(self.saver.current_target_grads, device=self.device)

            _scores[j] = g @ g_target.T
            _avg_out_to_losses += ch.as_tensor(self.saver.current_out_to_loss, device=self.device)
            _completed[j] = True

            if del_grads:
                self.saver.del_grads(model_id, target=True)
            else:
                self.saver.clear_target_grad_count(model_id)

        _scores = _scores[_completed].mean(dim=0)

        _num_models_used = sum(_completed)
        self.scores = _scores * (_avg_out_to_losses / _num_models_used)
        self.saver.save_scores(self.scores.cpu().numpy(), exp_name)

        return self.scores
