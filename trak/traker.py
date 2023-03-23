from .modelout_functions import AbstractModelOutput, TASK_TO_MODELOUT
from .projectors import ProjectionType, AbstractProjector, CudaProjector
from .gradient_computers import FunctionalGradientComputer,\
                                AbstractGradientComputer
from .score_computers import BasicScoreComputer
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
                 load_from_save_dir: bool = True,
                 device: Union[str, torch.device] = 'cuda',
                 gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
                 projector: Optional[AbstractProjector] = None,
                 proj_dim: int = 2048,
                 ) -> None:
        """ The main front-facing class for TRAK. See the README and docs for
        example usage.

        Args:
            model (torch.nn.Module): model to use for TRAK
            task (Union[AbstractModelOutput, str]): Type of model that TRAK will
                be ran on. Accepts either one of the following strings:
                - :code:`image_classification`
                - :code:`clip`
                - :code:`bert_...` TODO: @Sam
                or an implementation of :func:`AbstractModelOutput`.
            train_set_size (int): Size of the train set that TRAK is featurizing
            save_dir (str, optional): Directory to save final TRAK scores,
                intermediate results, and metadata. Defaults to './trak_results'.
            load_from_save_dir (bool, optional): If True, the TRAKer instance
                will attempt to load existing metadata from save_dir. May lead
                to I/O issues if multiple TRAKer instances ran in parallel have
                this flag set to True. See the SLURM tutorial for more details.
            device (Union[str, torch.device], optional): torch device on which
                to do computations. Defaults to 'cuda'.
            gradient_computer (AbstractGradientComputer, optional):
                Class to use to get per-example gradients. See
                :func:`AbstractGradientComputer` for more details. Defaults to
                FunctionalGradientComputer.
            projector (Optional[AbstractProjector], optional): Either set
                :code:`proj_dim` and a :func:`CudaProjector` Rademacher
                projector will be used or give a custom :code:`Projector` class
                and leave :code:`proj_dim` to None. Defaults to None.
            proj_dim (int, optional): Dimension of the projected TRAK features.
                See Section 4.3 of (TODO: link)[our paper] for more details.
                Defaults to 2048.
        """

        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device

        self.num_params = get_num_params(self.model)
        self.init_projector(projector, proj_dim)  # inits self.projector

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir

        if type(self.task) is str:
            self.modelout_fn = TASK_TO_MODELOUT[(self.task, gradient_computer.is_functional)]

        self.gradient_computer = gradient_computer(model=self.model,
                                                   modelout_fn=self.modelout_fn,
                                                   grad_dim=self.num_params)

        self.score_computer = BasicScoreComputer(device=self.device)

        metadata = {
            'JL dimension': self.projector.proj_dim,
            'JL matrix type': self.projector.proj_type,
        }
        self.saver = MmapSaver(save_dir=self.save_dir,
                               metadata=metadata,
                               train_set_size=self.train_set_size,
                               proj_dim=self.proj_dim,
                               load_from_save_dir=self.load_from_save_dir)

    def init_projector(self, projector, proj_dim) -> None:
        """ Initialize the projector for a traker class

        Args:
            projector (AbstractProjector): JL projector
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
        """ Loads state dictionary for the given checkpoint; initializes arrays
        to store TRAK features for that checkpoint, tied to the model ID.

        Args:
            checkpoint (Iterable[Tensor]): state_dict to load
            model_id (int): a unique ID for a checkpoint
            _allow_featurizing_already_registered (bool, optional): Only use if
                you want to override the default behaviour that
                :code:`featurize` is forbidden on already registered model IDs.
                Defaults to None.
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
        self._num_featurized = 0

    def featurize(self,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]] = None,
                  num_samples: Optional[int] = None
                  ) -> None:
        """ Creates TRAK features for the given batch by computing the gradient
        of the model output function and projecting it. In the notation of the
        paper, for an input pair :math:`z=(x,y)`, model parameters
        :math:`\\theta`, and JL projection matrix :math:`P`, this method
        computes :math:`P^\\top \\nabla_\\theta f(z_i, \\theta)`.
        Additionally, this method computes the gradient of the out-to-loss
        function (in the notation of the paper, the :math:`Q` term in Section
        3.4).

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :func:`TRAKer`.

        Args:
            batch (Iterable[Tensor]): input batch
            inds (Optional[Iterable[int]], optional): Indices of the batch
            samples in the train set. Defaults to None.
            num_samples (Optional[int], optional): Number of samples in the
            batch. Defaults to None.
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
        self._num_featurized += num_samples

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch,
                                                               batch_size=num_samples)
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        self.saver.current_grads[inds] = grads.cpu().clone().detach()

        loss_grads = self.gradient_computer.compute_loss_grad(batch)
        self.saver.current_out_to_loss[inds] = loss_grads.cpu().clone().detach()

        if self._num_featurized == self.train_set_size:
            # TODO: currently this is breaking abstraction -- either define dict
            # items in abstract __init__, or don't access them here
            self.saver.model_ids[self.saver.current_model_id]['featurized'] = 1
            self.saver.serialize_model_id_metadata(self.saver.current_model_id)

    def finalize_features(self,
                          model_ids: Iterable[int] = None,
                          del_grads: bool = False) -> None:
        """ For a set of checkpoints :math:`C` (specified by model IDs), and
        gradients :math:`\\{ \\Phi_c \\}_{c\\in C}`, this method computes
        :math:`\\Phi_c (\\Phi_c^\\top\\Phi_c)^{-1}` for all :math:`c\\in C`
        and stores the results in the internal store of the :func:`TRAKer`
        class.

        Args:
            model_ids (Iterable[int], optional): A list of model IDs for which
                features should be finalized. If None, features are finalized
                for all model IDs in the :code:`save_dir` of the :func:`TRAKer`
                class. Defaults to None.
        """
        if model_ids is None:
            model_ids = list(self.saver.model_ids.keys())

        self._last_ind = 0

        for model_id in tqdm(model_ids, desc='Finalizing features for all model IDs..'):
            if self.saver.model_ids.get(model_id) is None:
                raise ModelIDException(f'Model ID {model_id} not registered, not ready for finalizing.')
            elif self.saver.model_ids[model_id]['finalized'] == 1:
                print(f'Model ID {model_id} already finalized, skipping .finalize_features for it.')
                continue

            self.saver.load_store(model_id)

            g = ch.as_tensor(self.saver.current_grads)
            xtx = self.score_computer.get_xtx(g)

            self.saver.current_features[:] = self.score_computer.get_x_xtx_inv(g, xtx).cpu()
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]['finalized'] = 1
            self.saver.serialize_model_id_metadata(self.saver.current_model_id)

    def start_scoring_checkpoint(self,
                                 checkpoint: Iterable[Tensor],
                                 model_id: int,
                                 num_targets: int,
                                 ) -> None:
        """ This method prepares the internal store of the :func:`TRAKer` class
        to start computing scores for a set of targets.

        Args:
            checkpoint (Iterable[Tensor]): model checkpoint (state dict)
            model_id (int): a unique ID for a checkpoint
            num_targets (int): number of targets to score
        """
        self.saver.load_target_store(model_id, num_targets, mode='w+')

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self._last_ind_target = 0

    def score(self,
              batch: Iterable[Tensor],
              inds: Optional[Iterable[int]] = None,
              num_samples: Optional[int] = None,
              ) -> None:
        """ This method computes the (intermediate per-checkpoint) TRAK scores
        for a batch of targets and stores them in the internal store of the
        :func:`TRAKer` class.

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :func:`TRAKer`.

        Args:
            batch (Iterable[Tensor]): input batch
            inds (Optional[Iterable[int]], optional): Indices of the batch
            samples in the train set. Defaults to None.
            num_samples (Optional[int], optional): Number of samples in the
            batch. Defaults to None.
        """
        assert (inds is None) or (num_samples is None),\
            "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None),\
            "Exactly one of num_samples and inds should be specified"

        # TODO: currently this is breaking abstraction -- either define dict
        # items in abstract __init__, or don't access them here
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
        """ This method computes the final TRAK scores for the given targets,
        train samples, and model checkpoints (specified by model IDs).

        Args:
            model_ids (Iterable[int], optional): A list of model IDs for which
                scores should be finalized. If None, scores are computed
                for all model IDs in the :code:`save_dir` of the :func:`TRAKer`
                class. Defaults to None.
            del_grads (bool, optional): If True, the target gradients
                (intermediate results) are deleted from the internal store of the
                :func:`TRAKer` class.  Defaults to True.
            exp_name (str, optional): Used to name the scores :code:`.npy`
                array produced by this method in the :code:`save_dir` of the
                :func:`TRAKer` class. If None, a random uuid is generated.
                Defaults to None.

        Returns:
            Tensor: TRAK scores
        """
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids

        _completed = [False] * len(model_ids)
        _scores = ch.zeros(self.train_set_size,
                           self.saver.num_targets,
                           device=self.device)
        _avg_out_to_losses = ch.zeros(self.saver.train_set_size, 1, device=self.device)

        for j, model_id in enumerate(tqdm(model_ids, desc='Finalizing scores for all model IDs..')):
            self.saver.load_store(model_id)
            self.saver.load_target_store(model_id, self.saver.num_targets)

            # TODO: currently this is breaking abstraction -- either define dict
            # items in abstract __init__, or don't access them here
            if self.saver.model_ids[self.saver.current_model_id]['finalized'] == 0:
                print(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
                continue

            g = ch.as_tensor(self.saver.current_features, device=self.device)
            g_target = ch.as_tensor(self.saver.current_target_grads, device=self.device)

            _scores += self.score_computer.get_scores(g, g_target)
            _avg_out_to_losses += ch.as_tensor(self.saver.current_out_to_loss, device=self.device)
            _completed[j] = True

            if del_grads:
                self.saver.del_grads(model_id, target=True)
            else:
                self.saver.clear_target_grad_count(model_id)

        _num_models_used = float(sum(_completed))
        self.scores = (_scores / _num_models_used) * (_avg_out_to_losses / _num_models_used)
        self.saver.save_scores(self.scores.cpu().numpy(), exp_name)

        return self.scores
