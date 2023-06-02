from .modelout_functions import AbstractModelOutput, TASK_TO_MODELOUT
from .projectors import ProjectionType, AbstractProjector, CudaProjector, BasicProjector
from .gradient_computers import FunctionalGradientComputer,\
                                AbstractGradientComputer
from .score_computers import AbstractScoreComputer, BasicScoreComputer
from .savers import AbstractSaver, MmapSaver, ModelIDException
from .utils import get_num_params

from typing import Iterable, Optional, Union
from pathlib import Path
from tqdm import tqdm
from torch import Tensor

import logging
import numpy as np
import torch
ch = torch


class TRAKer():
    """ The main front-facing class for TRAK. See the `README
    <https://github.com/MadryLab/trak>`_ and `docs
    <https://trak.readthedocs.io/en/latest/>`_ for example usage.

    """
    def __init__(self,
                 model: torch.nn.Module,
                 task: Union[AbstractModelOutput, str],
                 train_set_size: int,
                 save_dir: str = './trak_results',
                 load_from_save_dir: bool = True,
                 device: Union[str, torch.device] = 'cuda',
                 gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
                 projector: Optional[AbstractProjector] = None,
                 saver: Optional[AbstractSaver] = None,
                 score_computer: Optional[AbstractScoreComputer] = None,
                 proj_dim: int = 2048,
                 logging_level=logging.INFO,
                 use_half_precision: bool = True,
                 proj_max_batch_size: int = 32,
                 ) -> None:
        """

        Args:
            model (torch.nn.Module):
                model to use for TRAK
            task (Union[AbstractModelOutput, str]):
                Type of model that TRAK will be ran on. Accepts either one of
                the following strings:
                - :code:`image_classification`
                - :code:`text_classification`
                - :code:`clip`
                or an instance of some implementation of the abstract class
                :class:`.AbstractModelOutput`.
            train_set_size (int):
                Size of the train set that TRAK is featurizing
            save_dir (str, optional):
                Directory to save final TRAK scores, intermediate results, and
                metadata. Defaults to :code:'./trak_results'.
            load_from_save_dir (bool, optional):
                If True, the :class`.TRAKer` instance will attempt to load
                existing metadata from save_dir. May lead to I/O issues if
                multiple TRAKer instances ran in parallel have this flag set to
                True. See the SLURM tutorial for more details.
            device (Union[str, torch.device], optional):
                torch device on which to do computations. Defaults to 'cuda'.
            gradient_computer (AbstractGradientComputer, optional):
                Class to use to get per-example gradients. See
                :class:`.AbstractGradientComputer` for more details. Defaults to
                :class:`.FunctionalGradientComputer`.
            projector (Optional[AbstractProjector], optional):
                Either set :code:`proj_dim` and a :class:`.CudaProjector`
                Rademacher projector will be used or give a custom subclass of
                :class:`.AbstractProjector` class and leave :code:`proj_dim` as
                None. Defaults to None.
            saver (Optional[AbstractSaver], optional):
                Class to use for saving intermediate results and final TRAK
                scores to RAM/disk. If None, the :class:`.MmapSaver` will
                be used. Defaults to None.
            score_computer (Optional[AbstractScoreComputer], optional):
                Class to use for computing the final TRAK scores. If None, the
                :class:`.BasicScoreComputer` will be used. Defaults to None.
            proj_dim (int, optional):
                Dimension of the projected TRAK features. See Section 4.3 of
                `our paper <https://arxiv.org/abs/2303.14186>`_ for more
                details. Defaults to 2048.
            logging_level (int, optional):
                Logging level for TRAK loggers. Defaults to logging.INFO.
            use_half_precision (bool, optional):
                If True, TRAK will use half precision (float16) for all
                computations and arrays will be stored in float16. Otherwise, it
                will use float32. Defaults to True.

        """

        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.device = device
        self.dtype = ch.float16 if use_half_precision else ch.float32

        logging.basicConfig()
        self.logger = logging.getLogger('TRAK')
        self.logger.setLevel(logging_level)
        self.logger.warning('TRAK is still in an early 0.x.x version.\n\
                             Report any issues at https://github.com/MadryLab/trak/issues')

        self.num_params = get_num_params(self.model)
        # inits self.projector
        self.init_projector(projector, proj_dim, proj_max_batch_size)

        # normalize to make X^TX numerically stable
        # doing this instead of normalizing the projector matrix
        self.normalize_factor = ch.sqrt(ch.tensor(self.num_params, dtype=ch.float32))

        self.save_dir = Path(save_dir).resolve()
        self.load_from_save_dir = load_from_save_dir

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]()

        self.gradient_computer = gradient_computer(model=self.model,
                                                   task=self.task,
                                                   grad_dim=self.num_params)

        if score_computer is None:
            score_computer = BasicScoreComputer
        self.score_computer = score_computer(dtype=self.dtype,
                                             device=self.device)

        metadata = {
            'JL dimension': self.proj_dim,
            'JL matrix type': self.projector.proj_type,
            'train set size': self.train_set_size,
        }

        if saver is None:
            saver = MmapSaver
        self.saver = saver(save_dir=self.save_dir,
                           metadata=metadata,
                           train_set_size=self.train_set_size,
                           proj_dim=self.proj_dim,
                           load_from_save_dir=self.load_from_save_dir,
                           logging_level=logging_level,
                           use_half_precision=use_half_precision)

        self.ckpt_loaded = 'no ckpt loaded'

    def init_projector(self, projector, proj_dim, proj_max_batch_size) -> None:
        """ Initialize the projector for a traker class

        Args:
            projector (AbstractProjector):
                JL projector

        """

        self.projector = projector
        if projector is not None:
            self.proj_dim = self.projector.proj_dim
            if self.proj_dim == 0:  # using NoOpProjector
                self.proj_dim = self.num_params

        else:
            self.proj_dim = proj_dim
            try:
                import fast_jl
                test_gradient = ch.ones(1, self.num_params).cuda()
                num_sms = ch.cuda.get_device_properties('cuda').multi_processor_count
                fast_jl.project_rademacher_8(test_gradient, self.proj_dim, 0, num_sms)
                projector = CudaProjector

            except (ImportError, RuntimeError, AttributeError) as e:
                self.logger.error(f'Could not use CudaProjector.\nReason: {str(e)}')
                self.logger.error('Defaulting to BasicProjector.')
                projector = BasicProjector

            self.projector = projector(grad_dim=self.num_params,
                                       proj_dim=self.proj_dim,
                                       seed=0,
                                       proj_type=ProjectionType.rademacher,
                                       max_batch_size=proj_max_batch_size,
                                       dtype=self.dtype,
                                       device=self.device)

    def load_checkpoint(self,
                        checkpoint: Iterable[Tensor],
                        model_id: int,
                        _allow_featurizing_already_registered=False) -> None:
        """ Loads state dictionary for the given checkpoint; initializes arrays
        to store TRAK features for that checkpoint, tied to the model ID.

        Args:
            checkpoint (Iterable[Tensor]):
                state_dict to load
            model_id (int):
                a unique ID for a checkpoint
            _allow_featurizing_already_registered (bool, optional):
                Only use if you want to override the default behaviour that
                :code:`featurize` is forbidden on already registered model IDs.
                Defaults to None.

        """
        if self.saver.model_ids.get(model_id) is None:
            self.saver.register_model_id(model_id,
                                         _allow_featurizing_already_registered)
        else:
            self.saver.load_current_store(model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.gradient_computer.load_model_params(self.model)

        self._last_ind = 0
        self.ckpt_loaded = model_id

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
            batch (Iterable[Tensor]):
                input batch
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        assert self.ckpt_loaded == self.saver.current_model_id,\
            "Load a checkpoint using traker.load_checkpoint before featurizing"
        assert (inds is None) or (num_samples is None),\
            "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None),\
            "Exactly one of num_samples and inds should be specified"

        if num_samples is not None:
            inds = np.arange(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        # handle re-starting featurizing from a partially featurized model (some inds already featurized)
        _already_done = (self.saver.current_store['is_featurized'][inds] == 1).reshape(-1)
        inds = inds[~_already_done]
        if len(inds) == 0:
            self.logger.debug('All samples in batch already featurized.')
            return 0

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)
        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor
        self.saver.current_store['grads'][inds] = grads.to(self.dtype).cpu().clone().detach()

        loss_grads = self.gradient_computer.compute_loss_grad(batch)
        self.saver.current_store['out_to_loss'][inds] = loss_grads.to(self.dtype).cpu().clone().detach()

        self.saver.current_store['is_featurized'][inds] = 1
        self.saver.serialize_current_model_id_metadata()

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
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.

        """
        if model_ids is None:
            model_ids = list(self.saver.model_ids.keys())

        self._last_ind = 0

        for model_id in tqdm(model_ids, desc='Finalizing features for all model IDs..'):
            if self.saver.model_ids.get(model_id) is None:
                raise ModelIDException(f'Model ID {model_id} not registered, not ready for finalizing.')
            elif self.saver.model_ids[model_id]['is_featurized'] == 0:
                raise ModelIDException(f'Model ID {model_id} not fully featurized, not ready for finalizing.')
            elif self.saver.model_ids[model_id]['is_finalized'] == 1:
                self.logger.warning(f'Model ID {model_id} already finalized, skipping .finalize_features for it.')
                continue

            self.saver.load_current_store(model_id)

            g = ch.as_tensor(self.saver.current_store['grads'])
            xtx = self.score_computer.get_xtx(g)
            self.logger.debug(f'XTX is {xtx}')

            features = self.score_computer.get_x_xtx_inv(g, xtx)
            self.logger.debug(f'Features are {features}')
            self.saver.current_store['features'][:] = features.to(self.dtype).cpu()
            if del_grads:
                self.saver.del_grads(model_id)

            self.saver.model_ids[self.saver.current_model_id]['is_finalized'] = 1
            self.saver.serialize_current_model_id_metadata()

    def start_scoring_checkpoint(self,
                                 exp_name: str,
                                 checkpoint: Iterable[Tensor],
                                 model_id: int,
                                 num_targets: int,
                                 ) -> None:
        """ This method prepares the internal store of the :class:`.TRAKer` class
        to start computing scores for a set of targets.

        Args:
            exp_name (str):
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            checkpoint (Iterable[Tensor]):
                model checkpoint (state dict)
            model_id (int):
                a unique ID for a checkpoint
            num_targets (int):
                number of targets to score

        """
        self.saver.init_experiment(exp_name, num_targets, model_id)

        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.gradient_computer.load_model_params(self.model)

        # TODO: make this exp_name-dependent
        # e.g. make it a value in self.saver.experiments[exp_name]
        self._last_ind_target = 0

    def score(self,
              batch: Iterable[Tensor],
              inds: Optional[Iterable[int]] = None,
              num_samples: Optional[int] = None,
              ) -> None:
        """ This method computes the (intermediate per-checkpoint) TRAK scores
        for a batch of targets and stores them in the internal store of the
        :class:`.TRAKer` class.

        Either :code:`inds` or :code:`num_samples` must be specified. Using
        :code:`num_samples` will write sequentially into the internal store of
        the :class:`.TRAKer`.

        Args:
            batch (Iterable[Tensor]):
                input batch
            inds (Optional[Iterable[int]], optional):
                Indices of the batch samples in the train set. Defaults to None.
            num_samples (Optional[int], optional):
                Number of samples in the batch. Defaults to None.

        """
        assert (inds is None) or (num_samples is None),\
            "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None),\
            "Exactly one of num_samples and inds should be specified"

        if self.saver.model_ids[self.saver.current_model_id]['is_finalized'] == 0:
            self.logger.error(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
            return None

        if num_samples is not None:
            inds = np.arange(self._last_ind_target, self._last_ind_target + num_samples)
            self._last_ind_target += num_samples
        else:
            num_samples = inds.reshape(-1).shape[0]

        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)

        grads = self.projector.project(grads, model_id=self.saver.current_model_id)
        grads /= self.normalize_factor

        exp_name = self.saver.current_experiment_name
        self.saver.current_store[f'{exp_name}_grads'][inds] = grads.to(self.dtype).cpu().clone().detach()

    def finalize_scores(self,
                        exp_name: str,
                        model_ids: Iterable[int] = None,
                        allow_skip: bool = False,
                        ) -> Tensor:
        """ This method computes the final TRAK scores for the given targets,
        train samples, and model checkpoints (specified by model IDs).

        Args:
            exp_name (str):
                Experiment name. Each experiment should have a unique name, and
                it corresponds to a set of targets being scored. The experiment
                name is used as the name for saving the target features, as well
                as scores produced by this method in the :code:`save_dir` of the
                :class:`.TRAKer` class.
            model_ids (Iterable[int], optional):
                A list of model IDs for which
                scores should be finalized. If None, scores are computed
                for all model IDs in the :code:`save_dir` of the :class:`.TRAKer`
                class. Defaults to None.
            allow_skip (bool, optional):
                If True, raises only a warning, instead of an error, when target
                gradients are not computed for a given model ID. Defaults to
                False.

        Returns:
            Tensor: TRAK scores

        """
        # reset counter for inds used for scoring
        self._last_ind_target = 0

        if model_ids is None:
            model_ids = self.saver.model_ids
        else:
            model_ids = {model_id: self.saver.model_ids[model_id] for model_id in model_ids}
        assert len(model_ids) > 0, 'No model IDs to finalize scores for'

        if self.saver.experiments.get(exp_name) is None:
            raise ValueError(f'Experiment {exp_name} does not exist. Create it\n\
                              and compute scores first before finalizing.')

        num_targets = self.saver.experiments[exp_name]['num_targets']
        _completed = [False] * len(model_ids)

        self.saver.load_current_store(list(model_ids.keys())[0], exp_name, num_targets)
        _scores = self.saver.current_store[f'{exp_name}_scores']
        _scores[:] = 0.

        _avg_out_to_losses = np.zeros((self.saver.train_set_size, 1),
                                      dtype=np.float16 if self.dtype == ch.float16 else np.float32)

        for j, model_id in enumerate(tqdm(model_ids, desc='Finalizing scores for all model IDs..')):
            self.saver.load_current_store(model_id)
            try:
                self.saver.load_current_store(model_id, exp_name, num_targets)
            except OSError as e:
                if allow_skip:
                    self.logger.warning(f'Could not read target gradients for model ID {model_id}. Skipping.')
                    continue
                else:
                    raise e

            if self.saver.model_ids[self.saver.current_model_id]['is_finalized'] == 0:
                self.logger.warning(f'Model ID {self.saver.current_model_id} not finalized, cannot score')
                continue

            g = ch.as_tensor(self.saver.current_store['features'], device=self.device)
            g_target = ch.as_tensor(self.saver.current_store[f'{exp_name}_grads'],
                                    device=self.device)

            _scores[:] += self.score_computer.get_scores(g, g_target).cpu().clone().detach().numpy()

            _avg_out_to_losses += self.saver.current_store['out_to_loss']
            _completed[j] = True

        _num_models_used = float(sum(_completed))
        _scores[:] = (_scores / _num_models_used) * (_avg_out_to_losses / _num_models_used)

        self.logger.dtype(f'Scores dtype is {_scores.dtype}')
        self.saver.save_scores(exp_name)
        self.scores = _scores

        return self.scores
