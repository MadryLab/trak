"""
TODO
"""
from typing import Iterable, Optional, Union
from pathlib import Path
import numpy as np
import torch
ch = torch
from torch import Tensor
try:
    from functorch import make_functional_with_buffers
except:
    print('functorch could not be imported, running TRAKer in functional mode will not work')

from .modelout_functions import AbstractModelOutput, TASK_TO_MODELOUT
from .projectors import ProjectionType, AbstractProjector, CudaProjector
from .reweighters import BasicReweighter
from .gradient_computers import FunctionalGradientComputer, IterativeGradientComputer
from .savers import MmapSaver
from .utils import get_num_params, get_params_dict

class TRAKer():
    def __init__(self,
                 model: torch.nn.Module,
                 task: Union[AbstractModelOutput, str],
                 train_set_size: int,
                 save_dir: str='/tmp/trak_results',
                 projector: Optional[AbstractProjector]=None,
                 device: Union[str, torch.device]=None,
                 functional: bool=True,
                 proj_dim: int=2048, # either set proj_dim and
                 # a CudaProjector Rademacher projector will be used
                 # or give a custom Projector class and leave proj_dim to None
                #  grad_dtype=ch.float32,
                 ):
        """ Main class for TRAK. See [TODO: add link] for detailed examples.

        Args:
            model (torch.nn.Module): _description_
            task (Union[AbstractModelOutput, str]): _description_
            train_set_size (int): _description_
            save_dir (str, optional): _description_. Defaults to '/tmp/trak_results'.
            projector (Optional[AbstractProjector], optional): _description_. Defaults to None.
            device (Union[str, torch.device], optional): _description_. Defaults to None.
            functional (bool, optional): _description_. Defaults to True.
            proj_dim (int, optional): _description_. Defaults to 2000.
        """
        self.model = model
        self.task = task
        self.train_set_size = train_set_size
        self.functional = functional
        self.device = device

        self.num_params = get_num_params(self.model)
        self.proj_dim = proj_dim
        self.init_projector(projector)

        self.save_dir = Path(save_dir).resolve()
        self.saver = MmapSaver(grads_shape=[self.train_set_size, self.proj_dim],
                               save_dir=self.save_dir,
                               device=self.device)

        if type(self.task) is str:
            self.modelout_fn = TASK_TO_MODELOUT[(self.task, self.functional)]
        
        if self.functional:
            self.func_model, _, _ = make_functional_with_buffers(model)
            self.params_dict = get_params_dict(self.model)
            self.gradient_computer = FunctionalGradientComputer(func_model=self.func_model,
                                                                modelout_fn=self.modelout_fn,
                                                                device=self.device,
                                                                params_dict=self.params_dict)
        else:
            self.gradient_computer = IterativeGradientComputer(model=self.model,
                                                               modelout_fn=self.modelout_fn,
                                                               device=self.device,
                                                               grad_dim=self.grad_dim)

    def init_projector(self, projector):
        """Initialize the projector for a traker class

        Args:
            projector (AbstractProjector): _description_
        """
        self.projector = projector
        if self.projector is None:
            self.projector = CudaProjector(grad_dim=self.num_params,
                                           proj_dim=self.proj_dim,
                                           seed=0,
                                           proj_type=ProjectionType.rademacher,
                                           device=self.device)

    def load_checkpoint(self, checkpoint: Iterable[Tensor], model_id:int,
                        populate_batch_norm_buffers :bool=False, loader_for_bn=None):
        """ Loads state dictionary for the given checkpoint, initializes arrays to store
        TRAK features for that checkpoint, tied to the model id.

        Args:
            checkpoint (Iterable[Tensor]): _description_
            model_id (int): _description_
            populate_batch_norm_buffers (bool, optional): _description_. Defaults to False.
            loader_for_bn (_type_, optional): _description_. Defaults to None.
        """
        self.saver.register_model_id(model_id)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        if populate_batch_norm_buffers:
            with ch.no_grad():
                for batch in loader_for_bn:
                    # TODO: fix
                    self.modelout_fn.forward(model, batch)

        if self.functional:
            self.func_model, self.weights, self.buffers = make_functional_with_buffers(self.model)
        else:
            self.model_params = list(self.model.parameters())

        self._last_ind = 0

    def featurize(self,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]]=None,
                  num_samples: Optional[int]=None
                  ) -> Tensor:
        """ Featurizes a batch (TODO: actual summary here)

        Either inds or num_samples must be specified. Using num_samples will write
        sequentially into the internal store of the TRAKer.

        Args:
            batch (Iterable[Tensor]): _description_
            inds (Optional[Iterable[int]], optional): _description_. Defaults to None.
            num_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        assert (inds is None) or (num_samples is None), "Exactly one of num_samples and inds should be specified"
        assert (inds is not None) or (num_samples is not None), "Exactly one of num_samples and inds should be specified"
        if num_samples is not None:
            inds = np.arange(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples

        grads = self.gradient_computer.compute_per_sample_grad(func_model=self.func_model,
                                                               weights=self.weights,
                                                               buffers=self.buffers,
                                                               batch=batch)
        grads = self.projector.project(grads,
                                       model_id=self.saver.current_model_id)

        self.saver.current_grads[inds] = grads.cpu().clone().detach()

        loss_grads = self.gradient_computer.compute_loss_grad(self.func_model,
                                                              self.weights,
                                                              self.buffers,
                                                              batch)

        self.saver.current_out_to_loss[inds] = loss_grads.cpu().clone().detach()

    def finalize_features(self, model_ids: Iterable[int]=None):
        """_summary_

        Args:
            model_ids (Iterable[int], optional): _description_. Defaults to None.
        """
        if model_ids is None:
            model_ids = self.saver.model_ids

        self.reweighter = BasicReweighter(device=self.device)

        for model_id in self.saver.model_ids:
            self.saver.load_store(model_id)

            g = ch.tensor(self.saver.current_grads)
            xtx = self.reweighter.reweight(g)

            self.saver.current_features = self.reweighter.finalize(g, xtx)

    def score(self, out_fn, model, batch, model_id=0) -> Tensor:
        return self.scorer.score(self.saver.features_get(model_id=model_id),
                                 out_fn,
                                 model,
                                 model_id,
                                 batch)
    
    def finalize_score(self):
        pass