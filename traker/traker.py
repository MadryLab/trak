from typing import Iterable, Optional
from pathlib import Path
import torch as ch
from torch import Tensor
from .projectors import BasicProjector, ProjectionType
from .reweighters import BasicReweighter
from .gradient_computers import FunctionalGradientComputer, IterativeGradientComputer
from .scorers import FunctionalScorer, IterScorer
from .savers import KeepInRAMSaver
from .utils import parameters_to_vector, AverageMeter

class TRAKer():
    def __init__(self,
                 model,
                 grad_wrt=None,
                 projector=None,
                 proj_dim=1000,
                 proj_type=ProjectionType.rademacher,
                 proj_seed=0,
                 proj_num_blocks=1,
                 save_dir: str='/tmp/trak_results',
                 device=None,
                 train_set_size=1,
                 grad_dtype=ch.float32,
                 functional=True,
                 ):
        """ Main class for computing TRAK scores.
        See [User guide link here] for detailed examples.

        Parameters
        ----------
        save_dir : str, default='./trak_results'
            Directory to save TRAK scores and intermediate values
            like projected gradients of the train set samples and
            targets.

        you can  either
        1) specify an already initialized `projector`
        2) specify `proj_dim`, `proj_type`, and `proj_seed` and
           a projector will be initialized for you

        Attributes
        ----------

        """
        self.model = model
        self.grad_wrt = grad_wrt
        self.grad_dtype = grad_dtype
        self.device = device
        self.functional = functional
        self.train_set_size = train_set_size

        self.params_dict = [x[0] for x in list(self.model.named_parameters())]
        if self.grad_wrt is not None:
            self.grad_dim = parameters_to_vector(self.grad_wrt).numel()
        else:
            self.grad_dim = parameters_to_vector(self.model.parameters()).numel()
        self.model_params = {}

        if projector is None:
            projector = BasicProjector
            self.projector = projector(seed=proj_seed,
                                       proj_dim=proj_dim,
                                       grad_dim=self.grad_dim,
                                       proj_type=proj_type,
                                       num_blocks=proj_num_blocks,
                                       dtype=self.grad_dtype,
                                       device=self.device)
        else:
            self.projector = projector
        


        self.save_dir = Path(save_dir)
        self.saver = KeepInRAMSaver(grads_shape=[train_set_size, proj_dim],
                                    save_dir=self.save_dir,
                                    device=self.device)
        
        if self.functional:
            self.gradient_computer = FunctionalGradientComputer(func_model=self.model,
                                                                device=self.device,
                                                                params_dict=self.params_dict)
            self.scorer = FunctionalScorer(device=self.device,
                                           projector=self.projector,
                                           grad_dtype=self.grad_dtype)
        else:
            self.gradient_computer = IterativeGradientComputer(model=self.model,
                                                               device=self.device,
                                                               grad_dim=self.grad_dim)
            self.scorer = IterScorer(device=self.device,
                                     projector=self.projector,
                                     grad_dtype=self.grad_dtype,
                                     grad_dim=self.grad_dim,
                                     grad_wrt=self.grad_wrt)

        self.features = {}
        self.loss_grads = AverageMeter()
    
    def load_params(self, model_params, model_id:int = 0):
        self.model_params[model_id] = model_params
        self.gradient_computer.model_params[model_id] = model_params

    def featurize(self,
                  out_fn,
                  loss_fn,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]]=None,
                  model_id: Optional[int]=0,
                  ) -> Tensor:
        """
        """
        grads = self.gradient_computer.compute_per_sample_grad(out_fn=out_fn,
                                                               batch=batch,
                                                               grad_wrt=self.grad_wrt,
                                                               model_id=model_id)

        grads = self.projector.project(grads.to(self.grad_dtype), model_id=model_id)
        self.saver.grad_set(grads=grads.detach().clone(), inds=inds, model_id=model_id)

        loss_grads = self.gradient_computer.compute_loss_grad(loss_fn=loss_fn,
                                                              batch=batch,
                                                              model_id=model_id)
        self.saver.loss_set(loss_grads=loss_grads, inds=inds, model_id=model_id)

    def finalize(self):
        self.reweighter = BasicReweighter(device=self.device)
        for model_id in self.saver.model_ids:
            self.loss_grads.update(self.saver.loss_get(model_id=model_id))

        for model_id in self.saver.model_ids:
            xtx = self.reweighter.reweight(self.saver.grad_get(model_id=model_id))
            g = self.saver.grad_get(model_id=model_id)
            features = self.reweighter.finalize(g, xtx) * self.loss_grads.avg
            self.saver.features_set(features=features, model_id=model_id)

    def score(self, out_fn, model, batch, model_id=0) -> Tensor:
        return self.scorer.score(self.saver.features_get(model_id=model_id),
                                 out_fn,
                                 model,
                                 model_id,
                                 batch)

    def save(self, features_only=False):
        self.saver.save(features_only=features_only)
        
    def load(self, path=None, features_only=True):
        self.saver.load(load_from=path, features_only=features_only)