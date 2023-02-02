from typing import Iterable, Optional
import torch as ch
from torch.nn.parameter import Parameter
from torch import Tensor
from trak.model_output_fns import AbstractModelOutput
from trak.projectors import BasicProjector, ProjectionType
from trak.utils import parameters_to_vector
try:
    from functorch import make_functional_with_buffers, grad, vmap
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')

class TRAKer():
    def __init__(self,
                 save_dir: str,
                 model,
                 model_output_fn: AbstractModelOutput,
                 functional: bool = True,
                 proj_seed=0,
                 proj_dim=10,
                 proj_type=ProjectionType.normal,
                 projector=BasicProjector,
                 device=None,
                 load_from_existing: bool = False):
        self.save_dir = save_dir
        self.model = model
        self.model_output_fn = model_output_fn
        self.functional = functional
        self.device = device

        self.last_ind = 0

        if self.functional:
            self.func_model, self.weights, self.buffers = make_functional_with_buffers(model)
        
        self.projector = projector(seed=proj_seed,
                                   proj_dim=proj_dim,
                                   grad_dim=parameters_to_vector(self.model.parameters()).numel(),
                                   proj_type=proj_type,
                                   device=self.device)
    
    def featurize(self,
                  fn,
                  params: Iterable[Parameter],
                  batch: Iterable[Tensor]) -> Tensor:
        if self.functional:
            weights, buffers = params
            self._featurize_functional(fn, weights, buffers, batch)
        else:
            # Make sure param grads are zero
            self._featurize_iter(fn, params, batch)
    
    def _featurize_functional(self, fn, weights, buffers, batch) -> Tensor:
        """
        Using the `vmap` feature of `functorch`
        """
        grads_loss = grad(fn, has_aux=False)
        return vmap(grads_loss,
                    in_dims=(None, None, 0, 0),
                    randomness='different')(weights, buffers, batch)

    def _featurize_iter(fn, params, batch) -> Tensor:
        pass

    def finalize(self, out_dir: Optional[str] = None, 
                       cleanup: bool = False, 
                       agg: bool = False):
        pass

    def score(self, val: Tensor, params: Iterable[Parameter]) -> Tensor:
        return ch.zeros(1, 1)