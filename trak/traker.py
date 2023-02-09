from typing import Iterable, Optional
import torch as ch
from torch.nn.parameter import Parameter
from torch import Tensor
from trak.projectors import BasicProjector, ProjectionType
from trak.reweighters import BasicSingleBlockReweighter
from trak.savers import KeepInRAMSaver
from trak.utils import parameters_to_vector, vectorize_and_ignore_buffers
try:
    from functorch import grad, vmap
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')

class TRAKer():
    def __init__(self,
                 model,
                 proj_dim=10,
                 projector=BasicProjector,
                 proj_type=ProjectionType.normal,
                 proj_seed=0,
                 save_dir: str='./trak_results',
                 device=None,
                 train_set_size=1,
                 grad_dtype=ch.float16,
                #  load_from: Optional[str]=None,
                 ):
        """ Main class for computing TRAK scores.
        See [User guide link here] for detailed examples.

        Parameters
        ----------
        save_dir : str, default='./trak_results'
            Directory to save TRAK scores and intermediate values
            like projected gradients of the train set samples and
            targets.

        Attributes
        ----------

        """
        self.model = model
        self.device = device
        self.grad_dtype = grad_dtype

        self.last_ind = 0

        self.projector = projector(seed=proj_seed,
                                   proj_dim=proj_dim,
                                   grad_dim=parameters_to_vector(self.model.parameters()).numel(),
                                   proj_type=proj_type,
                                   dtype=self.grad_dtype,
                                   device=self.device)
        
        self.train_set_size = train_set_size

        self.model_params = parameters_to_vector(model.parameters())
        self.grad_dim = self.model_params.numel()

        self.saver = KeepInRAMSaver(grads_shape=[train_set_size, proj_dim],
                                    device=self.device)
    
    def featurize(self,
                  out_fn,
                  loss_fn,
                  model,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]]=None,
                  model_id: Optional[int]=0,
                  functional: bool=False) -> Tensor:
        self.model_id = model_id
        if functional:
            func_model, weights, buffers = model
            self._featurize_functional(out_fn, weights, buffers, batch, inds)
            self._get_loss_grad_functional(loss_fn, func_model, weights, buffers, batch, inds)
        else:
            self._featurize_iter(out_fn, model, batch, inds)
            self._get_loss_grad_iter(loss_fn, model, batch, inds)

    def _featurize_functional(self, out_fn, weights, buffers, batch, inds) -> Tensor:
        """
        Using the `vmap` feature of `functorch`
        """

        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        self.record_grads(vectorize_and_ignore_buffers(grads), inds)

    def _featurize_iter(self, out_fn, model, batch, inds, batch_size=None) -> Tensor:
        """Computes per-sample gradients of the model output function
        This method does not leverage vectorization (and is hence much slower than
        `_featurize_vmap`).
        """
        if batch_size is None:
            # assuming batch is an iterable of torch Tensors, each of
            # shape [batch_size, ...]
            batch_size = batch[0].size(0)

        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)
        margin = out_fn(model, *batch)

        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               model.parameters(),
                                                               retain_graph=True))

        self.record_grads(grads, inds)

    def record_grads(self, grads, inds):
        grads = self.projector.project(grads.to(self.grad_dtype))
        self.saver.grad_set(grads=grads.detach().clone(), inds=inds, model_id=self.model_id)
    
    def _get_loss_grad_functional(self, loss_fn, func_model,
                                  weights, buffers, batch, inds):
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        self.saver.loss_set(loss_grads=loss_fn(weights, buffers, *batch),
                            inds=inds,
                            model_id=self.model_id)

    def _get_loss_grad_iter(self, loss_fn, model, batch, inds):
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        self.saver.loss_set(loss_grads=loss_fn(model, *batch),
                            inds=inds,
                            model_id=self.model_id)

    def finalize(self):
        self.reweighter = BasicSingleBlockReweighter(device=self.device)
        self.features = ch.zeros_like(self.saver.grad_get())
        for model_id in self.saver.model_ids:
            xtx = self.reweighter.reweight(self.saver.grad_get())
            g = self.saver.grad_get(model_id=model_id)
            self.features += self.reweighter.finalize(g, xtx) * \
                             self.saver.loss_get(model_id=model_id)

    def score(self, out_fn, model, batch, functional=True) -> Tensor:
        if functional:
            return self._score_functional(out_fn, model, batch)
        else:
            return self._score_iter(out_fn, model, batch)

    def _score_functional(self, out_fn, model, batch) -> Tensor:
        _, weights, buffers = model
        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        grads = self.projector.project(vectorize_and_ignore_buffers(grads).to(self.grad_dtype))
        return grads.detach().clone()

    def _score_iter(self, out_fn, model, batch, batch_size=None) -> Tensor:
        if batch_size is None:
            # assuming batch is an iterable of torch Tensors, each of shape [batch_size, ...]
            batch_size = batch[0].size(0)

        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)
        margin = out_fn(model, *batch)

        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               model.parameters(),
                                                               retain_graph=True))

        grads = self.projector.project(grads.to(self.grad_dtype))
        return grads.detach().clone()