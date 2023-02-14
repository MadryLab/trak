from typing import Iterable, Optional
from pathlib import Path
import torch as ch
from torch import Tensor
from trak.projectors import BasicProjector, ProjectionType
# from trak.reweighters import BasicReweighter, BasicSingleBlockReweighter
from trak.reweighters import BasicSingleBlockReweighter
from trak.savers import KeepInRAMSaver
from trak.utils import parameters_to_vector, vectorize_and_ignore_buffers, AverageMeter
BasicReweighter = BasicSingleBlockReweighter
try:
    from functorch import grad, vmap, make_functional_with_buffers
except ImportError:
    print('Cannot import `functorch`. Functional mode cannot be used.')

class TRAKer():
    def __init__(self,
                 model,
                 grad_wrt=None,
                 projector=None,
                 proj_dim=10,
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

        you can  either
        1) specify an already initialized `projector`
        2) specify `proj_dim`, `proj_type`, and `proj_seed` and
           a projector will be initialized for you

        Attributes
        ----------

        """
        self.model = model
        if grad_wrt is None:
            self.grad_wrt = list(self.model.parameters())
        else:
            self.grad_wrt = grad_wrt
        self.grad_dtype = grad_dtype
        self.device = device

        self.last_ind = 0
        self.model_params = parameters_to_vector(self.grad_wrt)
        self.grad_dim = self.model_params.numel()

        if projector is None:
            projector = BasicProjector
            self.projector = projector(seed=proj_seed,
                                    proj_dim=proj_dim,
                                    grad_dim=self.grad_dim,
                                    proj_type=proj_type,
                                    dtype=self.grad_dtype,
                                    device=self.device)
        else:
            self.projector = projector
        
        self.params_dict = [x[0] for x in list(self.model.named_parameters())]

        self.train_set_size = train_set_size

        self.save_dir = Path(save_dir)
        self.saver = KeepInRAMSaver(grads_shape=[train_set_size, proj_dim],
                                    save_dir=self.save_dir,
                                    device=self.device)
        

        self.features = {}
        self.loss_grads = AverageMeter()
    
    def featurize(self,
                  out_fn,
                  loss_fn,
                  model,
                  batch: Iterable[Tensor],
                  inds: Optional[Iterable[int]]=None,
                  model_id: Optional[int]=0,
                  functional: bool=False) -> Tensor:
        if functional:
            _, weights, buffers = make_functional_with_buffers(model)
            self._featurize_functional(out_fn,
                                       weights,
                                       buffers,
                                       batch,
                                       model_id,
                                       inds)
            self._get_loss_grad_functional(loss_fn,
                                           weights,
                                           buffers,
                                           batch,
                                           model_id,
                                           inds)
        else:
            self._featurize_iter(out_fn,
                                 model,
                                 batch,
                                 model_id,
                                 inds)
            self._get_loss_grad_iter(loss_fn,
                                     model,
                                     batch,
                                     model_id,
                                     inds)

    def _featurize_functional(self, out_fn, weights, buffers, batch, model_id, inds) -> Tensor:
        """
        Using the `vmap` feature of `functorch`
        """

        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        self.record_grads(vectorize_and_ignore_buffers(grads), model_id, inds)

    def _featurize_iter(self, out_fn, model, batch, model_id, inds, batch_size=None) -> Tensor:
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
                                                               self.grad_wrt,
                                                               retain_graph=True))

        self.record_grads(grads, model_id, inds)

    def record_grads(self, grads, model_id, inds):
        grads = self.projector.project(grads.to(self.grad_dtype), model_id=model_id)
        # print('grads', grads)
        self.saver.grad_set(grads=grads.detach().clone(), inds=inds, model_id=model_id)
    
    def _get_loss_grad_functional(self, loss_fn, weights, buffers, batch, model_id, inds):
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        self.saver.loss_set(loss_grads=loss_fn(weights, buffers, *batch),
                            inds=inds,
                            model_id=model_id)

    def _get_loss_grad_iter(self, loss_fn, model, batch, model_id, inds):
        """Computes
        .. math::
            \partial \ell / \partial \text{margin}
        
        """
        self.saver.loss_set(loss_grads=loss_fn(model, *batch),
                            inds=inds,
                            model_id=model_id)

    def finalize(self):
        self.reweighter = BasicReweighter(device=self.device)
        for model_id in self.saver.model_ids:
            self.loss_grads.update(self.saver.loss_get(model_id=model_id))

        for model_id in self.saver.model_ids:
            xtx = self.reweighter.reweight(self.saver.grad_get(model_id=model_id))
            g = self.saver.grad_get(model_id=model_id)
            self.features[model_id] = self.reweighter.finalize(g, xtx) * self.loss_grads.avg

    def score(self, out_fn, model, batch, model_id=0, functional=True) -> Tensor:
        if functional:
            return self._score_functional(out_fn, model, model_id, batch)
        else:
            return self._score_iter(out_fn, model, model_id, batch)

    def _score_functional(self, out_fn, model, model_id, batch) -> Tensor:
        _, weights, buffers = make_functional_with_buffers(model)
        grads_loss = grad(out_fn, has_aux=False)
        # map over batch dimension
        grads = vmap(grads_loss,
                     in_dims=(None, None, *([0] * len(batch))),
                     randomness='different')(weights, buffers, *batch)
        grads = self.projector.project(vectorize_and_ignore_buffers(grads).to(self.grad_dtype),
                                       model_id=model_id)

        # print(self.features[model_id].T.shape, self.features[model_id].T)
        return grads.detach().clone() @ self.features[model_id].T

    def _score_iter(self, out_fn, model, model_id, batch, batch_size=None) -> Tensor:
        if batch_size is None:
            # assuming batch is an iterable of torch Tensors, each of shape [batch_size, ...]
            batch_size = batch[0].size(0)

        grads = ch.zeros(batch_size, self.grad_dim).to(self.device)
        margin = out_fn(model, *batch)

        for ind in range(batch_size):
            grads[ind] = parameters_to_vector(ch.autograd.grad(margin[ind],
                                                               self.grad_wrt,
                                                               retain_graph=True))

        grads = self.projector.project(grads.to(self.grad_dtype), model_id=model_id)
        return grads.detach().clone() @ self.features[model_id].T
    
    def save(self):
        self.saver.save(self.features)
        
    def load(self, path):
        self.features = self.saver.load(path)