from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from torch.nn import Module
import torch as ch

class AbstractModelOutput(ABC):
    """
    ModelOutputFunction classes must implement:
    - a `get_output` method that takes in a batch of inputs and model weights
    to produce outputs that TRAK will be trained to predict.
    - a `get_out_to_loss_grad` method that takes in a batch of inputs and
    model weights to produce the gradient of the function that transforms the
    model outputs above into the loss wrt the batch
    """
    @abstractmethod
    def __init__(self, device) -> None:
        self.device = device

    @abstractmethod
    def get_output(self,
                   model_params: Iterable[Tensor],
                   batch: Iterable[Tensor]) -> Tensor:
        ...
    
    @abstractmethod
    def get_out_to_loss(self,
                        model_params: Iterable[Tensor],
                        batch: Iterable[Tensor]) -> Tensor:
        ...


class ImageClassificationModelOutput(AbstractModelOutput):
    """
    Margin for image classification.

    .. math::
        \text{logit}[\text{correct}] - \log\left(\sum_{i \neq \text{correct}}
        \exp(\text{logit}[i])\right)

    Version of margin proposed in 'Understanding Influence Functions
    and Datamodels via Harmonic Analysis'
    """

    def __init__(self, device, func_model, temperature=1.) -> None:
        super().__init__(device)
        self.softmax = ch.nn.Softmax(-1)
        self.func_model = func_model
        self.loss_temperature = temperature

    def get_output(self,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor) -> Tensor:
        logits = self.func_model(weights, buffers, image.unsqueeze(0))
        bindex = ch.arange(logits.shape[0]).to(self.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone().to(self.device, non_blocking=False)
        # a hacky way to remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf).to(self.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()
    
    def get_out_to_loss_grad(self, model: Module, batch: Iterable[Tensor]) -> Tensor:
        # TODO: fix this method
        images, labels = batch
        logits = model(images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach()


class IterImageClassificationModelOutput(AbstractModelOutput):
    """
    Margin for image classification.

    .. math::
        \text{logit}[\text{correct}] - \log\left(\sum_{i \neq \text{correct}}
        \exp(\text{logit}[i])\right)

    Version of margin proposed in 'Understanding Influence Functions
    and Datamodels via Harmonic Analysis'
    """

    def __init__(self, device, temperature=1.) -> None:
        super().__init__(device)
        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    def get_output(self,
                   model: Module,
                   image: Tensor,
                   label: Tensor) -> Tensor:
        # TODO: fix this method
        logits = model(image.unsqueeze(0))
        bindex = ch.arange(logits.shape[0]).to(self.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone().to(self.device, non_blocking=False)
        # a hacky way to remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf).to(self.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()
    
    def get_out_to_loss_grad(self, model: Module, batch: Iterable[Tensor]) -> Tensor:
        # TODO: fix this method
        images, labels = batch
        logits = model(images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach()


class CLIPModelOutput(AbstractModelOutput):
    def __init__(self, device, temperature=1., simulated_batch_size=300) -> None:
        super().__init__(device)
        self.partial_loss_fn = ch.nn.Softmax(-1)
        self.sim_batch_size = simulated_batch_size
        self.temperature = temperature
    
    def get_embeddings(self, model, loader, batch_size, size=50_000, embedding_dim=1024,
                       preprocess_fn_img=None, preprocess_fn_txt=None):
        img_embs, txt_embs = ch.zeros(size, embedding_dim).cuda(),\
                             ch.zeros(size, embedding_dim).cuda()
        
        cutoff = batch_size
        with ch.no_grad():
            for ind, (images, text) in enumerate(loader):
                if preprocess_fn_img is not None:
                    images = preprocess_fn_img(images)
                if preprocess_fn_txt is not None:
                    text = preprocess_fn_txt(text)
                st, ed = ind * batch_size, min((ind + 1) * batch_size, size)
                if ed == size:
                    cutoff = size - ind * batch_size
                image_embeddings, text_embeddings, _ = model(images, text)
                img_embs[st: ed] = image_embeddings[:cutoff].clone().detach()
                txt_embs[st: ed] = text_embeddings[:cutoff].clone().detach()
                if (ind + 1) * batch_size >= size:
                    break

        self.all_image_fts = img_embs
        self.all_text_fts = txt_embs

    def get_output(self,
                   logits: Iterable[Tensor],
                   labels: Optional[Tensor]) -> Tensor:
        """
        In this case, "logits" are the image embeddings, and "labels" are the
        text embeddings.
        - simulating a batch by sampling inds
        - doing a smooth min with -logsumexp(-x)
        """
        ii = ch.multinomial(input=ch.arange(self.all_image_fts.shape[0]).float(),
                            num_samples=self.sim_batch_size,
                            replacement=False)
        return -ch.logsumexp(-logits @ (labels - self.all_text_fts[ii]).T, dim=1) +\
            -ch.logsumexp(-logits @ (labels - self.all_image_fts[ii]).T, dim=1)


    def get_out_to_loss(self, logits: Module, labels: Iterable[Tensor]) -> Tensor:
        res = self.temperature * logits @ labels.T
        ps = (self.partial_loss_fn(res) + self.partial_loss_fn(res.T)).diag() / 2.
        return (1 - ps).clone().detach()


TASK_TO_MODELOUT = {
    ('image_classification', True): ImageClassificationModelOutput,
    ('image_classification', False): IterImageClassificationModelOutput,
}