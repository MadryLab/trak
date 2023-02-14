from abc import ABC, abstractmethod
from typing import Iterable, Optional
from torch import Tensor
from torch.nn import Module
import torch as ch


class AbstractModelOutput(ABC):
    """
    ModelOutputFunction classes must implement a `get_output` method that takes
    in a batch of inputs and a model to produce "margins". Those margins will be
    used by TRAK: the gradients for produciung the after kernel will be of the
    margin, rather than the loss.  This is especially useful for exponentially
    saturating loss f-ns like cross entropy.
    """
    @abstractmethod
    def __init__(self, device) -> None:
        self.device = device

    @abstractmethod
    def get_output(self,
                   model: Module,
                   batch: Iterable[Tensor]) -> Tensor:
        ...
    
    @abstractmethod
    def get_out_to_loss(self,
                        model: Module,
                        batch: Iterable[Tensor]) -> Tensor:
        ...


class CrossEntropyModelOutput(AbstractModelOutput):
    """
    Margin for image classification.

    .. math::
        \text{logit}[\text{correct}] - \log\left(\sum_{i \neq \text{correct}}
        \exp(\text{logit}[i])\right)

    Version proposed in 'Understanding Influence Functions
    and Datamodels via Harmonic Analysis'
    """

    def __init__(self, device, temperature=1.) -> None:
        super().__init__(device)
        self.partial_loss_fn = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    def get_output(self,
                   logits: Iterable[Tensor],
                   labels: Optional[Tensor]) -> Tensor:
        bindex = ch.arange(logits.shape[0]).to(self.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone().to(self.device, non_blocking=False)
        # exp(-inf) = 0
        cloned_logits[bindex, labels] = ch.tensor(-ch.inf).to(self.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins
    
    def get_out_to_loss(self, logits: Module, labels: Iterable[Tensor]) -> Tensor:
        ps = self.partial_loss_fn(logits / self.loss_temperature)[ch.arange(logits.size(0)),
                                                                  labels]
        return (1 - ps).clone().detach()



class NLPModelOutput(AbstractModelOutput):
    """
    Margin for fact-tracing in NLP.

    Logits: [batch size, sequence length, vocab size]
    Labels: [batch size, sequence length]
    """

    def __init__(self, device) -> None:
        super().__init__(device)
        self.inf = ch.tensor(-ch.inf).to(self.device)

    def get_output(self,
                   logits: Iterable[Tensor],
                   labels: Optional[Tensor]) -> Tensor:
        logits_correct = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        cloned_logits = logits.clone()
        inf = ch.tensor(-ch.inf).to(self.device)
        if logits.shape[0] == 1:
            r = ch.arange(logits.shape[1]).to(cloned_logits.device)
            cloned_logits[0, r, labels] = inf
            margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        else:
            cloned_logits.scatter_(-1, labels.unsqueeze(-1), -ch.inf)
            margins = logits_correct - cloned_logits.logsumexp(dim=-1)

        return (margins * (labels != 0)).sum(-1) / (labels != 0).sum(-1)


class CLIPModelOutput(AbstractModelOutput):
    def __init__(self, device, temperature=1., simulated_batch_size=50) -> None:
        super().__init__(device)
        self.partial_loss_fn = ch.nn.Softmax(-1)
        self.sim_batch_size = simulated_batch_size
        self.temperature = temperature
    
    def get_embeddings(self, model, loader, size=50_000, embedding_dim=1024,
                       preprocess_fn_img=None, preprocess_fn_txt=None):
        img_embs, txt_embs = ch.zeros(size, embedding_dim).cuda(),\
                             ch.zeros(size, embedding_dim).cuda()
        
        with ch.no_grad():
            for ind, (images, text) in enumerate(loader):
                if preprocess_fn_img is not None:
                    images = preprocess_fn_img(images)
                if preprocess_fn_txt is not None:
                    text = preprocess_fn_txt(text)
                image_embeddings, text_embeddings, _ = model(images, text)
                img_embs[ind] = image_embeddings.mean(dim=0).clone().detach()
                txt_embs[ind] += text_embeddings.mean(dim=0).clone().detach()
                if ind == size - 1:
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