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


class CrossEntropyModelOutput(AbstractModelOutput):
    """
    Margin for image classification.

    .. math::
        \text{logit}[\text{correct}] - \log\left(\sum_{i \neq \text{correct}}
        \exp(\text{logit}[i])\right)

    Version proposed in 'Understanding Influence Functions
    and Datamodels via Harmonic Analysis'
    """

    def __init__(self, device) -> None:
        super().__init__(device)

    def get_output(self,
                   logits: Iterable[Tensor],
                   labels: Optional[Tensor]) -> Tensor:
        bindex = ch.arange(logits.shape[0]).to(self.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone().to(self.device, non_blocking=False)
        # exp(-inf) = 0
        cloned_logits[bindex, labels] = ch.tensor(-ch.inf).to(self.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()


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
