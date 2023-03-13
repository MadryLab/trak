"""
Here we provide an abstract "model output" class AbstractModelOutput, together with a number
of subclasses for particular applications (vision, language, etc):
- ImageClassificationModelOutput
- IterImageClassificationModelOutput
- CLIPModelOutput

These classes implement methods that transform input batches to the desired model output
(e.g. logits, loss, etc).
See Sections 2 & 3 of (TODO: link)[our paper] for more details on what model output functions are
in the context of TRAK and how to use & design them.

See [TODO: docs link] for examples on how to subclass AbstractModelOutput for a task of your choice.
"""
from abc import ABC, abstractmethod
from typing import Iterable
from torch import Tensor
from torch.nn import Module
import torch as ch


class AbstractModelOutput(ABC):
    """
    Subclasses must implement:
    - a `get_output` method that takes in a batch of inputs and model weights to
    produce outputs that TRAK will be trained to predict.

    - a `get_out_to_loss_grad` method that takes in a batch of inputs and model
    weights to produce the gradient of the function that transforms the model
    outputs above into the loss wrt the batch
    """
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_output(self,
                   model,
                   batch: Iterable[Tensor]) -> Tensor:
        """ See Sections 2 & 3 of (TODO: link)[our paper] for more details on
        what model output functions are in the context of TRAK and how to use &
        design them.

        Args:
            model (_type_): _description_
            batch (Iterable[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        ...

    @abstractmethod
    def get_out_to_loss_grad(self,
                             model,
                             batch: Iterable[Tensor]) -> Tensor:
        ...


class ImageClassificationModelOutput(AbstractModelOutput):
    """
    Margin for (multiclass) image classification. See Section 3.3 of (TODO: link)[our paper]
    for more details.
    """

    def __init__(self, temperature: float = 1.) -> None:
        """
        Args:
            temperature (float, optional): Temperature to use inside the
            softmax for the out-to-loss function. Defaults to 1.
        """
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    @staticmethod
    def get_output(func_model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor) -> Tensor:
        """ For a given output :math:`z=(x, y)` and model parameters :math:`\theta`,
        let :math:`p(z, \theta)` be the softmax probability of the correct class.
        This method implements the model output function
        .. math::
            \log(\frac{p(z, \theta)}{1 - p(z, \theta)}).

        It uses functorch's functional models to make the per-sample gradient computations faster.
        For more details on what func_model, weights & buffers are, and how to use them, please
        refer to https://pytorch.org/functorch/stable/ and
        https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Args:
            func_model (_type_): functorch functional model
            weights (Iterable[Tensor]): functorch model weights
            buffers (Iterable[Tensor]): functorch model buffers
            image (Tensor): input image, should not have batch dimension
            label (Tensor): input label, should not have batch dimension

        Returns:
            Tensor: model output for the given image-label pair :math:`z` and
            weights & buffers :math:`\theta`.
        """
        logits = func_model(weights, buffers, image.unsqueeze(0))
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        # a hacky way to remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, label.unsqueeze(0)] = ch.tensor(-ch.inf).to(logits.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum()

    def forward(self, model: Module, batch: Iterable[Tensor]) -> Tensor:
        """ Utility method to make forward passes compatible across different models,
        e.g. image classification models take in only the images in the batch,
        but CLIP takes in both the images and captions --- using this method,
        the TRAKer class does not need to be modified when we have a new model
        that uses different parts of the input batch.

        Args:
            model (Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: model output (not to be confused with the model output function)
        """
        images, _ = batch
        return model(images)

    def get_out_to_loss_grad(self, func_model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        """ Computes the (reweighting term Q in the paper) 

        Args:
            func_model (_type_): functorch functional model
            weights (Iterable[Tensor]): functorch model weights
            buffers (Iterable[Tensor]): functorch model buffers
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: out-to-loss (reweighting term) for the input batch
        """
        images, labels = batch
        logits = func_model(weights, buffers, images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)


class IterImageClassificationModelOutput(AbstractModelOutput):
    """
    Margin for image classification.

    .. math::
        \text{logit}[\text{correct}] - \log\left(\sum_{i \neq \text{correct}}
        \exp(\text{logit}[i])\right)

    Version of margin proposed in 'Understanding Influence Functions
    and Datamodels via Harmonic Analysis'
    """

    def __init__(self, temperature=1.) -> None:
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    def get_output(self,
                   model: Module,
                   images: Tensor,
                   labels: Tensor) -> Tensor:
        logits = model(images)
        bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        # a hacky way to remove the logits of the correct labels from the sum
        # in logsumexp by setting to -ch.inf
        cloned_logits[bindex, labels] = ch.tensor(-ch.inf).to(logits.device)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins

    def get_out_to_loss_grad(self, model: Module, batch: Iterable[Tensor]) -> Tensor:
        images, labels = batch
        logits = model(images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)


class CLIPModelOutput(AbstractModelOutput):
    num_computed_embeddings = 0
    sim_batch_size = 0
    image_embeddings = None
    text_embeddings = None

    def __init__(self, temperature=1., simulated_batch_size=300) -> None:
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.temperature = temperature

        self.sim_batch_size = simulated_batch_size
        CLIPModelOutput.sim_batch_size = simulated_batch_size

    @staticmethod
    def get_embeddings(model, loader, batch_size, size=50_000, embedding_dim=1024,
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

        CLIPModelOutput.image_embeddings = img_embs
        CLIPModelOutput.text_embeddings = txt_embs
        CLIPModelOutput.num_computed_embeddings = size

    @staticmethod
    def get_output(func_model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor) -> Tensor:
        """
        TODO: proper summary
        - simulating a batch by sampling inds
        - doing a smooth min with -logsumexp(-x)
        """
        all_im_embs = CLIPModelOutput.image_embeddings
        all_txt_embs = CLIPModelOutput.text_embeddings
        N = CLIPModelOutput.num_computed_embeddings
        sim_bs = CLIPModelOutput.sim_batch_size

        if all_im_embs is None:
            raise AssertionError('Run traker.modelout_fn.get_embeddings first before featurizing!')

        image_embeddings, text_embeddings, _ = func_model(weights,
                                                          buffers,
                                                          image.unsqueeze(0),
                                                          label.unsqueeze(0))

        ii = ch.multinomial(input=ch.arange(N).float(),
                            num_samples=sim_bs,
                            replacement=False)

        result = -ch.logsumexp(-image_embeddings @ (text_embeddings - all_txt_embs[ii]).T, dim=1) +\
                 -ch.logsumexp(-text_embeddings @ (image_embeddings - all_im_embs[ii]).T, dim=1)
        return result.sum()  # shape of result should be [1]

    def get_out_to_loss_grad(self, func_model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        image_embeddings, text_embeddings, _ = func_model(weights, buffers, *batch)
        res = self.temperature * image_embeddings @ text_embeddings.T
        ps = (self.softmax(res) + self.softmax(res.T)).diag() / 2.
        return (1 - ps).clone().detach()


TASK_TO_MODELOUT = {
    ('image_classification', True): ImageClassificationModelOutput,
    ('image_classification', False): IterImageClassificationModelOutput,
    ('clip', True): CLIPModelOutput,
}
