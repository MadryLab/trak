"""
Here we provide an abstract "model output" class AbstractModelOutput, together with a number
of subclasses for particular applications (vision, language, etc):
- :code:`ImageClassificationModelOutput`
- :code:`IterImageClassificationModelOutput`
- :code:`CLIPModelOutput`

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
    """ See [TODO: docs link] for examples on how to subclass
    :code:`AbstractModelOutput` for a task of your choice.

    Subclasses must implement:
    - a :code:`get_output` method that takes in a batch of inputs and model
    weights to produce outputs that TRAK will be trained to predict. In the
    notation of the paper, :code:`get_output` should return :math:`f(z,\\theta)`

    - a :code:`get_out_to_loss_grad` method that takes in a batch of inputs and
    model weights to produce the gradient of the function that transforms the
    model outputs above into the loss with respect to the batch. In the notation
    of the paper, :code:`get_out_to_loss_grad` returns (entries along the
    diagonal of) :math:`Q`.
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
            model (torch.nn.Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: model output function
        """
        ...

    @abstractmethod
    def get_out_to_loss_grad(self,
                             model,
                             batch: Iterable[Tensor]) -> Tensor:
        """ See Sections 2 & 3 of (TODO: link)[our paper] for more details on
        what the out-to-loss functions (in the notation of the paper, :math:`Q`)
        are in the context of TRAK and how to use & design them.

        Args:
            model (torch.nn.Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: gradient of the out-to-loss function
        """
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
        """ For a given input :math:`z=(x, y)` and model parameters :math:`\\theta`,
        let :math:`p(z, \\theta)` be the softmax probability of the correct class.
        This method implements the model output function
        .. math::
            \\log(\\frac{p(z, \\theta)}{1 - p(z, \\theta)}).

        It uses functorch's functional models to make the per-sample gradient computations faster.
        For more details on what func_model, weights & buffers are, and how to use them, please
        refer to https://pytorch.org/functorch/stable/ and
        https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Note: this method slightly breaks abstraction since it has a different
        signature from the abstract :code:`get_output`. It's only used
        internally by :func:`GradientComputer` so it shouldn't be a big problem.
        If you're reading this and feel strongly about it, feel free to make a
        PR with a fix.

        Args:
            func_model (func): functorch functional model
            weights (Iterable[Tensor]): functorch model weights
            buffers (Iterable[Tensor]): functorch model buffers
            image (Tensor): input image, should not have batch dimension
            label (Tensor): input label, should not have batch dimension

        Returns:
            Tensor: model output for the given image-label pair :math:`z` and
            weights & buffers :math:`\\theta`.
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
            func_model (func): functorch functional model
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
    Margin for (multiclass) image classification. See Section 3.3 of (TODO: link)[our paper]
    for more details.
    """

    def __init__(self, temperature=1.) -> None:
        """
        Args:
            temperature (float, optional): Temperature to use inside the
            softmax for the out-to-loss function. Defaults to 1.
        """
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    def get_output(self,
                   model: Module,
                   images: Tensor,
                   labels: Tensor) -> Tensor:
        """ For a given input :math:`z=(x, y)` and model parameters :math:`\\theta`,
        let :math:`p(z, \\theta)` be the softmax probability of the correct class.
        This method implements the model output function
        .. math::
            \\log(\\frac{p(z, \\theta)}{1 - p(z, \\theta)}).

        Args:
            model (torch.nn.Module): model
            image (Tensor): input images
            label (Tensor): input labels

        Returns:
            Tensor: model output for the given image-label pair :math:`z` and
            model parameters :math:`\theta`.
        """
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
        """ Computes the (reweighting term Q in the paper)

        Args:
            model (torch.nn.Module): model
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: out-to-loss (reweighting term) for the input batch
        """
        images, labels = batch
        logits = model(images)
        # here we are directly implementing the gradient instead of relying on autodiff to do
        # that for us
        ps = self.softmax(logits / self.loss_temperature)[ch.arange(logits.size(0)), labels]
        return (1 - ps).clone().detach().unsqueeze(-1)


class CLIPModelOutput(AbstractModelOutput):
    """ Margin for multimodal contrastive learning (CLIP). See Section 5.1 of
    (TODO: link)[our paper] for more details.

    Raises:
        AssertionError: this model output function requires using additional
        CLIP embeddings, which are computed using the :func:`get_embeddings`
        method. This method should be invoked before featurizing.
    """
    num_computed_embeddings = 0
    sim_batch_size = 0
    image_embeddings = None
    text_embeddings = None

    def __init__(self, temperature: float = None, simulated_batch_size: int = 300) -> None:
        """
        Args:
            temperature (float, optional): Temperature to use inside the
                softmax for the out-to-loss function. If None, CLIP's
                :code:`logit_scale` is used.  Defaults to None
            simulated_batch_size (int, optional): Size of the "simulated" batch
                size for the model output function. See Section 5.1 of the TRAK
                paper for more details. Defaults to 300.
        """
        super().__init__()
        self.softmax = ch.nn.Softmax(-1)
        self.temperature = temperature

        self.sim_batch_size = simulated_batch_size
        CLIPModelOutput.sim_batch_size = simulated_batch_size

    @staticmethod
    def get_embeddings(model,
                       loader,
                       batch_size: int,
                       size: int = 50_000,
                       embedding_dim: int = 1024,
                       preprocess_fn_img=None,
                       preprocess_fn_txt=None) -> None:
        """ Computes (image and text) embeddings and saves them in the class
        attributes :code:`image_embeddings` and :code:`text_embeddings`.

        Args:
            model (torch.nn.Module): model
            loader (): data loader
            batch_size (int): input batch size
            size (int, optional): Maximum number of embeddings to compute. Defaults to 50_000.
            embedding_dim (int, optional): Dimension of CLIP embedding. Defaults to 1024.
            preprocess_fn_img (func, optional): Transforms to apply to images
                from the loader before forward pass. Defaults to None.
            preprocess_fn_txt (func, optional): Transforms to apply to images
                from the loader before forward pass. Defaults to None.
        """
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
        """ For a given input :math:`z=(x, y)` and model parameters :math:`\\theta`,
        let :math:`\\phi(x, \\theta)` be the CLIP image embedding and
        :math:`\\psi(y, \\theta)` be the CLIP text embedding. Last, let
        :math:`B` be a (simulated) batch. This method implements the model
        output function
        .. math::
            -\\log(\\frac{\\phi(x)\\cdot \\psi(y)}{\\sum_{(x', y')\\in B}
            \\phi(x)\\cdot \\psi(y')})
            -\\log(\\frac{\\phi(x)\\cdot \\psi(y)}{\\sum_{(x', y')\\in B}
            \\phi(x')\\cdot \\psi(y)})

        It uses functorch's functional models to make the per-sample gradient
        computations faster.  For more details on what func_model, weights &
        buffers are, and how to use them, please refer to
        https://pytorch.org/functorch/stable/ and
        https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Note: this method slightly breaks abstraction since it has a different
        signature from the abstract :code:`get_output`. It's only used
        internally by :func:`GradientComputer` so it shouldn't be a big problem.
        If you're reading this and feel strongly about it, feel free to make a
        PR with a fix.

        Args:
            func_model (func): functorch functional model
            weights (Iterable[Tensor]): functorch model weights
            buffers (Iterable[Tensor]): functorch model buffers
            image (Tensor): input image, should not have batch dimension
            label (Tensor): input label, should not have batch dimension

        Returns:
            Tensor: model output for the given image-label pair :math:`z` and
            weights & buffers :math:`\\theta`.
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
        """ Computes the (reweighting term Q in the paper)

        Args:
            func_model (func): functorch functional model
            weights (Iterable[Tensor]): functorch model weights
            buffers (Iterable[Tensor]): functorch model buffers
            batch (Iterable[Tensor]): input batch

        Returns:
            Tensor: out-to-loss (reweighting term) for the input batch
        """
        image_embeddings, text_embeddings, temp = func_model(weights, buffers, *batch)
        if self.temperature is None:
            self.temperature = temp
        res = self.temperature * image_embeddings @ text_embeddings.T
        ps = (self.softmax(res) + self.softmax(res.T)).diag() / 2.
        return (1 - ps).clone().detach()


TASK_TO_MODELOUT = {
    ('image_classification', True): ImageClassificationModelOutput,
    ('image_classification', False): IterImageClassificationModelOutput,
    ('clip', True): CLIPModelOutput,
}
