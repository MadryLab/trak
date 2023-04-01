# TRAK Examples

### CIFAR-10
`cifar_quickstart.ipynb` is a complete notebook that trains models from scratch and computes TRAK scores
using the trained checkpoints.
See [tutorial](https://trak.readthedocs.io/en/latest/quickstart.html) for a walk-through.

### ImageNet
`imagenet.py` computes TRAK scores using a pre-trained PyTorch ImageNet model.

### BERT QNLI
`qnli.py` computes TRAK scores for a BERT model.
To run it, you need to supply a checkpoint finetuned on GLUE QNLI.
You can get them by running the `run_glue.py` script from [Hugging Face](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

See [tutorial](https://trak.readthedocs.io/en/latest/bert.html) for a walk-through.
