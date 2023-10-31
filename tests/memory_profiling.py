from tqdm import tqdm
from pathlib import Path
from pytorch_memlab import LineProfiler, MemReporter
from trak import TRAKer
import logging
import torch

from utils import construct_rn9, get_dataloader
from utils import download_cifar_checkpoints, download_cifar_betons

ch = torch


def test_cifar_acc(
    serialize=False, dtype=ch.float32, batch_size=100, tmp_path="/tmp/trak_results/"
):
    device = "cuda:0"
    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    BETONS_PATH = Path(tmp_path).joinpath("cifar_betons")
    BETONS = download_cifar_betons(BETONS_PATH)

    loader_train = get_dataloader(BETONS, batch_size=batch_size, split="train")
    loader_val = get_dataloader(BETONS, batch_size=batch_size, split="val")

    CKPT_PATH = Path(tmp_path).joinpath("cifar_ckpts")
    ckpt_files = download_cifar_checkpoints(CKPT_PATH)
    ckpts = [ch.load(ckpt, map_location="cpu") for ckpt in ckpt_files]

    reporter = MemReporter()

    traker = TRAKer(
        model=model,
        task="image_classification",
        proj_dim=1024,
        train_set_size=10_000,
        save_dir=tmp_path,
        logging_level=logging.DEBUG,
        device=device,
    )

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)

        for batch in tqdm(loader_train, desc="Computing TRAK embeddings..."):
            traker.featurize(batch=batch, num_samples=len(batch[0]))
            reporter.report()

    traker.finalize_features()

    if serialize:
        del traker
        traker = TRAKer(
            model=model,
            task="image_classification",
            proj_dim=1024,
            train_set_size=10_000,
            save_dir=tmp_path,
            device=device,
            logging_level=logging.DEBUG,
        )

    for model_id, ckpt in enumerate(ckpts):
        traker.start_scoring_checkpoint(
            "test_experiment", ckpt, model_id, num_targets=2_000
        )
        for batch in tqdm(loader_val, desc="Scoring..."):
            traker.score(batch=batch, num_samples=len(batch[0]))

    traker.finalize_scores("test_experiment")


with LineProfiler(test_cifar_acc, TRAKer.featurize, TRAKer.load_checkpoint) as prof:
    test_cifar_acc()

prof.print_stats()
