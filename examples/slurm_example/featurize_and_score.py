from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from argparse import ArgumentParser

from trak import TRAKer


def main(model_id):
    # replace this with your dataset, model, and checkpoints of choice
    # ==================================
    model = models.resnet18(weights='DEFAULT').cuda()
    model.eval()
    ds_train = datasets.CIFAR10(root='/tmp',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor())
    loader_train = DataLoader(ds_train, batch_size=100, shuffle=False)

    ds_val = datasets.CIFAR10(root='/tmp',
                              download=True,
                              train=False,
                              transform=transforms.ToTensor())
    loader_val = DataLoader(ds_val, batch_size=100, shuffle=False)

    # use model_id here to load the proper checkpoint
    # e.g. ckpt = torch.load(f'/path/to/checkpoints/ckpt_{model_id}.pt')
    ckpt = model.state_dict()
    # ==================================

    traker = TRAKer(model=model,
                    task='image_classification',
                    save_dir='./slurm_example_results',
                    train_set_size=len(ds_train),
                    device='cuda')

    traker.load_checkpoint(ckpt, model_id=model_id)
    for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=loader_train.batch_size)
    traker.finalize_features(model_ids=[model_id])

    traker.start_scoring_checkpoint(ckpt, model_id, num_targets=len(loader_val.dataset))
    for batch in tqdm(loader_val, desc='Scoring...'):
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=loader_val.batch_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True, type=int)
    args = parser.parse_args()
    main(args.model_id)
