from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch as ch

from trak import TRAKer


def init_model_and_data(device='cuda:0'):
    model = models.resnet18(weights='DEFAULT').to(device)
    model.eval()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds_train = datasets.ImageFolder(root='path/to/imagenet/train', transform=transform)

    # taking the first 1000 examples
    ds_train = ch.utils.data.Subset(ds_train, list(range(1_000)))
    loader_train = DataLoader(ds_train, batch_size=20, shuffle=False)

    ds_val = datasets.ImageFolder(root='path/to/imagenet/val', transform=transform)
    # computing scores for the first 100 val samples
    ds_val = ch.utils.data.Subset(ds_val, list(range(100)))
    loader_val = DataLoader(ds_val, batch_size=20, shuffle=False)
    return model, loader_train, loader_val


if __name__ == "__main__":
    device = 'cuda:0'
    model, loader_train, loader_val = init_model_and_data(device)

    traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=len(loader_train.dataset),
                    save_dir='./trak_results',
                    device=device)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing..'):
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

    traker.start_scoring_checkpoint(model.state_dict(), model_id=0, num_targets=100)
    for batch in tqdm(loader_val, desc='Scoring..'):
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores()
