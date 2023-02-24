import torch as ch
from tqdm import tqdm
from functorch import make_functional_with_buffers
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from traker.traker import TRAKer
from traker.modelout_functions import CrossEntropyModelOutput

def init_model_and_data(device='cuda:0'):
    model = models.resnet18(weights='DEFAULT').to(device)
    model.eval()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds_train = datasets.ImageFolder(root='/mnt/cfs/datasets/pytorch_imagenet/train', transform=transform)
    # taking the first 1000 examples
    ds_train = ch.utils.data.Subset(ds_train, list(range(1_000)))
    loader_train = DataLoader(ds_train, batch_size=20, shuffle=False)

    ds_val= datasets.ImageFolder(root='/mnt/cfs/datasets/pytorch_imagenet/train', transform=transform)
    # computing scores for the first 100 val samples
    ds_val = ch.utils.data.Subset(ds_val, list(range(100)))
    loader_val = DataLoader(ds_val, batch_size=20, shuffle=False)
    return model, loader_train, loader_val

if __name__ == "__main__":
    device = 'cuda:0'
    model, loader_train, loader_val = init_model_and_data(device)

    modelout_fn = CrossEntropyModelOutput(device=device)
    trak = TRAKer(model=model,
                  train_set_size=1000,
                  save_dir='./trak_results',
                  device=device)

    func_model, weights, buffers = make_functional_with_buffers(model)
    trak.load_params(model_params=(weights, buffers))

    def compute_outputs(weights, buffers, image, label):
        out = func_model(weights, buffers, image.unsqueeze(0))
        return modelout_fn.get_output(out, label.unsqueeze(0)).sum()

    def compute_out_to_loss(weights, buffers, images, labels):
        out = func_model(weights, buffers, images)
        return modelout_fn.get_out_to_loss(out, labels)

    for bind, batch in enumerate(tqdm(loader_train, desc='Computing TRAK embeddings...')):
        batch = [x.cuda() for x in batch]
        inds = list(range(bind * loader_train.batch_size,
                        (bind + 1) * loader_train.batch_size))
        trak.featurize(out_fn=compute_outputs,
                       loss_fn=compute_out_to_loss,
                       batch=batch,
                       inds=inds)
    trak.finalize()

    scores = []
    for bind, batch in enumerate(tqdm(loader_val, desc='Scoring...')):
        batch = [x.cuda() for x in batch]
        scores.append(
            trak.score(out_fn=compute_outputs, batch=batch, model=model).cpu()
        )
    scores = ch.cat(scores)
