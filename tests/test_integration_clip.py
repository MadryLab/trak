import pytest
from tqdm import tqdm
from torchvision import datasets
import open_clip

from trak import TRAKer
# from trak.gradient_computers import IterativeGradientComputer


@pytest.mark.cuda
def test_mscoco(tmp_path, device='cuda:0'):
    model, _, preprocess = open_clip.create_model_and_transforms('RN50')
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer('RN50')

    ds_train = datasets.CocoCaptions(root='/path/to/coco_csv/train2014',
                                     annFile='/path/to/coco_csv/coco_train_karpathy.json'
                                     )

    traker = TRAKer(model=model,
                    task='clip',
                    save_dir=tmp_path,
                    train_set_size=len(ds_train),
                    device=device,
                    proj_dim=512,
                    )

    traker.modelout_fn.get_embeddings(model, ds_train, batch_size=1, size=600,
                                      preprocess_fn_img=lambda x: preprocess(x).to(device).unsqueeze(0),
                                      preprocess_fn_txt=lambda x: tokenizer(x[0]).to(device))

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for bind, (img, captions) in enumerate(tqdm(ds_train)):
        x = preprocess(img).to(device).unsqueeze(0)
        # selecting (wlog) the first out of 5 captions
        y = tokenizer(captions[0]).to(device)

        traker.featurize(batch=(x, y), num_samples=x.shape[0])
        if bind == 2:
            break

    traker.finalize_features()
