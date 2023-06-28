from tqdm import tqdm
from torchvision import datasets
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)

import open_clip

from trak import TRAKer
import logging
import pytest
import open_clip


@pytest.mark.cuda
def test_mscoco(tmp_path, device='cuda:0'):
    model, _, preprocess = open_clip.create_model_and_transforms('RN50')
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer('RN50')

    ds_train = datasets.CocoCaptions(root='/path/to/coco2014/images/train2014',
                                     annFile='/path/to/coco2014/annotations/annotations/captions_train2014.json')

    traker = TRAKer(model=model,
                    task='clip',
                    save_dir=tmp_path,
                    train_set_size=len(ds_train),
                    device=device,
                    proj_dim=512,
                    logging_level=logging.DEBUG
                    )

    traker.task.get_embeddings(model, ds_train, batch_size=1, size=600, embedding_dim=1024,
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
