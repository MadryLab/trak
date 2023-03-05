import pytest
import torch as ch
from tqdm import tqdm
from torchvision import datasets
import open_clip

from traker.traker import TRAKer
from traker.modelout_functions import CLIPModelOutput
from traker.gradient_computers import IterativeGradientComputer

@pytest.mark.cuda
def test_mscoco(device='cuda:0'):
    model, _, preprocess = open_clip.create_model_and_transforms('RN50')
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer('RN50')

    ds_train = datasets.CocoCaptions(root = '/mnt/xfs/projects/trak/datasets/coco_csv/train2014',
                                   annFile = '/mnt/xfs/projects/trak/datasets/coco_csv/coco_train_karpathy.json',
                               )

    logit_scale = [v for (k, v) in model.named_parameters() if k == 'logit_scale'][0]
    modelout_fn = CLIPModelOutput(device=device, temperature=logit_scale)
    trak = TRAKer(model=model,
                  grad_wrt=[x[1] for x in model.named_parameters() if x[0] != 'logit_scale'],
                  device=device,
                  train_set_size=len(ds_train),
                  gradient_projector=IterativeGradientComputer,
                  proj_dim=100)


    def compute_outputs(model, imgs, txt_tokens):
        img_ft, txt_ft, _ = model(imgs, txt_tokens)
        return modelout_fn.get_output(img_ft, txt_ft)

    def compute_out_to_loss(model, imgs, txt_tokens):
        img_ft, txt_ft, _ = model(imgs, txt_tokens)
        return modelout_fn.get_out_to_loss(img_ft, txt_ft)

    modelout_fn.get_embeddings(model, ds_train, batch_size=1, size=100,
                               preprocess_fn_img=lambda x: preprocess(x).to(device).unsqueeze(0),
                               preprocess_fn_txt=lambda x: tokenizer(x[0]).to(device))

    trak.load_params(list(model.parameters()))
    for bind, (img, captions) in enumerate(tqdm(ds_train)):
        x = preprocess(img).to(device).unsqueeze(0)
        # selecting (wlog) the first out of 5 captions
        y = tokenizer(captions[0]).to(device)

        trak.featurize(out_fn=compute_outputs,
                       loss_fn=compute_out_to_loss,
                       batch=(x, y))

        if bind == 20:
            break
