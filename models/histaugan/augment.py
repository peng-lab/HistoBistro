
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


# options for initializing HistAuGAN
class opts:
    input_dim = 3
    num_domains = 7
    log_train_img_freq = 100
    log_val_img_freq = 100
    d_iter = 3
    lambda_rec = 10.
    lambda_cls = 1.
    lambda_cls_G = 5.


def augment(x: torch.Tensor, model: pl.LightningModule, domain: torch.Tensor, z_attr: torch.Tensor) -> torch.Tensor:
    bs, _, w, h = x.shape

    assert domain.shape[1] == model.opts.num_domains
    domain = domain.repeat(bs, 1).to(x.device)

    assert z_attr.shape[1] == model.dim_attribute
    z_attr = z_attr.repeat(bs, 1).to(x.device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # to accelerate the forward pass through the GAN
            # compute content encoding
            z_content = model.enc_c(x)

            # generate augmentations
            x = model.gen(z_content, z_attr, domain)  # in range [-1, 1]
    x = F.interpolate(x, (w, h), mode='bilinear')  # otherwise, size 300 for input 299

    return x
