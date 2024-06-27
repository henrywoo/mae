import argparse
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hiq import set_seed, str_to_filename
from hiq.vis import print_model

import models_mae
"""
🌳 MaskedAutoencoderViT<trainable_params:329239296,all_params:329541888,percentage:99.90818%>
├── PatchEmbed(patch_embed)
│   └── Conv2d(proj)|weight[1024,3,16,16]🇸 -(16, 16)|bias[1024]🇸 -(16, 16)
├── ModuleList(blocks)
│   └── 💠 Block(0-23)<🦜:12596224x24>
│       ┣━━ 💠 LayerNorm(norm1,norm2)<🦜:2048x2>|weight[1024]|bias[1024]
│       ┣━━ Attention(attn)
│       ┃   ┣━━ Linear(qkv)|weight[3072,1024]|bias[3072]
│       ┃   ┗━━ Linear(proj)|weight[1024,1024]|bias[1024]
│       ┗━━ Mlp(mlp)
│           ┣━━ Linear(fc1)|weight[4096,1024]|bias[4096]
│           ┗━━ Linear(fc2)|weight[1024,4096]|bias[1024]
├── LayerNorm(norm)|weight[1024]|bias[1024]
├── Linear(decoder_embed)|weight[512,1024]|bias[512]
├── ModuleList(decoder_blocks)
│   └── 💠 Block(0-7)<🦜:3152384x8>
│       ┣━━ 💠 LayerNorm(norm1,norm2)<🦜:1024x2>|weight[512]|bias[512]
│       ┣━━ Attention(attn)
│       ┃   ┣━━ Linear(qkv)|weight[1536,512]|bias[1536]
│       ┃   ┗━━ Linear(proj)|weight[512,512]|bias[512]
│       ┗━━ Mlp(mlp)
│           ┣━━ Linear(fc1)|weight[2048,512]|bias[2048]
│           ┗━━ Linear(fc2)|weight[512,2048]|bias[512]
├── LayerNorm(decoder_norm)|weight[512]|bias[512]
└── Linear(decoder_pred)|weight[768,512]|bias[768]
"""

# Define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title='', show=False):
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=9)
    plt.axis('off')
    if show:
        plt.show()


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model


def run_one_image(img, model, title=None):
    x0 = torch.tensor(img).unsqueeze(dim=0)
    x1 = torch.einsum('nhwc->nchw', x0)
    loss, y0, mask = model(x1.float(), mask_ratio=0.75)
    y1 = model.unpatchify(y0)
    y = torch.einsum('nchw->nhwc', y1).detach().cpu()
    m0 = mask.detach().unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    m1 = model.unpatchify(m0)
    m2 = torch.einsum('nchw->nhwc', m1).detach().cpu()
    x = torch.einsum('nchw->nhwc', x1)
    im_masked = x * (1 - m2)
    im_paste = x * (1 - m2) + y * m2
    plt.rcParams['figure.figsize'] = [9, 3]
    plt.subplot(1, 4, 1)
    show_image(x[0], "original")
    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")
    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")
    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")
    if title:
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(str_to_filename(title, suffix='png'))
    plt.show()


def load_image(img_url):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.
    assert img.shape == (224, 224, 3)
    img = (img - imagenet_mean) / imagenet_std
    return img


def main(args):
    img = load_image(args.img_url)
    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img), show=True)

    model_mae = prepare_model(args.chkpt_dir, args.arch)
    print_model(model_mae)
    run_one_image(img, model_mae, "MAE")

    if args.gan_chkpt_dir:
        model_mae_gan = prepare_model(args.gan_chkpt_dir, args.arch)
        print_model(model_mae_gan)
        run_one_image(img, model_mae_gan, "MAE with extra GAN loss")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAE image processing")
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    parser.add_argument('--img_url', type=str, default=img_url, help='URL of the image to process')
    parser.add_argument('--chkpt_dir', type=str, default='mae_visualize_vit_large.pth', help='Path to the checkpoint file')
    parser.add_argument('--gan_chkpt_dir', type=str, default='mae_visualize_vit_large_ganloss.pth', help='Path to the GAN checkpoint file (optional)')
    parser.add_argument('--arch', type=str, default='mae_vit_large_patch16', help='Architecture of the model')

    args = parser.parse_args()
    set_seed(has_torch=True)
    main(args)