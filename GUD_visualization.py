import argparse
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from constants import DEFORMATOR_TYPE_DICT
from latent_deformator import LatentDeformator
from trainer import Trainer, Params
from models.gan_load import make_style_gan2
from GUD_tools import save_results


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)


# def make_noise(batch, dim, truncation=None, seed=None):
#     if isinstance(dim, int):
#         dim = [dim]
#     if truncation is None or truncation == 1.0:
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#             torch.manual_seed(seed)
#         return torch.tensor([random.random() for _ in range(batch * np.prod(dim))]).reshape([batch] + dim)


def save_img(tensor, path):
    img = tensor.cpu().detach().numpy()
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    cv2.imwrite(path, img)


device = torch.device('cuda')
parser = argparse.ArgumentParser(description="editing SAR image")

for key, val in Params().__dict__.items():
    target_type = type(val) if val is not None else int
    parser.add_argument('--{}'.format(key), type=target_type, default=None)
parser.add_argument('--output', type=str, default='output1/sample_4', help='path to save dir')
parser.add_argument('--gan_weights', type=str, default='models/StyleGAN2/checkpoint/060000.pt',
                    help='path to GAN weight')
parser.add_argument('--deformator_weights', type=str,
                    default='results_dir/models/deformator_100000.pt',
                    help='deformator path')
parser.add_argument('--distance', type=int, default=6, help='move distance in latent space')
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--deformator', type=str, default='ortho',
                    choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
parser.add_argument('--deformator_random_init', type=bool, default=True)
parser.add_argument('--w_shift', type=bool, default=False,
                    help='latent directions search in w-space for StyleGAN2')
parser.add_argument('--gan_resolution', type=int, default=128,
                    help='generator out images resolution. Required only for StyleGAN2')

args = parser.parse_args()
torch.cuda.set_device(args.device)

gan_weight_path = args.gan_weights
G = make_style_gan2(args.gan_resolution, gan_weight_path, args.w_shift)

deformator_weight_path = args.deformator_weights
deformator = LatentDeformator(shift_dim=512,
                              input_dim=args.directions_count,
                              out_dim=args.max_latent_dim,
                              type=DEFORMATOR_TYPE_DICT[args.deformator],
                              random_init=args.deformator_random_init).cuda()
deformator.load_state_dict(torch.load(deformator_weight_path))

params = Params(**args.__dict__)
params.directions_count = int(deformator.input_dim)
params.max_latent_dim = int(deformator.out_dim)


z = make_noise(1, 512).to(device)
noises = None

save_results(G, deformator, params, args.output, z, noises, args.distance)








