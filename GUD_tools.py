import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import io
import os
import cv2
from torch_tools.visualization import to_image
from utils import make_noise, one_hot

import matplotlib
matplotlib.use("Agg")


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(1, 2, 0)
        .to("cpu")
        .numpy()
    )


def save_results(G, deformator, params, out_dir, z, noises, distance=1):
    deformator.eval()
    G.eval()
    inspect_directions(
        G, deformator, out_dir, noises, zs=z, shifts_r=distance * params.shift_scale
    )


def inspect_directions(G, deformator, out_dir, noises, zs=None, shifts_r=8):
    os.makedirs(out_dir, exist_ok=True)

    step = 1
    max_dim = G.dim_shift
    shifts_count = 6

    for start in range(0, max_dim - 1, step):
    # for start in [40, 149, 177, 207, 295, 327, 355, 390, 393, 420, 486]:
        imgs = []
        dims = range(start, min(start + step, max_dim))
        z = zs
        fig = make_interpolation(
            G, out_dir, noises, deformator=deformator, z=z,
            shifts_count=shifts_count, dims=dims, shifts_r=shifts_r,
            dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 1))
        fig.canvas.draw()
        plt.close(fig)
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
        img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
        imgs.append(img)

        out_file = os.path.join(out_dir, '{}.jpg'.format(dims[0]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)


# def make_interpolation(G, deformator=None, z=None,
#                              shifts_r=10.0, shifts_count=5,
#                              dims=None, dims_count=10, texts=None, **kwargs):
#     with_deformation = deformator is not None
#     if with_deformation:
#         deformator_is_training = deformator.training
#         deformator.eval()
#     z = z if z is not None else make_noise(1, G.dim_z).cuda()
#
#     if with_deformation:
#         original_img = G(z).cpu()
#     else:
#         original_img = G(z).cpu()
#     imgs = []
#     if dims is None:
#         dims = range(dims_count)
#     for i in dims:
#         imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))
#
#     rows_count = len(imgs) + 1
#     fig, axs = plt.subplots(rows_count, **kwargs)
#
#     axs[0].axis('off')
#     axs[0].imshow(to_image(original_img, True))
#
#     if texts is None:
#         texts = dims
#     for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
#         ax.axis('off')
#         plt.subplots_adjust(left=0.5)
#         ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
#
#     if deformator is not None and deformator_is_training:
#         deformator.train()
#
#     return fig


def make_interpolation(G, outdir, noises, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()

    if with_deformation:
        original_img = G(z).cpu()
    else:
        original_img = G(z).cpu()
    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, noises, shifts_r, shifts_count, i, deformator))

    for i in range(len(imgs[0])):
        tensor = imgs[0][i]
        img = tensor.numpy()

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))

        save_path = os.path.join(outdir, str(dims[0]))
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, '{}.jpg'.format(i))
        cv2.imwrite(save_file, img)

    rows_count = len(imgs)
    fig, axs = plt.subplots(rows_count, **kwargs)
    axs = [axs] if rows_count == 1 else axs

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs, imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        # ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def interpolate(G, z, noises, shifts_r, shifts_count, dim, deformator=None, with_central_border=True):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r, shifts_r / shifts_count):
        if deformator is not None:
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
        else:
            latent_shift = one_hot(G.dim_shift, shift, dim).cuda()
        latent_code = z.clone().detach()
        shifted_image = G.gen_shifted(latent_code, latent_shift, noises).cpu()[0]
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor