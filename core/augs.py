import torch
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as fun
import torchvision.transforms as transforms

from core.config import MAX_MASK_SIDE, MIN_MASK_SIDE


def patch_cut(x, patch_size):
    """Splits an image to patches of size `patch_size`.
    :param x: shape [B, C, H, W]
    :return x_patches: shape [patch_size ** 2 * B, 3, patch_size, patch_size]

    Args:
        x (torch.tensor): shape [B, C, H, W]
        patch_size (int): square patch width and height

    Returns:
        torch.tensor: shape [H // patch_size * W // patch_size * B, 3, patch_size, patch_size]
    """
    # TODO: enforce divisible by `patch_size`
    B, C, H, W = x.shape
    x_patches = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x_patches = x_patches.permute(0, 1, 2, 4, 3, 5)  # [B, C, H // patch_size, W // patch_size, patch_size, patch_size]
    x_patches = x_patches.reshape(-1, C, patch_size, patch_size)

    return x_patches


def patch_parse(img_patches, height, width):
    """Parses image patches created by `patch_cut` back to an image/ feature map.

    Args:
        img_patches (torch.tensor): shape [H // patch_size * W // patch_size * B, 3, patch_size, patch_size]
        height (int): H_image // patch_size; needed for reconstruction
        width (int): W_image // patch_size; needed for reconstruction

    Returns:
        torch.tensor: reconstructed image tensor of shape [B, C, H, W]
    """
    # TODO: doesn't enforce shapes
    batch_times_patches, C, patch_size = img_patches.shape[:3]
    batch_size = batch_times_patches // (height * width)
    img_recon = img_patches.reshape(batch_size, C, height, width, patch_size, patch_size)
    img_recon = img_recon.permute(0, 1, 2, 4, 3, 5)  # [B, C, H/p, p, W/p, p]
    img_recon = img_recon.reshape(batch_size, C, height * patch_size, width * patch_size)

    return img_recon


class ResizeToMultiple(object):
    """Resize the input PIL Image's shorter side to the nearest multiple of `multiple`.

    Args:
        multiple: resize *shorter* side to nearest floor multiple of this value, e.g. [260, 320] -> [256, 315]
            for default 32
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, multiple=32, interpolation=Image.BILINEAR):
        assert isinstance(multiple, int)
        self.multiple = multiple
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        shorter = min(img.size)
        shorter_rounded = (shorter // self.multiple) * self.multiple
        if shorter_rounded < shorter:
            img = fun.resize(img, shorter_rounded, self.interpolation)
        return img


class CropToMultiple(object):
    """Crop the input PIL Image's sizes to the nearest multiple of `multiple`.

    Args:
        multiple: crop sizes to nearest floor multiple of this value, e.g. [256, 315] -> [256, 288]
            for default 32
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, multiple=32):
        assert isinstance(multiple, int)
        self.multiple = multiple

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        w_rounded = (w // self.multiple) * self.multiple
        h_rounded = (h // self.multiple) * self.multiple

        img = fun.crop(img, 0, 0, h_rounded, w_rounded)

        return img


class AspectRatioCrop(object):
    """Crops the given PIL Image to max aspect ratio.
    """

    def __init__(self, max_aspect=MAX_MASK_SIDE/MIN_MASK_SIDE):
        self.max_aspect = max_aspect

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        w, h = img.size
        if w > h:
            h_crop = h
            w_crop = int(self.max_aspect * h)
        else:
            h_crop = int(self.max_aspect * w)
            w_crop = w

        img = fun.center_crop(img, (h_crop, w_crop))

        return img


class MaybeResize:

    def __init__(self, max_shorter_side):
        """Resize shorter side to max_shorter_side if shorter side higher than that value. Preserves aspect ratio.

        :param max_shorter_side: maximum shortest side
        """
        self.max_shorter_side = max_shorter_side

    def __call__(self, img):
        shorter_size = min(img.size)
        if shorter_size > self.max_shorter_side:
            img = fun.resize(img, self.max_shorter_side)

        return img


class ToNumpy(object):

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            numpy array uint8
        """
        img_np = np.array(img)
        return img_np


def load_augs(resize_to=256):
    # Base preprocessing transform:
    augs_base = transforms.Compose([
        transforms.Resize(resize_to),
        CropToMultiple(32)
    ])

    # segnet transform for detectron (consumes numpy):
    augs_seg = ToNumpy()

    # NOTE: SimCLR doesn't have standard normalization!
    augs_rep = transforms.Compose([
        transforms.ToTensor()
    ])
    return dict(augs_base=augs_base, augs_seg=augs_seg, augs_rep=augs_rep)


def mask_to_fixed_shape(mask, length, is_landscape=True):
    """Pads longer side to `length`. If not is_landscape, returns the transpose so that the segmentation is always
    same size for storage reasons.

    :param mask: shape [H, W]
    :param length: int
    :return:
    """
    # TODO: test
    if is_landscape:
        dlength = length - mask.shape[1]
        assert dlength >= 0
        pad_sequence = ((0, 0), (0, dlength))
    else:
        dlength = length - mask.shape[0]
        assert dlength >= 0
        pad_sequence = ((0, dlength), (0, 0))

    mask = np.pad(mask, pad_sequence, mode='constant', constant_values=0)
    if not is_landscape:
        mask = np.transpose(mask)

    return mask
