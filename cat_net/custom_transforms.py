import numbers
import random

from PIL import Image, ImageOps


class Clamp:
    """Clamp all elements in input into the range [min, max].

    Args:
        min (Number): lower-bound of the range to be clamped to
        min (Number): upper-bound of the range to be clamped to
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): the input Tensor

        Returns:
            Tensor: the result Tensor
        """
        return tensor.clamp(self.min, self.max)


class UnNormalize:
    """Scale a normalized tensor image to have mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] * std[channel]) + mean[channel]) ``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be un-normalized.

        Returns:
            Tensor: Un-normalized Tensor image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class StatefulRandomCrop:
    """Crop the given PIL.Image at a random location, but retain the location for subsequent calls.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

        self.x1 = None
        self.y1 = None

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        if self.x1 is None:
            self.x1 = random.randint(0, w - tw)
            self.y1 = random.randint(0, h - th)
        return img.crop((self.x1, self.y1, self.x1 + tw, self.y1 + th))
