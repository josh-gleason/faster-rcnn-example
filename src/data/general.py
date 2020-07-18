import numpy as np
import torch
from torchvision import transforms as tvt
import torchvision.transforms.functional as tvtf
from PIL import Image


def compute_scales(orig_shape, min_shape, max_shape):
    orig_height, orig_width = orig_shape
    max_height, max_width = max_shape
    min_height, min_width = min_shape

    scale_h_min = float(min_height) / orig_height
    scale_w_min = float(min_width) / orig_width
    scale_min = max(scale_w_min, scale_h_min)

    scale_h_max = float(max_height) / orig_height
    scale_w_max = float(max_width) / orig_width
    scale_max = min(scale_h_max, scale_w_max)

    scale_h = scale_w = min(scale_max, scale_min)

    # make sure both scales result in integer length
    scale_h = round(scale_h * orig_height) / orig_height
    scale_w = round(scale_w * orig_width) / orig_width

    return scale_h, scale_w


class DynamicResize:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'PIL.Image.NEAREST',
        Image.BILINEAR: 'PIL.Image.BILINEAR',
        Image.BICUBIC: 'PIL.Image.BICUBIC',
        Image.LANCZOS: 'PIL.Image.LANCZOS',
        Image.HAMMING: 'PIL.Image.HAMMING',
        Image.BOX: 'PIL.Image.BOX',
    }

    def __init__(self, min_shape=(600, 600), max_shape=(1000, 1000), interpolation=Image.BILINEAR):
        """ Resize the image to the shape, retaining aspect ratio and padding the right and bottom of image with
            reflection padding.

        Args:
            min_shape: (h, w) minimum shape of image.
            max_shape: (h, w) maximum shape of image.
            interpolation: Type of interpolation, one of Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS,
                Image.HAMMING, or Image.BOX.
        """
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.interpolation = interpolation

    def __call__(self, img):
        img_width, img_height = img.size
        scale_h, scale_w = compute_scales((img_height, img_width), self.min_shape, self.max_shape)
        scaled_height = int(round(scale_h * img_height))
        scaled_width = int(round(scale_w * img_width))
        return tvtf.resize(img, (scaled_height, scaled_width))

    def __repr__(self):
        interpolate_str = self._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(min_shape={0}, max_shape={1}, interpolation={2})'.format(
            self.min_shape, self.max_shape, interpolate_str)


class PadToShape:
    def __init__(self, target_shape=(1000, 1000), mode='constant', value=0):
        self.target_shape = target_shape
        self.mode = mode
        self.value = value

    def __call__(self, img):
        assert torch.is_tensor(img)
        img_width, img_height = img.shape[-2:]
        target_height, target_width = self.target_shape
        pad_height = target_height - img_height
        pad_width = target_width - img_width
        assert pad_width >= 0 and pad_height >= 0
        return torch.nn.functional.pad(img, [0, pad_height, 0, pad_width], mode=self.mode, value=self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(target_shape={0}, mode={1}, value={2})'.format(
            self.target_shape, self.mode, self.value)


def faster_rcnn_collate_fn(batch):
    def pad_zero_rows(ary, total_rows):
        """ pad end of dimension 0 with zeros to ensure ary.shape[0] = total_rows """
        return np.pad(ary, [[0, total_rows - ary.shape[0]]] + [[0, 0]] * (len(ary.shape) - 1))

    # pad boxes with additional zeros so they are all the same size, this allows DataParallel to scatter properly
    gt_count = [len(b[1][3]) for b in batch]
    gt_max_count = np.max(gt_count)

    gt_boxes = [pad_zero_rows(b[1][2], gt_max_count) for b in batch]
    gt_class_labels = [pad_zero_rows(b[1][3], gt_max_count) for b in batch]

    imgs = [b[1][-1] for b in batch]
    valid_shapes = [b[1][-2] for b in batch]
    gt_ignore_labels = [b[1][-3] for b in batch]
    batch = [(b[0], (b[1][0], b[1][1], boxes, labels, counts, batch_idx, b[1][-4]))
             for batch_idx, (b, boxes, labels, counts) in enumerate(zip(batch, gt_boxes, gt_class_labels, gt_count))]
    data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels, gt_count, data_batch_idx, gt_image_ids) = \
        torch.utils.data.dataloader.default_collate(batch)
    return data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels, gt_count, data_batch_idx,
                  gt_image_ids, gt_ignore_labels, valid_shapes, imgs)
