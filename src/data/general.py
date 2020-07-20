import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional as tvtf
from PIL import Image

from utils.box_utils import create_rpn_targets


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


class ComposeTransforms(object):
    """Composes several transforms together. Similar to Compose except each transform takes two inputs and
       returns two outputs.

    Args:
        transforms (list of callables): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability = flip_probability

    def __call__(self, img, labels):
        if np.random.rand() < self.flip_probability:
            img = tvtf.hflip(img)
            img_width = img.size[0]
            labels['gt_boxes'][:, ::2] = (img_width - 1) - labels['gt_boxes'][:, -2::-2]
            labels['original_image'] = img
        return img, labels

    def __repr__(self):
        return f'{self.__class__.__name__}(flip_probability={self.flip_probability})'


class DynamicResize(object):
    _pil_interpolation_to_str = {
        Image.NEAREST: 'PIL.Image.NEAREST',
        Image.BILINEAR: 'PIL.Image.BILINEAR',
        Image.BICUBIC: 'PIL.Image.BICUBIC',
        Image.LANCZOS: 'PIL.Image.LANCZOS',
        Image.HAMMING: 'PIL.Image.HAMMING',
        Image.BOX: 'PIL.Image.BOX',
    }

    def __init__(self, min_shape=(600, 600), max_shape=(1000, 1000), interpolation=Image.BILINEAR):
        """ Resize the image so that image to fit into min_shape/max_shape. See compute_scales for more information.

        Args:
            min_shape: (h, w) minimum shape of image.
            max_shape: (h, w) maximum shape of image.
            interpolation: Type of interpolation, one of Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS,
                Image.HAMMING, or Image.BOX.
        """
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.interpolation = interpolation

    def __call__(self, img, labels):
        img_width, img_height = img.size
        scale_h, scale_w = compute_scales((img_height, img_width), self.min_shape, self.max_shape)
        scaled_height = int(round(scale_h * img_height))
        scaled_width = int(round(scale_w * img_width))
        labels['valid_shape'] = np.array((scaled_height, scaled_width), dtype=np.int64)
        labels['gt_boxes'] = labels['gt_boxes'] * np.array([[scale_w, scale_h, scale_w, scale_h]], dtype=np.float32)
        img = tvtf.resize(img, (scaled_height, scaled_width), interpolation=self.interpolation)
        return img, labels

    def __repr__(self):
        interpolate_str = self._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(min_shape={0}, max_shape={1}, interpolation={2})'.format(
            self.min_shape, self.max_shape, interpolate_str)


class PadToShape(object):
    """ Pad image to the right and bottom so that it is the desired shape """

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
        return f'{self.__class__.__name__}(target_shape={self.target_shape}, mode="{self.mode}", value={self.value})'


class CreateRPNLabels(object):
    """ Create RPN targets and add them to the labels dict. Assumes all other manipulations have been finished """

    def __init__(self, sub_sample=16):
        self.sub_sample = sub_sample

    def __call__(self, img, labels):
        assert torch.is_tensor(img)
        img_height, img_width = img.shape[-2:]
        rpn_obj_label, rpn_loc_label = create_rpn_targets(
            (img_height, img_width), labels['valid_shape'], labels['gt_boxes'], labels['gt_class_labels'],
            self.sub_sample)
        labels['rpn_obj_label'] = rpn_obj_label
        labels['rpn_loc_label'] = rpn_loc_label
        return img, labels

    def __repr__(self):
        return f'{self.__class__.__name__}(sub_sample={self.sub_sample})'



def faster_rcnn_collate_fn(batch):
    def pad_zero_rows(ary, total_rows):
        """ pad end of dimension 0 with zeros to ensure ary.shape[0] = total_rows """
        return np.pad(ary, [[0, total_rows - ary.shape[0]]] + [[0, 0]] * (len(ary.shape) - 1))

    # pad boxes with additional zeros so they are all the same size, this allows DataParallel to scatter properly
    gt_count = [b[1]['gt_class_labels'].size for b in batch]
    gt_max_count = np.max(gt_count)
    gt_boxes = [pad_zero_rows(b[1]['gt_boxes'], gt_max_count) for b in batch]
    gt_class_labels = [pad_zero_rows(b[1]['gt_class_labels'], gt_max_count) for b in batch]

    difficult = [b[1]['difficult'] for b in batch]
    ignore = [b[1]['ignore'] for b in batch]
    image_id = [b[1]['image_id'] for b in batch]
    original_image = [b[1]['original_image'] for b in batch]
    valid_shape = [b[1]['valid_shape'] for b in batch]

    # NOTE: collate only works if all input images are the same size (batch_size=1 or pad_to_max=True)
    # collate these labels (they need to be passed to the network or objective)
    batch = [(b[0], {'gt_boxes': boxes,
                     'gt_class_labels': labels,
                     'gt_count': count,
                     'rpn_obj_label': b[1]['rpn_obj_label'],
                     'rpn_loc_label': b[1]['rpn_loc_label'],
                     'batch_idx': batch_idx})
             for batch_idx, (b, boxes, labels, count, shape) in enumerate(zip(
                 batch, gt_boxes, gt_class_labels, gt_count, valid_shape))]
    data, labels = torch.utils.data.dataloader.default_collate(batch)

    # pass through these labels (they don't need to be tensors)
    labels['original_image'] = original_image
    labels['difficult'] = np.array(difficult, dtype=np.bool)
    labels['valid_shape'] = np.array(valid_shape, dtype=np.int64)
    labels['ignore'] = np.array(ignore, dtype=np.bool)
    labels['image_id'] = np.array(image_id, dtype=np.int64)

    return data, labels
