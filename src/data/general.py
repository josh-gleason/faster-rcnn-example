import numpy as np
import torch
from torchvision import transforms as tvt
import torchvision.transforms.functional as tvtf


class ResizeAndPad(tvt.Resize):
    def __call__(self, img):
        img_width, img_height = img.size
        new_width, new_height = self.size

        # pad to desired aspect ratio first then resize
        # if desired_aspect == 1 then pixels are scaled perfectly
        desired_aspect = float(new_width) / new_height
        if float(new_width) / img_width < float(new_height) / img_height:
            pad_width = 0
            scale_height = img_width / (img_height * desired_aspect)
            pad_height = max(0, int(round(img_height * scale_height - img_height)))
        else:
            pad_height = 0
            scale_width = desired_aspect * img_height / img_width
            pad_width = max(0, int(round(img_width * scale_width - img_width)))

        return tvtf.resize(tvtf.pad(img, (0, 0, pad_width, pad_height)), self.size)


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
    gt_ignore_labels = [b[1][-2] for b in batch]
    batch = [(b[0], (b[1][0], b[1][1], boxes, labels, counts, batch_idx, b[1][-3]))
             for batch_idx, (b, boxes, labels, counts) in enumerate(zip(batch, gt_boxes, gt_class_labels, gt_count))]
    data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels, gt_count, data_batch_idx, gt_image_ids) = \
        torch.utils.data.dataloader.default_collate(batch)
    return data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels, gt_count, data_batch_idx,
                  gt_image_ids, gt_ignore_labels, imgs)
