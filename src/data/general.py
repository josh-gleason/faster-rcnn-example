import numpy as np
import torch


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
