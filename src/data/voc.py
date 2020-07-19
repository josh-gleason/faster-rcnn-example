import numpy as np
from utils.box_utils import get_anchor_labels
from utils.data_mappings import voc_name_to_id
from data.general import compute_scales
import torchvision.transforms.functional as tvtf
from utils.box_utils import define_anchor_boxes


def create_voc_targets(data, labels, data_transform, min_shape=(600, 600), max_shape=(1000, 1000),
                       sub_sample=16, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256,
                       mark_max_gt_anchors=True, random_flip=False, use_difficult=True):
    flipped = False
    if random_flip and np.random.choice([True, False]):
        data = tvtf.hflip(data)
        flipped = True

    orig_img = data
    orig_width, orig_height = data.size

    scale_h, scale_w = compute_scales((orig_height, orig_width), min_shape, max_shape)
    valid_height = int(round(scale_h * orig_height))
    valid_width = int(round(scale_w * orig_width))

    anchor_boxes = define_anchor_boxes(sub_sample, valid_width, valid_height)

    num_anchors = anchor_boxes.shape[0]
    valid_anchors = np.nonzero(
        np.concatenate((anchor_boxes[:, :2] >= 0,
                        anchor_boxes[:, 2:3] <= valid_width - 1,
                        anchor_boxes[:, 3:4] <= valid_height - 1),
                       axis=1).all(axis=1))[0]

    obj_labels = labels['annotation']['object']
    if not isinstance(obj_labels, list):
        obj_labels = [obj_labels]

    if len(obj_labels) > 0:
        gt_class_labels = np.array([voc_name_to_id[label['name']] for label in obj_labels if use_difficult or
                                    not int(label['difficult'])], dtype=np.int32)
        gt_difficult = np.array([int(label['difficult']) for label in obj_labels if use_difficult or not
                                 int(label['difficult'])], dtype=np.bool)
        gt_boxes = np.array([[int(label['bndbox'][k]) - 1 for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                              for label in obj_labels if use_difficult or not int(label['difficult'])], dtype=np.float32)
        gt_boxes[:, :2] -= 0.5
        gt_boxes[:, 2:] += 0.5
        if flipped:
            gt_boxes[:, ::2] = orig_width - gt_boxes[:, -2::-2]

        scale_y, scale_x = scale_h, scale_w
        gt_boxes[:, ::2] *= scale_x
        gt_boxes[:, 1::2] *= scale_y
    else:
        gt_class_labels = np.zeros((0,), dtype=np.int32)
        gt_difficult = np.zeros((0,), dtype=np.bool)
        gt_boxes = np.zeros((0, 4), dtype=np.float32)

    # image_ids and ignore are not used in pascal voc
    gt_image_id = 0
    ignore = np.zeros(gt_class_labels.shape, dtype=np.bool)

    valid_anchor_boxes = anchor_boxes[valid_anchors, :]

    # randomly select num_samples anchors to score against try to choose pos/neg ratio of pos_ratio
    valid_positive_index, valid_negative_index, positive_class, positive_loc = \
        get_anchor_labels(valid_anchor_boxes, gt_boxes, gt_class_labels, pos_iou_thresh,
                          neg_iou_thresh, pos_ratio, num_samples, mark_max_gt_anchors)

    positive_index = valid_anchors[valid_positive_index]
    negative_index = valid_anchors[valid_negative_index]

    # construct final labels
    anchor_loc_final = np.zeros((num_anchors, 4), dtype=np.float32)
    anchor_loc_final[positive_index] = positive_loc

    anchor_obj_final = np.full((num_anchors,), -1, dtype=np.int64)
    anchor_obj_final[positive_index] = 1
    anchor_obj_final[negative_index] = 0

    valid_shape = (valid_height, valid_width)

    # gt_boxes and gt_class_labels need to be ignored during collate
    return data_transform(data), (anchor_boxes, anchor_obj_final, anchor_loc_final, gt_boxes, gt_class_labels,
                                  gt_image_id, gt_difficult, ignore, valid_shape, orig_img)
