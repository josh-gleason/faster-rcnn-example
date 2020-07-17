import numpy as np
from utils.box_utils import get_anchor_labels
from utils.data_mappings import voc_name_to_id


def create_voc_targets(data, labels, data_transform, resize_shape, anchor_boxes, pos_iou_thresh=0.7,
                       neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256, mark_max_gt_anchors=True):
    orig_width, orig_height = data.size
    new_width, new_height = resize_shape

    orig_img = data

    num_anchors = anchor_boxes.shape[0]

    if float(new_width) / orig_width < float(new_height) / orig_height:
        scale = float(new_width) / orig_width
    else:
        scale = float(new_height) / orig_height

    valid_width = int(round(orig_width * scale))
    valid_height = int(round(orig_height * scale))

    valid_anchors = np.nonzero(
        np.concatenate((anchor_boxes[:, :2] >= 0,
                        anchor_boxes[:, 2:3] < valid_width,
                        anchor_boxes[:, 3:4] < valid_height),
                       axis=1).all(axis=1))[0]

    obj_labels = labels['annotation']['object']
    if not isinstance(obj_labels, list):
        obj_labels = [obj_labels]

    if len(obj_labels) > 0:
        gt_class_labels = np.array([voc_name_to_id[label['name']] for label in obj_labels], dtype=np.int32)
        gt_boxes = np.array([[label['bndbox'][k] for k in ['xmin', 'ymin', 'xmax', 'ymax']] for label in obj_labels],
                            dtype=np.float32)

        gt_boxes *= scale
    else:
        gt_class_labels = np.zeros((0,), dtype=np.int32)
        gt_boxes = np.zeros((0, 4), dtype=np.float32)

    # image_ids and ignore are not used in pascal voc
    gt_image_id = 0
    ignore = np.zeros(gt_class_labels.shape, dtype=np.bool)

    valid_anchor_boxes = anchor_boxes[valid_anchors, :]

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

    # gt_boxes and gt_class_labels need to be ignored during collate
    return data_transform(data), (anchor_obj_final, anchor_loc_final, gt_boxes, gt_class_labels,
                                  gt_image_id, ignore, orig_img)
