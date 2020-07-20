import os
from torchvision import datasets
from PIL import Image
import numpy as np
from utils.box_utils import select_roi_indices


class CocoDetectionWithImgId(datasets.CocoDetection):
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target, img_id)

        return img, target


def create_coco_targets(data, labels, image_id, data_transform, resize_shape, anchor_boxes, valid_anchors,
                        pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256,
                        mark_max_gt_anchors=True):
    orig_width, orig_height = data.size
    new_width, new_height = resize_shape

    orig_img = data

    num_anchors = anchor_boxes.shape[0]

    if len(labels) > 0:
        ignore = np.array([('ignore' in label and label['ignore']) or ('iscrowd' in label and label['iscrowd'])
                           for label in labels],
                          dtype=np.bool)

        gt_class_labels = np.array([label['category_id'] for label in labels], dtype=np.int32)
        gt_boxes = np.array([label['bbox'] for label in labels], dtype=np.float32)

        gt_boxes[:, 2:] += gt_boxes[:, 0:2]

        scale_x = float(new_width) / orig_width
        scale_y = float(new_height) / orig_height
        gt_boxes[:, ::2] *= scale_x
        gt_boxes[:, 1::2] *= scale_y
    else:
        ignore = np.zeros((0,), dtype=np.bool)
        gt_class_labels = np.zeros((0,), dtype=np.int32)
        gt_boxes = np.zeros((0, 4), dtype=np.float32)

    valid_anchor_boxes = anchor_boxes[valid_anchors, :]

    valid_positive_index, valid_negative_index, positive_class, positive_loc = \
        select_roi_indices(valid_anchor_boxes, gt_boxes, gt_class_labels, pos_iou_thresh,
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
                                  image_id, ignore, orig_img)
