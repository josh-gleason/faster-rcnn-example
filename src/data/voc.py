import numpy as np
from utils.data_mappings import voc_name_to_id


class FormatVOCLabels:
    def __init__(self, use_difficult=True):
        self.use_difficult = use_difficult

    def __call__(self, data, labels):
        obj_labels = labels['annotation']['object']
        if not isinstance(obj_labels, list):
            obj_labels = [obj_labels]

        # filter difficult if necessary
        difficult_all = np.array([int(label['difficult']) for label in obj_labels], dtype=np.bool)
        if self.use_difficult:
            keep_mask = np.ones(len(obj_labels), dtype=np.bool)
        else:
            keep_mask = np.logical_not(difficult_all)

        obj_labels = [label for idx, label in enumerate(obj_labels) if keep_mask[idx]]

        gt_boxes = np.array([[int(label['bndbox'][k]) - 1 for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                             for label in obj_labels], dtype=np.float32)
        gt_class_labels = np.array([voc_name_to_id[label['name']] for label in obj_labels], dtype=np.int32)
        difficult = difficult_all[keep_mask]

        # neither of these labels are used for pascal VOC but are there for consistency
        image_id = 0
        ignore = np.zeros(gt_class_labels.shape, dtype=np.bool)

        labels_dict = {
            'gt_boxes': gt_boxes,
            'gt_class_labels': gt_class_labels,
            'difficult': difficult,
            'ignore': ignore,
            'image_id': image_id,
            'original_image': data,
        }

        return data, labels_dict

    def __repr__(self):
        return f'{self.__class__.__name__}(use_difficult={self.use_difficult})'
