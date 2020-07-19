import numpy as np
import torch
from collections import defaultdict
from torchvision import ops
from utils.data_mappings import coco_id_to_name, voc_id_to_name


def compute_iou(boxes1, boxes2):
    boxes1_areas = np.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1).reshape(-1, 1)
    boxes2_areas = np.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1).reshape(1, -1)
    int_left_top = np.maximum(boxes1[:, :2].reshape(-1, 1, 2), boxes2[:, :2])
    int_right_bot = np.minimum(boxes1[:, 2:].reshape(-1, 1, 2), boxes2[:, 2:])
    int_width_height = np.maximum(0, int_right_bot - int_left_top)
    int_areas = np.prod(int_width_height, axis=2)
    union_areas = np.maximum(1e-6, boxes1_areas + boxes2_areas - int_areas)

    return int_areas / union_areas


def choose_anchor_subset(num_samples, pos_ratio, len_positive, len_negative):
    num_positive = min(int(round(num_samples * pos_ratio)), len_positive)
    num_negative = num_samples - num_positive
    if num_negative > len_negative:
        # shouldn't happen unless image is just packed with objects, but just in case
        num_negative = len_negative
        num_positive = num_samples - num_negative

        # if this assert fails then reduce num_samples or loosen the thresholds
        assert num_positive <= len_positive

    positive_choice = np.random.choice(np.arange(len_positive), num_positive, replace=False)
    negative_choice = np.random.choice(np.arange(len_negative), num_negative, replace=False)

    return positive_choice, negative_choice


def get_loc_labels(anchor_boxes, gt_boxes, loc_mean=None, loc_std=None):
    assert anchor_boxes.shape[0] == gt_boxes.shape[0]

    if anchor_boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    anchor_width_height = anchor_boxes[:, 2:] - anchor_boxes[:, :2]
    anchor_center_x_y = 0.5 * (anchor_boxes[:, :2] + anchor_boxes[:, 2:])
    gt_width_height = gt_boxes[:, 2:] - gt_boxes[:, :2]
    gt_center_x_y = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])

    loc_x_y = (gt_center_x_y - anchor_center_x_y) / anchor_width_height
    loc_width_height = np.log(gt_width_height / anchor_width_height)

    loc = np.concatenate((loc_x_y, loc_width_height), axis=1)

    if loc_mean is not None:
        loc = (loc - loc_mean.reshape(1, 4)) / loc_std.reshape(1, 4)

    return loc


def get_boxes_from_loc_batch(anchor_boxes, loc, img_height, img_width, loc_mean=None, loc_std=None):
    """ Convert batch of loc predictions (as torch.tensors) to bounding boxes.

    Args:
        anchor_boxes (torch.tensor): Shape (N, 4)
        loc (torch.tensor): Shape (B, N, 4)
        img_width (int): Image width
        img_height (int): Image height
        loc_mean (torch.tensor, optional): Shape (4,). Used to un-normalize loc.
        loc_std (torch.tensor, optional): Shape (4,). Used to un-normalize loc.

    Returns:
        torch.tensor: Shape (B, N, 4). Converts all loc predictions to bounding boxes (clamped to
            image bounds).
    """
    batch_size = loc.shape[0]

    if loc_mean is not None:
        loc = loc * loc_std.reshape(1, 1, 4) + loc_mean.reshape(1, 1, 4)

    anchor_center_x_y = 0.5 * (anchor_boxes[None, :, None, :2] + anchor_boxes[None, :, None, 2:])
    anchor_width_height = anchor_boxes[None, :, None, 2:] - anchor_boxes[None, :, None, :2]

    boxes_center_x_y = loc[:, :, None, :2] * anchor_width_height + anchor_center_x_y
    boxes_width_height = (torch.exp(loc[:, :, None, 2:]) * anchor_width_height)

    neg_pos = torch.tensor([-0.5, 0.5], dtype=torch.float32).reshape(1, 1, 2, 1).to(device=loc.device)
    boxes = (boxes_center_x_y + neg_pos * boxes_width_height).reshape(batch_size, -1, 4)

    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], 0, img_width)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], 0, img_height)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], 0, img_width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], 0, img_height)

    return boxes


def get_boxes_from_loc_multiclass(anchor_boxes, loc, img_width, img_height, loc_mean=None, loc_std=None):
    """ Convert loc predictions to bounding boxes for loc predictions applied to multiple loc predictions.

    Args:
        anchor_boxes (np.array or torch.tensor): Shape (N, 4)
        loc (np.array or torch.tensor): Shape (N, C, 4)
        img_width (int): Image width used to clamp results
        img_height (int): Image height used to clamp final result
        loc_mean (np.array or torch.tensor, optional): Shape (4,). Used to un-normalize loc.
        loc_std (np.array or torch.tensor, optional): Shape (4,). Used to un-normalize loc.

    Returns:
        np.array or torch.tensor: Shape (N, C, 4). Converts all loc predictions to bounding boxes (clamped to
            image bounds).
    """
    if loc_mean is not None:
        loc = loc * loc_std.reshape(1, 1, 4) + loc_mean.reshape(1, 1, 4)

    anchor_center_xy = 0.5 * (anchor_boxes[:, None, None, :2] + anchor_boxes[:, None, None, 2:])
    anchor_wh = anchor_boxes[:, None, None, 2:] - anchor_boxes[:, None, None, :2]

    boxes_center_xy = loc[:, :, None, :2] * anchor_wh + anchor_center_xy
    boxes_wh = (np.exp(loc[:, :, None, 2:]) * anchor_wh)

    # add -0.5 and 0.5 times width and height to centers to construct final output
    neg_pos = np.array([-0.5, 0.5], dtype=loc.dtype).reshape(1, 1, 2, 1)
    boxes = boxes_center_xy + neg_pos * boxes_wh

    # clamp
    boxes = np.clip(boxes, 0, np.array([img_width - 1, img_height - 1], dtype=loc.dtype).reshape(1, 1, 1, 2))

    return boxes.reshape(*loc.shape)


def get_boxes_from_loc(anchor_boxes, loc, img_width, img_height, loc_mean=None, loc_std=None):
    """ Convert loc predictions to bounding boxes.

    Args:
        anchor_boxes (np.ndarray): Shape (N, 4)
        loc (np.ndarray): Shape (N, 4)
        img_width (int): Image width used to clamp results
        img_height (int): Image height used to clamp final result
        loc_mean (np.ndarray, optional): Shape (4,). Used to un-normalize loc.
        loc_std (np.ndarray, optional): Shape (4,). Used to un-normalize loc.

    Returns:
        np.array: Shape (N, 4). Converts all loc predictions to bounding boxes.
    """
    if loc_mean is not None:
        loc = loc * loc_std.reshape(1, 4) + loc_mean.reshape(1, 4)

    anchor_center_xy = 0.5 * (anchor_boxes[:, None, :2] + anchor_boxes[:, None, 2:])
    anchor_wh = anchor_boxes[:, None, 2:] - anchor_boxes[:, None, :2]

    boxes_center_xy = loc[:, None, :2] * anchor_wh + anchor_center_xy
    boxes_wh = (np.exp(loc[:, None, 2:]) * anchor_wh)

    neg_pos = np.array([-0.5, 0.5], dtype=loc.dtype).reshape(1, 2, 1)
    boxes = (boxes_center_xy + neg_pos * boxes_wh).reshape(-1, 4)

    # clamp
    boxes[:, ::2] = np.clip(boxes[:, ::2], 0, img_width - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_height - 1)

    return boxes


def define_anchor_boxes(sub_sample, height, width):
    feature_map_w, feature_map_h = (width // sub_sample), (height // sub_sample)

    # using np.array
    ratios = np.array((2, 1, 0.5), dtype=np.float32).reshape(-1, 1)
    anchor_scales = np.array((8, 16, 32), dtype=np.float32).reshape(1, -1)
    anchor_base_x = (sub_sample * anchor_scales * np.sqrt(ratios) / 2).reshape(1, -1, 1, 1)
    anchor_base_y = (sub_sample * anchor_scales * np.sqrt(1.0 / ratios) / 2).reshape(1, -1, 1, 1)
    ctr_x, ctr_y = np.meshgrid(
        sub_sample // 2 + sub_sample * np.arange(feature_map_w, dtype=np.float32),
        sub_sample // 2 + sub_sample * np.arange(feature_map_h, dtype=np.float32))
    ctr_x = ctr_x.reshape(-1, 1, 1, 1)
    ctr_y = ctr_y.reshape(-1, 1, 1, 1)
    neg_pos = np.array([-1.0, 1.0], dtype=np.float32).reshape(1, 1, -1, 1)
    anchors_x = ctr_x + (anchor_base_x * neg_pos)
    anchors_y = ctr_y + (anchor_base_y * neg_pos)
    anchor_boxes = np.concatenate((anchors_x, anchors_y), axis=3).reshape(-1, 4)

    return anchor_boxes


def get_boxes_from_output(output, resized_shapes, orig_shapes, score_thresh=0.05, nms_thresh=0.3):
    all_pred_roi_cls_sm = torch.softmax(output['pred_roi_cls'], dim=1).detach().cpu().numpy()
    all_pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    all_pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    all_pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    dtype = all_pred_roi_loc.dtype

    final_preds = []
    batch_indices = np.sort(np.unique(all_pred_roi_batch_idx))
    for batch_idx in batch_indices:
        img_height, img_width = orig_shapes[batch_idx]

        pred_roi_cls_sm = all_pred_roi_cls_sm[all_pred_roi_batch_idx == batch_idx]
        pred_roi_boxes = all_pred_roi_boxes[all_pred_roi_batch_idx == batch_idx, :]
        pred_roi_loc = all_pred_roi_loc[all_pred_roi_batch_idx == batch_idx, ...]

        # scale pred_roi_boxes to original image coordinates
        resized_shape = np.array(resized_shapes[batch_idx], dtype=dtype)
        orig_shape = np.array(orig_shapes[batch_idx], dtype=dtype)
        scale_xyxy = np.tile((orig_shape[None, ::-1] / resized_shape[None, ::-1]), (1, 2))
        pred_roi_boxes = pred_roi_boxes * scale_xyxy

        # apply pred_roi_loc corrections to pred_roi_boxes
        pred_result_boxes = get_boxes_from_loc_multiclass(
            pred_roi_boxes, pred_roi_loc, img_width, img_height,
            loc_mean=np.array((0.0, 0.0, 0.0, 0.0), dtype=pred_roi_loc.dtype),
            loc_std=np.array((0.1, 0.1, 0.2, 0.2), dtype=pred_roi_loc.dtype))

        # suppress boxes based on score and add to final dict
        final_preds.append(defaultdict(lambda: {'rects': np.zeros((0, 4)), 'confs': np.zeros((0,))}))
        for class_idx in range(1, pred_roi_cls_sm.shape[1]):
            class_boxes = pred_result_boxes[:, class_idx, :]
            class_scores = pred_roi_cls_sm[:, class_idx]
            keep_mask = class_scores > score_thresh
            class_boxes = class_boxes[keep_mask, :]
            class_scores = class_scores[keep_mask]
            if class_boxes.size > 0:
                keep_index = ops.nms(
                    torch.from_numpy(class_boxes),
                    torch.from_numpy(class_scores),
                    nms_thresh).cpu().numpy()
                class_boxes = class_boxes[keep_index, :]
                class_scores = class_scores[keep_index]
            if class_boxes.size > 0:
                final_preds[-1][class_idx]['rects'] = class_boxes
                final_preds[-1][class_idx]['confs'] = class_scores
    return final_preds, batch_indices


def get_anchor_labels(anchor_boxes, gt_boxes, gt_class_labels, pos_iou_thresh,
                      neg_iou_thresh, pos_ratio, num_samples, mark_max_gt_anchors):
    if gt_boxes.size != 0:
        iou = compute_iou(anchor_boxes, gt_boxes)

        # get positive & negative anchors
        anchor_max_iou = np.max(iou, axis=1)

        anchor_positive_index1 = np.argmax(iou, axis=0) if mark_max_gt_anchors \
            else np.empty(0, dtype=np.int32)
        # handle tied cases robustly
        anchor_positive_index1 = np.nonzero(iou == anchor_max_iou[anchor_positive_index1])[0]
        anchor_positive_index2 = np.nonzero(anchor_max_iou >= pos_iou_thresh)[0]
        anchor_positive_index = np.unique(
            np.append(anchor_positive_index1, anchor_positive_index2))
        anchor_negative_index = np.nonzero(anchor_max_iou < neg_iou_thresh)[0]
    else:
        anchor_positive_index = np.zeros((0,), dtype=np.int32)
        anchor_negative_index = np.arange(anchor_boxes.shape[0], dtype=np.int32)

    positive_choice, negative_choice = choose_anchor_subset(
        num_samples, pos_ratio, len(anchor_positive_index), len(anchor_negative_index))

    anchor_positive_index = anchor_positive_index[positive_choice]
    anchor_negative_index = anchor_negative_index[negative_choice]

    if gt_boxes.size != 0:
        gt_positive_index = np.argmax(iou[anchor_positive_index, :], axis=1)

        # collect all anchor indices and generate labels
        anchor_positive_class_labels = gt_class_labels[gt_positive_index]
        anchor_positive_loc_labels = get_loc_labels(
            anchor_boxes[anchor_positive_index, :], gt_boxes[gt_positive_index, :])
    else:
        anchor_positive_class_labels = np.zeros((0,), dtype=np.int32)
        anchor_positive_loc_labels = np.zeros((0, 4), dtype=np.float32)

    return anchor_positive_index, anchor_negative_index, \
        anchor_positive_class_labels, anchor_positive_loc_labels


def get_display_pred_boxes(output, resized_shapes, orig_shapes, id_to_name_map, batch_idx=0, top3=False):
    pred_boxes, batch_indices = get_boxes_from_output(output, resized_shapes, orig_shapes)
    box_index = next(idx for idx, b in enumerate(batch_indices) if b == batch_idx)
    batch_pred_boxes = pred_boxes[box_index]

    rect_list = [(rect, conf, cls) for cls in batch_pred_boxes
                 for rect, conf in zip(batch_pred_boxes[cls]['rects'], batch_pred_boxes[cls]['confs'])]
    rect_list = sorted(rect_list, key=lambda x: -x[1])

    output_rects = []
    output_strs = []
    for rect_rank, (rect, conf, cls) in enumerate(rect_list):
        if conf >= 0.7 or (rect_rank < 3 and top3):
            output_rects.append(rect.astype(np.int32).tolist())
            output_strs.append('{}({:.4f})'.format(id_to_name_map[cls], conf))

    return output_rects, output_strs


def get_display_gt_boxes(gt_boxes, gt_class_labels, resized_shapes, orig_shapes, id_to_name_map, gt_count=None,
                         batch_idx=None):
    # if batch_idx is None we assume this is sample directly from dataset so create artificial batch of size 0
    if batch_idx is None:
        gt_boxes = [gt_boxes]
        gt_class_labels = [gt_class_labels]
        orig_shapes = [orig_shapes]
        resized_shapes = [resized_shapes]
        batch_idx = 0
        gt_count = None if gt_count is None else [gt_count]

    if gt_count is None:
        gt_count = gt_boxes[batch_idx].shape[0]
    else:
        gt_count = gt_count[batch_idx]

    # orig_shapes and resized_shapes provided as (height, width)
    scale_x = orig_shapes[batch_idx][1] / resized_shapes[batch_idx][1]
    scale_y = orig_shapes[batch_idx][0] / resized_shapes[batch_idx][0]
    rect_list_gt = gt_boxes[batch_idx][:gt_count]
    if torch.is_tensor(rect_list_gt):
        rect_list_gt = rect_list_gt.cpu().numpy()
    rect_list_gt = rect_list_gt.copy()
    rect_list_gt[:, ::2] *= scale_x
    rect_list_gt[:, 1::2] *= scale_y
    class_labels = gt_class_labels[batch_idx][:gt_count]
    if torch.is_tensor(class_labels):
        class_labels = class_labels.cpu().numpy()
    text_list_gt = [id_to_name_map[lid] for lid in class_labels]

    rect_list_gt = rect_list_gt.astype(np.int32).tolist()

    return rect_list_gt, text_list_gt