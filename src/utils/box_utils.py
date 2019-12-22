import numpy as np
import time
import torch
from collections import defaultdict
from torchvision import ops


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


def get_boxes_from_loc_batch(anchor_boxes, loc, img_width, img_height, loc_mean=None, loc_std=None):
    # assumes loc is provided with a batch dimension at dim 0
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


def get_boxes_from_loc(anchor_boxes, loc, img_width, img_height, loc_mean=None, loc_std=None,
                       orig_width=None, orig_height=None):
    if loc_mean is not None:
        loc = loc * loc_std.reshape(1, 4) + loc_mean.reshape(1, 4)

    if orig_width is None:
        orig_width = img_width
    if orig_height is None:
        orig_height = img_height

    anchor_center_x_y = 0.5 * (anchor_boxes[:, None, :2] + anchor_boxes[:, None, 2:])
    anchor_width_height = anchor_boxes[:, None, 2:] - anchor_boxes[:, None, :2]

    boxes_center_x_y = loc[:, None, :2] * anchor_width_height + anchor_center_x_y
    boxes_width_height = (np.exp(loc[:, None, 2:]) * anchor_width_height)

    neg_pos = np.array([-0.5, 0.5], dtype=np.float32).reshape(1, 2, 1)
    boxes = (boxes_center_x_y + neg_pos * boxes_width_height).reshape(-1, 4)

    boxes[:, ::2] = np.clip(boxes[:, ::2], 0, img_width) * (orig_width / img_width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_height) * (orig_height / img_height)

    return boxes


def define_anchor_boxes(sub_sample, width, height):
    feature_map_w, feature_map_h = (width // sub_sample), (height // sub_sample)

    # using np.array
    ratios = np.array((0.5, 1, 2), dtype=np.float32).reshape(-1, 1)
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
    anchor_boxes = np.stack((anchors_x, anchors_y), axis=3).reshape(-1, 4)
    valid_anchors = np.nonzero(
        np.concatenate((anchor_boxes[:, :2] >= 0,
                        anchor_boxes[:, 2:3] < width,
                        anchor_boxes[:, 3:4] < height),
                       axis=1).all(axis=1))[0]

    ratios = np.array((0.5, 1, 2), dtype=np.float32).reshape(-1, 1)
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
    valid_anchors = np.nonzero(
        np.concatenate((anchor_boxes[:, :2] >= 0,
                        anchor_boxes[:, 2:3] < width,
                        anchor_boxes[:, 3:4] < height),
                       axis=1).all(axis=1))[0]

    return anchor_boxes, valid_anchors


def apply_nms(boxes, scores, threshold, n_results=-1, return_scores=False):
    if len(scores) == 0:
        if return_scores:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        else:
            return np.zeros((0, 4), dtype=np.float32)

    t0 = time.time()
    idxs = np.argsort(scores)
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    print('compute areas', time.time() - t0)

    final_boxes = []
    if return_scores:
        final_scores = []
    discard = set()
    compute_time = 0.0
    discard_time = 0.0
    for last in reversed(range(len(idxs))):
        if last not in discard:
            final_boxes.append(boxes[idxs[last], :])
            if return_scores:
                final_scores.append(scores[idxs[last]])

            if n_results > 0 and len(final_boxes) >= n_results:
                break

            t0 = time.time()
            int_left_top = np.maximum(boxes[last, :2], boxes[:last, :2])
            int_right_bot = np.minimum(boxes[last, 2:], boxes[:last, 2:])
            int_width_height = np.maximum(0, int_right_bot - int_left_top)
            int_areas = np.prod(int_width_height, axis=1)
            union_areas = areas[last] + areas[:last] - int_areas
            iou = int_areas / union_areas
            compute_time += time.time() - t0

            t0 = time.time()
            discard.update(np.nonzero(iou > threshold)[0])
            discard_time += time.time() - t0

    print('compute iou time', compute_time)
    print('discard iou time', discard_time)

    t0 = time.time()
    if return_scores:
        output = (np.array(final_boxes, dtype=np.float32), np.array(final_scores, dtype=np.float32))
    else:
        output = np.array(final_boxes, dtype=np.float32)
    print('build output', time.time() - t0)

    return output


def get_bboxes_from_output(output, resized_shapes, orig_shapes, threshold=0.0):
    all_pred_roi_cls = torch.softmax(output['pred_roi_cls'], dim=1).detach().cpu().numpy()
    all_pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    all_pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    all_pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    final_preds = []
    batch_indices = np.sort(np.unique(all_pred_roi_batch_idx))
    for batch_idx in batch_indices:
        resized_shape = resized_shapes[batch_idx]
        orig_shape = orig_shapes[batch_idx]

        pred_roi_cls = all_pred_roi_cls[all_pred_roi_batch_idx == batch_idx]
        pred_roi_boxes = all_pred_roi_boxes[all_pred_roi_batch_idx == batch_idx, :]
        pred_roi_loc = all_pred_roi_loc[all_pred_roi_batch_idx == batch_idx, :]

        pred_cls = np.argmax(pred_roi_cls, axis=1)
        pred_conf = np.max(pred_roi_cls, axis=1)
        keep_idx = np.nonzero(pred_cls > 0)[0]

        final_preds.append(defaultdict(lambda: {'rects': [], 'confs': []}))
        if len(keep_idx) > 0:
            pred_cls = pred_cls[keep_idx]
            pred_conf = pred_conf[keep_idx]
            pred_boxes = pred_roi_boxes[keep_idx]
            pred_loc = pred_roi_loc[keep_idx, pred_cls, :]

            rects = get_boxes_from_loc(pred_boxes, pred_loc,
                                       img_width=resized_shape[0], img_height=resized_shape[1],
                                       loc_mean=np.array((0., 0., 0., 0.)),
                                       loc_std=np.array((0.1, 0.1, 0.2, 0.2)),
                                       orig_width=orig_shape[0], orig_height=orig_shape[1])
            pre_nms = defaultdict(list)
            for rect, conf, cls in zip(rects, pred_conf, pred_cls):
                if cls > 0:
                    pre_nms[cls].append((rect, conf))

            for cls in pre_nms:
                cls_rects, cls_conf = zip(*pre_nms[cls])
                cls_rects = np.concatenate([c.reshape(1, -1) for c in cls_rects], axis=0)
                cls_conf = np.array(cls_conf)

                cls_rects = torch.from_numpy(cls_rects).float()
                cls_conf = torch.from_numpy(cls_conf).float()
                keep_nms = ops.nms(cls_rects.cuda(), cls_conf.cuda(), 0.3).cpu()
                keep_nms = keep_nms[np.nonzero(cls_conf[keep_nms].numpy() >= threshold)[0]]

                if len(keep_nms) > 0:
                    final_preds[-1][cls]['rects'] = cls_rects[keep_nms].numpy()
                    final_preds[-1][cls]['confs'] = cls_conf[keep_nms].numpy()

    return final_preds, batch_indices


def get_anchor_labels(anchor_boxes, gt_boxes, gt_class_labels, pos_iou_thresh,
                      neg_iou_thresh, pos_ratio, num_samples, mark_max_gt_anchors):
    if gt_boxes.size != 0:
        iou = compute_iou(anchor_boxes, gt_boxes)

        # get positive & negative anchors
        anchor_max_iou = np.max(iou, axis=1)

        anchor_positive_index1 = np.argmax(iou, axis=0) if mark_max_gt_anchors \
            else np.empty(0, dtype=np.int32)
        anchor_positive_index2 = np.nonzero(anchor_max_iou > pos_iou_thresh)[0]
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
