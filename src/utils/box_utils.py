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

    # TODO THIS IS TEMPORARY SHOULD BE REMOVED!!!
    # positive_choice = np.arange(num_positive)
    # negative_choice = np.arange(num_negative)
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

    # TODO should this be img_width - 1 and img_height - 1?
    boxes[:, ::2] = np.clip(boxes[:, ::2], 0, img_width) * (orig_width / img_width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_height) * (orig_height / img_height)

    return boxes


def define_anchor_boxes(sub_sample, width, height):
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


def get_bboxes_from_output3(output, resized_shapes, orig_shapes, score_thresh=0.05, nms_thresh=0.3):
    def loc2bbox(src_bbox, loc):
        if src_bbox.shape[0] == 0:
            return np.zeros((0, 4), dtype=loc.dtype)

        src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

        src_height = src_bbox[:, 2] - src_bbox[:, 0]
        src_width = src_bbox[:, 3] - src_bbox[:, 1]
        src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
        src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

        dy = loc[:, 0::4]
        dx = loc[:, 1::4]
        dh = loc[:, 2::4]
        dw = loc[:, 3::4]

        ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
        ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
        h = np.exp(dh) * src_height[:, np.newaxis]
        w = np.exp(dw) * src_width[:, np.newaxis]

        dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
        dst_bbox[:, 0::4] = ctr_y - 0.5 * h
        dst_bbox[:, 1::4] = ctr_x - 0.5 * w
        dst_bbox[:, 2::4] = ctr_y + 0.5 * h
        dst_bbox[:, 3::4] = ctr_x + 0.5 * w

        return dst_bbox

    def suppress(raw_cls_bbox, raw_prob, n_class):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = ops.nms(cls_bbox_l, prob_l, nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append(l * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    all_pred_roi_cls = output['pred_roi_cls'].detach().cpu().numpy()
    all_pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    all_pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    all_pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    dtype = all_pred_roi_loc.dtype

    final_preds = []
    batch_indices = np.sort(np.unique(all_pred_roi_batch_idx))
    for batch_idx in batch_indices:
        final_preds.append(defaultdict(lambda: {'rects': [], 'confs': []}))

        resized_shape = np.array(resized_shapes[batch_idx], dtype=dtype)
        orig_shape = np.array(orig_shapes[batch_idx], dtype=dtype)

        # start working in y/x land
        pred_roi_cls = all_pred_roi_cls[all_pred_roi_batch_idx == batch_idx]
        pred_roi_boxes = all_pred_roi_boxes[all_pred_roi_batch_idx == batch_idx, :][:, [1, 0, 3, 2]]
        pred_roi_loc = all_pred_roi_loc[all_pred_roi_batch_idx == batch_idx, ...][:, :, [1, 0, 3, 2]]
        n_classes = pred_roi_loc.shape[1]

        pred_roi_cls = torch.tensor(pred_roi_cls).cuda()
        pred_roi_boxes = torch.tensor(pred_roi_boxes).cuda()
        pred_roi_loc = torch.tensor(pred_roi_loc.reshape((pred_roi_loc.shape[0], -1))).cuda()

        scale = resized_shape[1] / orig_shape[1]

        pred_roi_boxes = pred_roi_boxes / scale

        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float).cuda().repeat(n_classes)[None]
        std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float).cuda().repeat(n_classes)[None]

        pred_roi_loc = (pred_roi_loc * std + mean)
        pred_roi_loc = pred_roi_loc.view(-1, n_classes, 4)
        pred_roi_boxes = pred_roi_boxes.view(-1, 1, 4).expand_as(pred_roi_loc)
        cls_bbox = loc2bbox(pred_roi_boxes.detach().cpu().numpy().reshape((-1, 4)),
                            pred_roi_loc.detach().cpu().numpy().reshape((-1, 4)))
        cls_bbox = torch.tensor(cls_bbox).cuda()
        cls_bbox = cls_bbox.view(-1, n_classes * 4)
        # clip bounding box
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=orig_shape[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=orig_shape[1])

        prob = (torch.softmax(pred_roi_cls, dim=1))

        bbox, label, score = suppress(cls_bbox, prob, n_classes)

        bbox = bbox[:, [1, 0, 3, 2]]
        # end working in y/x land
        for idx in range(label.size):
            final_preds[-1][label[idx].item()]['rects'].append(bbox[idx, :])
            final_preds[-1][label[idx].item()]['confs'].append(score[idx])
        for k in final_preds[-1]:
            final_preds[-1][k]['rects'] = np.array(final_preds[-1][k]['rects'], dtype=np.float32)
            final_preds[-1][k]['confs'] = np.array(final_preds[-1][k]['confs'], dtype=np.float32)

    return final_preds, batch_indices


def get_bboxes_from_output2(output, resized_shapes, orig_shapes, score_thresh=0.05, nms_thresh=0.3):
    all_pred_roi_cls_sm = torch.softmax(output['pred_roi_cls'], dim=1).detach().cpu().numpy()
    all_pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    all_pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    all_pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    dtype = all_pred_roi_loc.dtype

    final_preds = []
    batch_indices = np.sort(np.unique(all_pred_roi_batch_idx))
    for batch_idx in batch_indices:
        img_height, img_width = orig_shapes[batch_idx]
        resized_shape = np.array(resized_shapes[batch_idx], dtype=dtype)
        orig_shape = np.array(orig_shapes[batch_idx], dtype=dtype)

        scale_xy = orig_shape[None, None, ::-1] / resized_shape[None, None, ::-1]

        pred_roi_cls_sm = all_pred_roi_cls_sm[all_pred_roi_batch_idx == batch_idx]
        pred_roi_boxes = all_pred_roi_boxes[all_pred_roi_batch_idx == batch_idx, :]
        pred_roi_loc = all_pred_roi_loc[all_pred_roi_batch_idx == batch_idx, ...]

        # apply pred_roi_loc correction to pred_roi_boxes and scale to orig_shape
        loc_mean = np.array((0., 0., 0., 0.), dtype=dtype)[None, None, :]
        loc_std = np.array((0.1, 0.1, 0.2, 0.2), dtype=dtype)[None, None, :]
        pred_roi_loc = pred_roi_loc * loc_std + loc_mean

        pred_roi_boxes_w_h = (pred_roi_boxes[:, None, 2:] - pred_roi_boxes[:, None, :2]) * scale_xy
        pred_roi_boxes_ctr_x_y = 0.5 * (pred_roi_boxes[:, None, :2] + pred_roi_boxes[:, None, 2:]) * scale_xy

        dxy = pred_roi_loc[:, :, :2]
        dwh = pred_roi_loc[:, :, 2:]

        pred_result_boxes_ctr_x_y = pred_roi_boxes_ctr_x_y + dxy * pred_roi_boxes_w_h
        pred_result_boxes_w_h = np.exp(dwh) * pred_roi_boxes_w_h

        pred_result_boxes = np.zeros(pred_roi_loc.shape, dtype=dtype)
        pred_result_boxes[:, :, :2] = pred_result_boxes_ctr_x_y - 0.5 * pred_result_boxes_w_h
        pred_result_boxes[:, :, 2:] = pred_result_boxes_ctr_x_y + 0.5 * pred_result_boxes_w_h

        # clamp to original image bounds
        # TODO should this be img_width - 1 and img_height - 1?
        pred_result_boxes[:, :, ::2] = np.clip(pred_result_boxes[:, :, ::2], 0, img_width)
        pred_result_boxes[:, :, 1::2] = np.clip(pred_result_boxes[:, :, 1::2], 0, img_height)

        # suppress boxes based on score and add to final dict
        final_preds.append(defaultdict(lambda: {'rects': [], 'confs': []}))
        for class_idx in range(1, pred_roi_cls_sm.shape[1]):
            class_boxes = pred_result_boxes[:, class_idx, :]
            class_scores = pred_roi_cls_sm[:, class_idx]
            keep_mask = class_scores > score_thresh
            class_boxes = class_boxes[keep_mask, :]
            class_scores = class_scores[keep_mask]
            if class_boxes.size > 0:
                keep_index = ops.nms(
                    torch.from_numpy(class_boxes).cuda(),
                    torch.from_numpy(class_scores).cuda(),
                    nms_thresh).cpu().numpy()
                class_boxes = class_boxes[keep_index, :]
                class_scores = class_scores[keep_index]
            if class_boxes.size > 0:
                final_preds[-1][class_idx]['rects'] = class_boxes
                final_preds[-1][class_idx]['confs'] = class_scores
    return final_preds, batch_indices


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
                                       img_width=resized_shape[1], img_height=resized_shape[0],
                                       loc_mean=np.array((0., 0., 0., 0.)),
                                       loc_std=np.array((0.1, 0.1, 0.2, 0.2)),
                                       orig_width=orig_shape[1], orig_height=orig_shape[0])
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
