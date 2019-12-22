import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from utils.box_utils import compute_iou


def create_coco_json_obj(preds, image_ids):
    results = []
    for pred, image_id in zip(preds, image_ids):
        if image_id == 0:
            print('WARNING got image_id=0. Removing from results')
            continue
        for cls in pred:
            for rect, conf in zip(pred[cls]['rects'], pred[cls]['confs']):
                x, y, x2, y2 = rect
                w, h = x2 - x, y2 - y
                bbox = [round(float(v), 1) for v in [x, y, w, h]]
                results.append({'image_id': int(image_id.item()),
                                'category_id': int(cls),
                                'bbox': bbox,
                                'score': float(conf)})
    return results


def save_coco_style(filename, preds, image_ids):
    results = create_coco_json_obj(preds, image_ids)
    with open(filename, 'w') as fout:
        json.dump(results, fout)


def compute_ap(gt, preds, iou_thresh=0.5):
    confs = defaultdict(list)
    tp = defaultdict(list)
    fp = defaultdict(list)
    total_gt = defaultdict(lambda: 0)

    for p, g in tqdm(zip(preds, gt), total=len(preds), ncols=0, desc=f'Computing Matches @ IoU {iou_thresh:.02f}'):
        all_cls = np.union1d(np.array(list(g.keys()), dtype=np.int64), np.array(list(p.keys()), dtype=np.int64))
        for cls in all_cls:
            if cls not in g:
                # everything is a false positive in this case
                confs[cls].extend(p[cls]['confs'].tolist())
                tp[cls].extend([0] * len(p[cls]['confs']))
                fp[cls].extend([1] * len(p[cls]['confs']))
                continue

            gt_rects = g[cls]['rects']
            gt_ignore = g[cls]['ignore']
            total_gt[cls] += len(gt_ignore) - sum(gt_ignore.astype(np.int32))

            if cls not in p:
                # everything is a false negative in this case
                continue

            # get sorted in descending order
            sorted_idx = np.argsort(-p[cls]['confs'])
            preds_rects = p[cls]['rects'][sorted_idx]
            preds_confs = p[cls]['confs'][sorted_idx]

            iou = compute_iou(preds_rects, gt_rects)
            matched_gt_indices = set()
            for pred_idx, pred_ious in enumerate(iou):
                confs[cls].append(preds_confs[pred_idx])
                gt_match_indices = np.argsort(-pred_ious)
                for top_gt_idx in gt_match_indices:
                    if pred_ious[top_gt_idx] >= iou_thresh and top_gt_idx not in matched_gt_indices:
                        matched_gt_indices.add(top_gt_idx)
                        if gt_ignore[top_gt_idx]:
                            # ignored matches count as neither false or true positive
                            tp[cls].append(0)
                            fp[cls].append(0)
                        else:
                            # this is a match count as true positive
                            tp[cls].append(1)
                            fp[cls].append(0)
                        break
                else:
                    # no match this is a false positive
                    fp[cls].append(1)
                    tp[cls].append(0)

    ap = dict()
    for cls in confs:

        sort_idx = np.argsort(confs[cls])[::-1]
        cum_tp = np.cumsum([tp[cls][idx] for idx in sort_idx])
        cum_fp = np.cumsum([fp[cls][idx] for idx in sort_idx])

        cls_precision = cum_tp / (cum_tp + cum_fp + 1e-16)
        if total_gt[cls] > 0:
            cls_recall = cum_tp / float(total_gt[cls])
        else:
            cls_recall = cum_tp * 0.0

        # max interpolation of precision
        cls_interp_precision = np.maximum.accumulate(cls_precision[::-1])[::-1]

        ap[cls] = 0.0
        prev_recall = 0.0
        prev_recall_step = 0.0
        prev_prec = cls_interp_precision[0]
        for prec, recall in zip(cls_interp_precision, cls_recall):
            if prec < prev_prec:
                ap[cls] += (prev_recall - prev_recall_step) * prev_prec
                prev_recall_step = prev_recall
            prev_prec = prec
            prev_recall = recall
        ap[cls] += (recall - prev_recall_step) * prec
    return ap


def compute_map(gt, preds, iou_thresh=0.5):
    ap = compute_ap(gt, preds, iou_thresh)
    mean_ap = np.mean(list(ap.values()))
    return mean_ap


def compute_ap_coco(gt_file, preds, pred_image_ids, iou_thresh=0.5):
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO

    preds_anns = create_coco_json_obj(preds, pred_image_ids)

    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(preds_anns)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.iouThrs = [iou_thresh]
    cocoEval.params.maxDets = [100]
    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    cocoEval.params.areaRngLbl = ['all']
    cocoEval.evaluate()
    cocoEval.accumulate()
    precision = cocoEval.eval['precision'][0, :, :, 0, 0]

    ap = dict()
    for idx in range(precision.shape[1]):
        cls = cocoEval.params.catIds[idx]
        p = precision[:, idx]
        ap[cls] = np.mean(p[p > -1])

    return ap


def compute_map_coco(gt_file, preds, pred_image_ids, iou_thresh=0.5):
    ap = compute_ap_coco(gt_file, preds, pred_image_ids, iou_thresh)
    mean_ap = np.mean(list(ap.values()))
    return mean_ap


class AverageMeter:
    def __init__(self, alpha=None, drop_first=False):
        """
        Keeps a running total with optionally limited memory. This is known as exponential smoothing. Some math
        is provided to help you choose alpha.

        Average calculated as
            running_average = alpha*running_average + (1-alpha)*sample

        Assuming each sample is IID with mean mu and standard deviation sigma, then after sufficient time has passed
        the mean of the average meter will be mu and the standard deviation is sigma*sqrt((1-alpha)/(1+alpha)). Based
        on this, if we want the standard deviation of the average to be sigma*(1/N) for some N then we should choose
        alpha=(N**2-1)/(N**2+1).

        The time constant (tau) of an exponential filter is the number of updates before the average meter is expected
        to reach (1 - 1/e) * mu = 0.632 * mu when initialized with running_average=0. This can be thought of as the
        delay in the filter. It's relation to alpha is alpha = exp(-1/tau). Note that this meter initializes
        running_average with the first sample value, rather than 0, so in reality the expected value of the average
        meter is always mu (still assuming IID). In a real system the average may be a non-stationary statistics (for
        example training loss) so choosing a alpha with a reasonable time constant is still important.

        Some reasonable values for alpha

        alpha = 0.9 results in
            sigma = 0.23 * sigma
            tau = 10

        alpha = 0.98 results in
            sigma_meter = 0.1 * sigma
            tau = 50

        alpha = 0.995 results in
            sigma_meter = 0.05 * sigma
            tau = 200

        Args
            alpha (None or float): Range 0 < alpha < 1. The closer to 1 the more accurate the estimate but
                the more delayed the estimate. If None the average meter simply keeps a running total and returns
                the current average.
            drop_first (bool): If True then ignore the first call to update. Useful in, for example, measuring data
                loading times since the first call to the loader takes much longer than subsequent calls.
        """
        self._alpha = alpha
        self._drop_first = drop_first

        self._first = None
        self._value = None
        self._running_value = None
        self._count = None
        self.reset()

    def update(self, value, batch_size=1):
        if self._drop_first and self._first:
            self._first = False
            return

        if self._alpha is not None:
            self._value = value
            w = self._alpha ** batch_size
            self._running_value = w * self._running_value + (1.0 - w) * value \
                if self._running_value is not None else value
        else:
            self._value = value
            if self._running_value is not None:
                self._running_value += self._value * batch_size
            else:
                self._running_value = self._value * batch_size
            self._count += batch_size

    @property
    def average(self):
        if self._alpha is not None:
            return self._running_value if self._running_value is not None else 0.0
        elif self._running_value is None:
            return 0
        else:
            return self._running_value / self._count if self._count > 0 else 0.0

    @property
    def value(self):
        return self._value if self._value is not None else 0

    @property
    def count(self):
        return self._count

    def reset(self):
        self._value = None
        self._running_value = None
        self._count = 0
        self._first = True
