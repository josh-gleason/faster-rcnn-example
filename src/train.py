import os
import json
import tempfile
from tqdm import tqdm
from datetime import datetime
import shutil
import argparse
from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchvision import ops
from torchvision import datasets
from torchvision import transforms as tvt
import torchvision.transforms.functional as tvtf

from utils.data_mappings import coco_num_obj_classes, coco_id_to_name, voc_num_obj_classes, voc_name_to_id, voc_id_to_name
from utils.box_utils import compute_iou, choose_anchor_subset, get_loc_labels, define_anchor_boxes, get_boxes_from_loc
from utils.image_utils import draw_detections
from models.faster_rcnn import FasterRCNN

from PIL import Image

parser = argparse.ArgumentParser('Training code for simple Faster-RCNN implementation')
parser.add_argument('dataset', default='voc', type=str, choices={'voc', 'coco'},
                    help='Which dataset to train on')
parser.add_argument('--coco-root', '-cr', default='../coco', type=str,
                    help='Root directory of COCO dataset')
parser.add_argument('--voc-root', '-vr', default='../pascal_voc', type=str,
                    help='Root directory of Pascal VOC dataset')
parser.add_argument('--train-batch-size', '-b', default=1, type=int,
                    help='Training batch size')
parser.add_argument('--test-batch-size', '-tb', default=16, type=int,
                    help='Test batch size')
parser.add_argument('--num-workers', '-j', default=8, type=int,
                    help='Number of workers')
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float,
                    help='Initial learning rate')
parser.add_argument('--momentum', '-m', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--weight-decay', '-wd', default=1e-5, type=float,
                    help='Weight decay parameter for SGD')
parser.add_argument('--num-epochs', '-ne', default=15, type=int,
                    help='Number of epochs')
parser.add_argument('--lr-step-freq', '-s', default=9, type=int,
                    help='How many epochs before reducing learning rate')
parser.add_argument('--lr-step-gamma', '-g', default=0.1, type=float,
                    help='How much to scale learning rate on step')
parser.add_argument('--batches-per-optimizer-step', '-bs', default=1, type=int,
                    help='How many batches to run before taking a step with the optimizer')
parser.add_argument('--checkpoint-format', '-cf', default='checkpoint_{epoch}.pt', type=str,
                    help='Format of checkpoints (default: checkpoint_{epoch}.pt')
parser.add_argument('--best-checkpoint-format', '-bf', default='best.pt', type=str,
                    help='Format for the best checkpoint')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='Resume from this checkpoint if provided')
parser.add_argument('--use-adam', '-a', action='store_true',
                    help='If this flag is provided then use Adam optimizer, otherwise use SGD')
parser.add_argument('--validate', '-v', action='store_true',
                    help='If this flag is provided then only perform validation and don\'t train')
parser.add_argument('--print-freq', '-pf', default=100, type=int,
                    help='Indicates how many batches to wait before updating tensorboard')
parser.add_argument('--quick-validate', '-qv', action='store_true',
                    help='Skip mAP and only evaluate the first 10 validation images to tensorboard')
args = parser.parse_args()


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


def remove_duplicate_anchors(anchor_index, gt_index, iou):
    # sort by ascending anchor index then by descending iou
    sorted_order = np.lexsort((anchor_index, -iou))

    # find first occurance of each unique anchor index (first occurance has highest IoU)
    _, unique_indices = np.unique(anchor_index[sorted_order], return_index=True)

    final_indices = sorted_order[unique_indices]

    anchor_index = anchor_index[final_indices]
    gt_index = gt_index[final_indices]

    return anchor_index, gt_index


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


def create_coco_targets(data, labels, image_id, data_transform, resize_shape, anchor_boxes, valid_anchors, pos_iou_thresh=0.7,
                        neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256, mark_max_gt_anchors=True):
    orig_width, orig_height = data.size
    new_width, new_height = resize_shape

    orig_img = data

    num_anchors = anchor_boxes.shape[0]

    if len(labels) > 0:
        ignore = np.array([('ignore' in label and label['ignore']) or ('iscrowd' in label and label['iscrowd']) for label in labels], dtype=np.bool)

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
                                  image_id, ignore, orig_img)


def create_voc_targets(data, labels, data_transform, resize_shape, anchor_boxes, valid_anchors, pos_iou_thresh=0.7,
                       neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256, mark_max_gt_anchors=True):
    orig_width, orig_height = data.size
    new_width, new_height = resize_shape

    orig_img = data

    num_anchors = anchor_boxes.shape[0]

    obj_labels = labels['annotation']['object']
    if not isinstance(obj_labels, list):
        obj_labels = [obj_labels]

    if len(obj_labels) > 0:
        gt_class_labels = np.array([voc_name_to_id[label['name']] for label in obj_labels], dtype=np.int32)
        gt_boxes = np.array([[label['bndbox'][k] for k in ['xmin', 'ymin', 'xmax', 'ymax']] for label in obj_labels], dtype=np.float32)

        scale_x = float(new_width) / orig_width
        scale_y = float(new_height) / orig_height
        gt_boxes[:, ::2] *= scale_x
        gt_boxes[:, 1::2] *= scale_y
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


def pad_zero_rows(ary, total_rows):
    """ pad end of dimension 0 with zeros to ensure ary.shape[0] = total_rows """
    return np.pad(ary, [[0, total_rows - ary.shape[0]]] + [[0, 0]] * (len(ary.shape) - 1))


def faster_rcnn_collate_fn(batch):
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


def load_datasets(sub_sample, resize_shape):
    # TODO Pad images with zeros and find appropriate valid anchors to allow for inputs
    #      of difference sizes and aspect ratios
    anchor_boxes, valid_anchors = define_anchor_boxes(sub_sample=sub_sample, height=resize_shape[0], width=resize_shape[1])

    # TODO Improve the transforms with better data augmentation
    train_transform = tvt.Compose([
        tvt.Resize(resize_shape),
        tvt.ToTensor(),
        tvt.Normalize([0.5] * 3, [0.5] * 3)
    ])
    val_transform = tvt.Compose([
        tvt.Resize(resize_shape),
        tvt.ToTensor(),
        tvt.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if args.dataset == 'coco':
        print('Loading MSCOCO Detection dataset')
        base_transforms = partial(create_coco_targets, resize_shape=resize_shape, anchor_boxes=anchor_boxes,
                                  valid_anchors=valid_anchors)

        train_transforms = partial(base_transforms, data_transform=train_transform)
        val_transforms = partial(base_transforms, data_transform=val_transform)

        train_dataset = CocoDetectionWithImgId(
            os.path.join(args.coco_root, 'images/train2017'),
            os.path.join(args.coco_root, 'annotations/instances_train2017.json'),
            transforms=train_transforms,
        )

        val_dataset = CocoDetectionWithImgId(
            os.path.join(args.coco_root, 'images/val2017'),
            os.path.join(args.coco_root, 'annotations/instances_val2017.json'),
            transforms=val_transforms,
        )

        num_classes = coco_num_obj_classes
    elif args.dataset == 'voc':
        print('Loading Pascal VOC 2012 Detection dataset')
        base_transforms = partial(create_voc_targets, resize_shape=resize_shape, anchor_boxes=anchor_boxes,
                                  valid_anchors=valid_anchors)

        train_transforms = partial(base_transforms, data_transform=train_transform)
        val_transforms = partial(base_transforms, data_transform=val_transform)

        download = not os.path.exists(os.path.join(args.voc_root, 'VOCdevkit/VOC2012'))
        train_dataset = datasets.VOCDetection(args.voc_root, year='2012', download=download,
                                              image_set='train',
                                              transforms=train_transforms)
        val_dataset = datasets.VOCDetection(args.voc_root, year='2012', download=download,
                                            image_set='val',
                                            transforms=val_transforms)
        num_classes = voc_num_obj_classes
    else:
        raise ValueError

    # train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=1000)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)  # equivalent to shuffle=True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=faster_rcnn_collate_fn,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=faster_rcnn_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, anchor_boxes, num_classes


def load_checkpoint(model, criterion, optimizer, lr_scheduler):
    print('====================================================')
    print('loading checkpoint', args.resume)
    checkpoint = torch.load(args.resume)
    model.module.load_state_dict(checkpoint['model.state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    criterion.load_state_dict(checkpoint['criterion.state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler.state_dict'])
    epoch = checkpoint['epoch']
    best_map = checkpoint['best_map']
    print('epoch:', epoch)
    print('best mAP:', best_map)
    print('finished loading checkpoint', args.resume)
    print('====================================================')

    return epoch, best_map


def save_checkpoint(model, criterion, optimizer, lr_scheduler, epoch, best_map, is_best):
    checkpoint_name = args.checkpoint_format.format(epoch=epoch, best_map=best_map)
    print('Saving checkpoint to', checkpoint_name)
    checkpoint = {
        'model.state_dict': model.module.state_dict(),
        'optimizer.state_dict': optimizer.state_dict(),
        'criterion.state_dict': criterion.state_dict(),
        'lr_scheduler.state_dict': lr_scheduler.state_dict(),
        'epoch': epoch + 1,
        'best_map': best_map}
    torch.save(checkpoint, checkpoint_name)
    print('Finished saving checkpoint', checkpoint_name)
    if is_best:
        best_checkpoint_name = args.best_checkpoint_format.format(epoch=epoch, best_map=best_map)
        print('Copying', checkpoint_name, 'to', best_checkpoint_name)
        shutil.copyfile(checkpoint_name, best_checkpoint_name)
        print('Finished copying to', best_checkpoint_name)


def get_optimizer(model, lr=0.01, weight_decay=1e-5, use_adam=True):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
    if use_adam:
        return torch.optim.Adam(params)
    else:
        return torch.optim.SGD(params, momentum=0.9)


class FasterRCNNCriterion(nn.Module):
    def __init__(self, lambda_rpn_loc=1., lambda_roi_loc=1., sigma_rpn_loc=3., sigma_roi_loc=1.):
        super().__init__()
        self.lambda_rpn_loc = lambda_rpn_loc
        self.lambda_roi_loc = lambda_roi_loc
        self.sigma_rpn_loc = sigma_rpn_loc
        self.sigma_roi_loc = sigma_roi_loc

    def forward(self, pred_roi_batch_idx, pred_roi_boxes, pred_roi_cls, pred_roi_loc,
                pred_loc, pred_obj, pred_roi_cls_labels, pred_roi_loc_labels, anchor_obj,
                anchor_loc):
        batch_size = pred_loc.shape[0]

        rpn_loc_loss = self.lambda_rpn_loc * self.get_rpn_loc_loss(pred_loc, anchor_loc, anchor_obj)
        rpn_obj_loss = self.get_rpn_obj_loss(pred_obj, anchor_obj)
        roi_loc_loss = self.lambda_roi_loc * self.get_roi_loc_loss(
            pred_roi_loc, pred_roi_loc_labels, pred_roi_cls_labels, pred_roi_batch_idx, batch_size)
        roi_cls_loss = self.get_roi_cls_loss(
            pred_roi_cls, pred_roi_cls_labels, pred_roi_batch_idx, batch_size)

        loss = rpn_obj_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss

        return loss, rpn_obj_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss

    @staticmethod
    def smooth_l1_loss(x, t, sigma):
        sigma2 = sigma ** 2
        diff = (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2)
        return y

    def get_rpn_loc_loss(self, pred_loc, gt_loc, gt_obj):
        batch_size = pred_loc.shape[0]
        loss = 0
        for batch_idx in range(batch_size):
            positive_gt_idx = torch.where(gt_obj[batch_idx] > 0)[0]
            if positive_gt_idx.shape[0] == 0:
                loss = loss + torch.sum(0. * pred_loc)
                continue

            batch_pred_loc = pred_loc[batch_idx, positive_gt_idx, :]
            batch_gt_loc = gt_loc[batch_idx, positive_gt_idx, :]
            loc_loss = FasterRCNNCriterion.smooth_l1_loss(batch_pred_loc, batch_gt_loc, sigma=self.sigma_rpn_loc)
            loss = loss + torch.sum(loc_loss) / torch.sum(gt_obj[batch_idx] >= 0).float()
        return loss / float(batch_size)

    @staticmethod
    def get_rpn_obj_loss(pred_obj, gt_obj):
        batch_size = pred_obj.shape[0]
        loss = 0
        for batch_idx in range(batch_size):
            valid_gt_idx = torch.where(gt_obj[batch_idx] > -1)[0]
            if valid_gt_idx.shape[0] == 0:
                loss = loss + torch.sum(0. * pred_obj)
                continue

            num_classes = pred_obj.shape[-1]
            batch_pred_obj = pred_obj[batch_idx, valid_gt_idx, :].view(-1, num_classes)
            batch_gt_obj = gt_obj[batch_idx, valid_gt_idx].view(-1)
            loss = loss + F.cross_entropy(batch_pred_obj, batch_gt_obj)
        return loss / float(batch_size)

    @staticmethod
    def get_roi_cls_loss(pred_roi_cls, gt_roi_cls, pred_roi_batch_idx, batch_size):
        loss = 0
        for batch_idx in range(batch_size):
            batch_indices = torch.where(pred_roi_batch_idx == batch_idx)[0]

            if batch_indices.shape[0] == 0:
                loss = loss + torch.sum(0. * pred_roi_cls)
                continue

            batch_pred_roi_cls = pred_roi_cls[batch_indices]
            batch_gt_roi_cls = gt_roi_cls[batch_indices]

            loss = loss + F.cross_entropy(batch_pred_roi_cls, batch_gt_roi_cls)
        return loss / float(batch_size)

    def get_roi_loc_loss(self, pred_roi_loc, gt_roi_loc, gt_roi_cls, pred_roi_batch_idx, batch_size):
        loss = 0
        for batch_idx in range(batch_size):
            batch_indices = torch.where(pred_roi_batch_idx == batch_idx)[0]
            part_divisor = batch_indices.shape[0]

            positive_idx = torch.where(gt_roi_cls[batch_indices] > 0)[0]
            if positive_idx.shape[0] == 0:
                return torch.sum(0. * pred_roi_loc)

            batch_gt_roi_cls = gt_roi_cls[batch_indices[positive_idx]]
            batch_gt_roi_loc = gt_roi_loc[batch_indices[positive_idx], :]
            batch_pred_roi_loc = pred_roi_loc[batch_indices[positive_idx], batch_gt_roi_cls, :]
            loss_part = FasterRCNNCriterion.smooth_l1_loss(batch_pred_roi_loc, batch_gt_roi_loc, sigma=self.sigma_roi_loc)
            loss = loss + torch.sum(loss_part) / float(part_divisor)
        return torch.sum(loss) / float(batch_size)


def get_display_boxes(output, orig_shapes, resized_shapes, batch_idx=0, top3=False):
    pred_roi_cls = torch.softmax(output['pred_roi_cls'], dim=1).detach().cpu().numpy()
    pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    pred_roi_cls = pred_roi_cls[pred_roi_batch_idx == batch_idx]
    pred_roi_boxes = pred_roi_boxes[pred_roi_batch_idx == batch_idx, :]
    pred_roi_loc = pred_roi_loc[pred_roi_batch_idx == batch_idx, :]

    pred_cls = np.argmax(pred_roi_cls, axis=1)
    pred_conf = np.max(pred_roi_cls, axis=1)
    keep_idx = np.nonzero(pred_cls > 0)[0]

    resized_width, resized_height = resized_shapes[batch_idx]
    orig_width, orig_height = orig_shapes[batch_idx]

    text_list = []
    rect_list = []
    if len(keep_idx) > 0:
        pred_cls = pred_cls[keep_idx]
        pred_conf = pred_conf[keep_idx]
        pred_boxes = pred_roi_boxes[keep_idx]
        pred_loc = pred_roi_loc[keep_idx, pred_cls, :]

        rects = get_boxes_from_loc(pred_boxes, pred_loc, img_width=resized_width, img_height=resized_height,
                                   loc_mean=np.array((0., 0., 0., 0.)),
                                   loc_std=np.array((0.1, 0.1, 0.2, 0.2)),
                                   orig_width=orig_width, orig_height=orig_height)
        pre_nms = defaultdict(list)
        for rect, conf, cls in zip(rects, pred_conf, pred_cls):
            if cls > 0:
                pre_nms[cls].append((rect, conf))

        post_nms_rects = []
        post_nms_confs = []
        post_nms_labels = []
        rect_list = []
        for cls in pre_nms:
            cls_rects, cls_conf = zip(*pre_nms[cls])
            cls_rects = np.concatenate([c.reshape(1, -1) for c in cls_rects], axis=0)
            cls_conf = np.array(cls_conf)

            cls_rects = torch.from_numpy(cls_rects).float()
            cls_conf = torch.from_numpy(cls_conf).float()
            keep_nms = ops.nms(cls_rects, cls_conf, 0.3)
            post_nms_rects.append(cls_rects[keep_nms].numpy())
            post_nms_confs.append(cls_conf[keep_nms].numpy())
            post_nms_labels.append(np.full(len(keep_nms), cls, dtype=np.int32))
        post_nms_rects = np.concatenate(post_nms_rects, axis=0)
        post_nms_confs = np.concatenate(post_nms_confs, axis=0)
        post_nms_labels = np.concatenate(post_nms_labels, axis=0)

        keep_idx = np.nonzero(post_nms_confs >= 0.7)[0]

        if len(keep_idx) == 0 and top3:
            keep_idx = np.argsort(post_nms_confs)[:-4:-1]

        if len(keep_idx) > 0:
            post_nms_rects = post_nms_rects[keep_idx]
            post_nms_confs = post_nms_confs[keep_idx]
            post_nms_labels = post_nms_labels[keep_idx]

            rect_list = np.round(post_nms_rects).astype(np.int32).tolist()

            for cls_id, conf in zip(post_nms_labels, post_nms_confs):
                text_list.append('{}({:.4f})'.format(
                    coco_id_to_name[cls_id] if args.dataset == 'coco' else voc_id_to_name[cls_id],
                    conf))

    return rect_list, text_list


def get_bboxes(output, resized_shapes, orig_shapes, threshold=0.0):
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


def compute_ap(preds, gt, iou_thresh=0.5):
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


def compute_map(preds, gt, iou_thresh=0.5):
    ap = compute_ap(preds, gt, iou_thresh)
    mean_ap = np.mean(list(ap.values()))
    return mean_ap


def train_epoch(epoch, writer, model, criterion, optimizer, lr_scheduler, train_loader):
    model.train()
    optimizer.zero_grad()
    loss_avg = AverageMeter(0.998)
    rpn_obj_loss_avg = AverageMeter(0.998)
    rpn_loc_loss_avg = AverageMeter(0.998)
    roi_cls_loss_avg = AverageMeter(0.998)
    roi_loc_loss_avg = AverageMeter(0.998)
    with tqdm(total=len(train_loader), ncols=0, desc='Training Epoch {}'.format(epoch)) as pbar:
        for batch_num, (data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels,
                               gt_count, initial_batch_idx, gt_image_ids, ignore, imgs)) in enumerate(train_loader):
            data = data.cuda()
            anchor_obj = anchor_obj.cuda()
            anchor_loc = anchor_loc.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_class_labels = gt_class_labels.cuda()
            initial_batch_idx = initial_batch_idx.cuda()

            output = model(data, initial_batch_idx, gt_boxes, gt_class_labels, gt_count)
            loss, rpn_obj_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss \
                = criterion(**output, anchor_obj=anchor_obj, anchor_loc=anchor_loc)

            batch_size = data.shape[0]
            loss_avg.update(loss.item(), batch_size)
            rpn_obj_loss_avg.update(rpn_obj_loss.item(), batch_size)
            rpn_loc_loss_avg.update(rpn_loc_loss.item(), batch_size)
            roi_cls_loss_avg.update(roi_cls_loss.item(), batch_size)
            roi_loc_loss_avg.update(roi_loc_loss.item(), batch_size)

            loss_scaled = loss / float(args.batches_per_optimizer_step)
            loss_scaled.backward()

            if (batch_num + 1) % args.batches_per_optimizer_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_num % args.print_freq == 0:
                img = imgs[0]

                with torch.no_grad():
                    model.eval()
                    test_img = data[0:1, :, :, :]
                    output = model(test_img, torch.tensor((0,)).to(device=test_img.device))
                    model.train()

                resized_shapes = {batch_idx: (d.shape[-1], d.shape[-2]) for batch_idx, d in enumerate(data)}
                orig_shapes = {batch_idx: img.size for batch_idx, img in enumerate(imgs)}
                rect_list, text_list = get_display_boxes(output, orig_shapes, resized_shapes, top3=True)
                pred_img = draw_detections(img, rect_list, text_list)

                writer.add_image('Images/train_{}'.format(epoch), tvtf.to_tensor(pred_img))

                writer.add_scalar('Loss {}/train loss'.format(epoch), loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train rpn_obj_loss'.format(epoch), rpn_obj_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train rpn_loc_loss'.format(epoch), rpn_loc_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train roi_cls_loss'.format(epoch), roi_cls_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train roi_loc_loss'.format(epoch), roi_loc_loss_avg.average, batch_num)

            pbar.update()


def save_coco_style(filename, preds, image_ids):
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

    with open(filename, 'w') as fout:
        json.dump(results, fout)


def validate(writer, model, val_loader, save_pred_filename='', save_gt_filename='', save_coco_filename=''):
    model.eval()
    with torch.no_grad():
        preds = []
        gt = []
        image_ids = []
        for batch_num, (data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels,
                               gt_count, data_batch_idx, gt_image_ids, ignore, imgs)) in \
                tqdm(enumerate(val_loader), total=len(val_loader), ncols=0, desc='Validation'):
            data = data.cuda()
            data_batch_idx = data_batch_idx.cuda()
            output = model(data, data_batch_idx)

            # get predicted boxes
            resized_shapes = {batch_idx: (d.shape[-1], d.shape[-2]) for batch_idx, d in enumerate(data)}
            orig_shapes = {batch_idx: img.size for batch_idx, img in enumerate(imgs)}
            batch_preds, pred_batch_indices = get_bboxes(output, resized_shapes, orig_shapes)

            batch_image_ids = []
            batch_gt = []
            for batch_idx in pred_batch_indices:
                scale_x = orig_shapes[batch_idx][0] / float(resized_shapes[batch_idx][0])
                scale_y = orig_shapes[batch_idx][1] / float(resized_shapes[batch_idx][1])
                batch_gt.append(defaultdict(lambda: {'rects': np.zeros((0, 4), dtype=np.float32),
                                                     'ignore': np.zeros((0,), dtype=np.bool)}))
                batch_gt_labels = gt_class_labels[batch_idx][:gt_count[batch_idx]].cpu().numpy().reshape(-1)
                for label in np.unique(batch_gt_labels):
                    label_idx = np.nonzero(batch_gt_labels == label)[0]
                    batch_gt[-1][label]['rects'] = gt_boxes[batch_idx][label_idx].numpy()
                    batch_gt[-1][label]['rects'][:, ::2] *= scale_x
                    batch_gt[-1][label]['rects'][:, 1::2] *= scale_y
                    batch_gt[-1][label]['ignore'] = ignore[batch_idx][label_idx]
                batch_image_ids.append(gt_image_ids[batch_idx])

            # append to output list
            preds.extend(batch_preds)
            gt.extend(batch_gt)
            image_ids.extend(batch_image_ids)

    preds = [dict(p) for p in preds]
    gt = [dict(g) for g in gt]

    if save_coco_filename:
        print(f'Saving {save_coco_filename}')
        save_coco_style(save_coco_filename, preds, image_ids)
    if save_pred_filename:
        print(f'Saving {save_pred_filename}')
        np.save(save_pred_filename, preds)
    if save_gt_filename:
        print(f'Saving {save_gt_filename}')
        np.save(save_gt_filename, gt)

    if args.dataset == 'coco':
        coco_gt_file = os.path.join(args.coco_root, 'annotations/instances_val2017.json')
        validate_map = compute_map_coco(coco_gt_file, preds, image_ids)
    else:
        validate_map = compute_map(preds, gt)
    print('mAP score @ 0.5 IoU: {:.5}'.format(validate_map))
    return validate_map


def eval_examples(writer, model, val_loader, num_shown_examples=10):
    model.eval()
    with torch.no_grad():
        total = min(int(num_shown_examples // args.test_batch_size), len(val_loader))
        with tqdm(total=total, ncols=0, desc='Val Examples') as pbar:
            shown_examples = 0
            for batch_num, (data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels,
                                   gt_count, data_batch_idx, gt_image_ids, imgs)) in enumerate(val_loader):
                data = data.cuda()
                data_batch_idx = data_batch_idx.cuda()
                output = model(data, data_batch_idx)
                batch_size = data.shape[0]

                resized_shapes = {batch_idx: (d.shape[-1], d.shape[-2]) for batch_idx, d in enumerate(data)}
                orig_shapes = {batch_idx: img.size for batch_idx, img in enumerate(imgs)}
                for batch_idx in range(batch_size):
                    if shown_examples >= num_shown_examples:
                        pbar.update()
                        return
                    img = imgs[batch_idx]

                    rect_list, text_list = get_display_boxes(output, orig_shapes, resized_shapes, batch_idx)
                    pred_img = draw_detections(img, rect_list, text_list)

                    writer.add_image('Images/val example {}'.format(shown_examples), tvtf.to_tensor(pred_img))

                    scale_x = orig_shapes[batch_idx][0] / resized_shapes[batch_idx][0]
                    scale_y = orig_shapes[batch_idx][1] / resized_shapes[batch_idx][1]
                    rect_list_gt = gt_boxes[batch_idx][:gt_count[batch_idx]].cpu().numpy()
                    rect_list_gt[:, ::2] *= scale_x
                    rect_list_gt[:, 1::2] *= scale_y
                    text_list_gt = [coco_id_to_name[lid.item()] if args.dataset == 'coco'
                                    else voc_id_to_name[lid.item()]
                                    for lid in gt_class_labels[batch_idx][:gt_count[batch_idx]]]
                    gt_img = draw_detections(img, rect_list_gt, text_list_gt)
                    writer.add_image('Images/val gt {}'.format(shown_examples), tvtf.to_tensor(gt_img))
                    shown_examples += 1

                pbar.update()


def compute_map_coco(gt_file, preds, pred_image_ids, iou_thresh=0.5):
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO

    tdir = tempfile.mkdtemp()
    try:
        temp_out = os.path.join(tdir, 'results.json')
        save_coco_style(temp_out, preds, pred_image_ids)

        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(temp_out)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.iouThrs = [iou_thresh]
        cocoEval.params.maxDets = [100]
        cocoEval.params.areaRng = ['all']
        cocoEval.evaluate()
        cocoEval.accumulate()
        precision = cocoEval.eval['precision'][0, :, 0, 0, 0]
        mean_ap = np.mean(precision[precision > -1])
    finally:
        shutil.rmtree(tdir, ignore_errors=True)

    return mean_ap


def main(writer):
    resize_width = 800
    resize_height = 800
    sub_sample = 16

    # define datasets
    train_loader, val_loader, anchor_boxes, num_classes = load_datasets(sub_sample, (resize_height, resize_width))

    # define model
    model = FasterRCNN(anchor_boxes, num_classes=num_classes, return_rpn_output=True)
    model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay, args.use_adam)

    # define criterion
    criterion = FasterRCNNCriterion()

    # define lr schedule
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_step_freq, args.lr_step_gamma)

    # load checkpoint
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        start_epoch, best_map = load_checkpoint(model, criterion, optimizer, lr_scheduler)

    # validate only if indicated
    if args.validate:
        eval_examples(writer, model, val_loader)
        if not args.quick_validate:
            validate_map = validate(writer, model, val_loader)
        return

    # train
    for epoch in range(start_epoch, args.num_epochs):
        train_epoch(epoch, writer, model, criterion, optimizer, lr_scheduler, train_loader)
        lr_scheduler.step()

        eval_examples(writer, model, val_loader)
        if args.quick_validate:
            validate_map = validate(writer, model, val_loader)

        is_best = validate_map > best_map
        if is_best:
            print("This is the best mAP score so far!")
            best_map = validate_map

        best_map = 0
        is_best = False
        save_checkpoint(model, criterion, optimizer, lr_scheduler, epoch, best_map, is_best)


if __name__ == '__main__':
    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = '../tensorboard/' + args.dataset + '_' + datestr
    print('Writing tensorboard output to', tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    try:
        main(writer)
    finally:
        writer.close()

