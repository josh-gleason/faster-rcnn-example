import os
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

from utils.data_mappings import coco_num_obj_classes, coco_id_to_name
from utils.box_utils import compute_iou, choose_anchor_subset, get_loc_labels, define_anchor_boxes, get_boxes_from_loc, apply_nms
from utils.image_utils import draw_detections
from models.faster_rcnn import FasterRCNN

parser = argparse.ArgumentParser('Training code for simple Faster-RCNN implementation')
parser.add_argument('--coco-root', '-cr', default='../coco', type=str,
                    help='Root directory of COCO dataset')
parser.add_argument('--train-batch-size', '-b', default=1, type=int,
                    help='Training batch size')
parser.add_argument('--test-batch-size', '-tb', default=2, type=int,
                    help='Test batch size')
parser.add_argument('--num-workers', '-j', default=8, type=int,
                    help='Number of workers')
parser.add_argument('--learning-rate', '-lr', default=0.01, type=float,
                    help='Initial learning rate')
parser.add_argument('--momentum', '-m', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--weight-decay', '-wd', default=1e-5, type=float,
                    help='Weight decay parameter for SGD')
parser.add_argument('--num-epochs', '-ne', default=20, type=int,
                    help='Number of epochs')
parser.add_argument('--lr-step-freq', '-s', default=5, type=int,
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
args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def create_targets(data, labels, data_transform, resize_shape, anchor_boxes, valid_anchors, pos_iou_thresh=0.7,
                   neg_iou_thresh=0.3, pos_ratio=0.5, num_samples=256, mark_max_gt_anchors=True):
    orig_width, orig_height = data.size
    new_width, new_height = resize_shape

    num_anchors = anchor_boxes.shape[0]

    if len(labels) > 0:
        gt_class_labels = np.array([label['category_id'] for label in labels], dtype=np.int32)
        gt_boxes = np.array([label['bbox'] for label in labels], dtype=np.float32)

        gt_boxes[:, 2:] += gt_boxes[:, 0:2]

        scale_x = float(new_width) / orig_width
        scale_y = float(new_height) / orig_height
        gt_boxes[:, ::2] *= scale_x
        gt_boxes[:, 1::2] *= scale_y
    else:
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
    return data_transform(data), (anchor_obj_final, anchor_loc_final, gt_boxes, gt_class_labels)


def pad_zero_rows(ary, total_rows):
    """ pad end of dimension 0 with zeros to ensure ary.shape[0] = total_rows """
    return np.pad(ary, [[0, total_rows - ary.shape[0]]] + [[0, 0]] * (len(ary.shape) - 1))


def faster_rcnn_collate_fn(batch):
    # pad boxes with additional zeros so they are all the same size, this allows DataParallel to scatter properly
    gt_count = [len(b[1][3]) for b in batch]
    gt_max_count = np.max(gt_count)

    gt_boxes = [pad_zero_rows(b[1][2], gt_max_count) for b in batch]
    gt_class_labels = [pad_zero_rows(b[1][3], gt_max_count) for b in batch]

    batch = [(b[0], (b[1][0], b[1][1], boxes, labels, counts))
             for b, boxes, labels, counts in zip(batch, gt_boxes, gt_class_labels, gt_count)]
    return torch.utils.data.dataloader.default_collate(batch)


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

    base_transforms = partial(create_targets, resize_shape=resize_shape, anchor_boxes=anchor_boxes,
                              valid_anchors=valid_anchors)

    train_transforms = partial(base_transforms, data_transform=train_transform)
    val_transforms = partial(base_transforms, data_transform=val_transform)

    train_dataset = datasets.CocoDetection(
        os.path.join(args.coco_root, 'images/val2017'),
        os.path.join(args.coco_root, 'annotations/instances_val2017.json'),
        # os.path.join(args.coco_root, 'images/train2017'),
        # os.path.join(args.coco_root, 'annotations/instances_train2017.json'),
        transforms=train_transforms,
    )

    val_dataset = datasets.CocoDetection(
        os.path.join(args.coco_root, 'images/val2017'),
        os.path.join(args.coco_root, 'annotations/instances_val2017.json'),
        transforms=val_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        # sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=1000),    # TODO
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=faster_rcnn_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=faster_rcnn_collate_fn
    )

    return train_loader, val_loader, anchor_boxes


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
    def __init__(self, lambda_rpn_loc=10., lambda_roi_loc=0.1):
        super().__init__()
        self.lambda_rpn_loc = lambda_rpn_loc
        self.lambda_roi_loc = lambda_roi_loc

    def forward(self, pred_roi_batch_idx, pred_roi_boxes, pred_roi_cls, pred_roi_loc,
                pred_loc, pred_obj, pred_roi_cls_labels, pred_roi_loc_labels, anchor_obj,
                anchor_loc):
        rpn_loc_loss = self.lambda_rpn_loc * self.get_rpn_loc_loss(pred_loc, anchor_loc, anchor_obj)
        rpn_obj_loss = self.get_rpn_obj_loss(pred_obj, anchor_obj)
        roi_loc_loss = self.lambda_roi_loc * self.get_roi_loc_loss(pred_roi_loc, pred_roi_loc_labels, pred_roi_cls_labels)
        roi_cls_loss = self.get_roi_cls_loss(pred_roi_cls, pred_roi_cls_labels)

        loss = rpn_obj_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss

        return loss, rpn_obj_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss

    @staticmethod
    def get_rpn_loc_loss(pred_loc, gt_loc, gt_obj):
        positive_gt_b, positive_gt_c = torch.where(gt_obj > 0)
        if positive_gt_b.shape[0] == 0:
            return torch.sum(0. * pred_loc)

        pred_loc = pred_loc[positive_gt_b, positive_gt_c, :]
        gt_loc = gt_loc[positive_gt_b, positive_gt_c, :]
        loc_loss = F.smooth_l1_loss(pred_loc, gt_loc, reduction='none')
        return torch.sum(loc_loss) / torch.sum(gt_obj >= 0).float()

    @staticmethod
    def get_rpn_obj_loss(pred_obj, gt_obj):
        valid_gt_b, valid_gt_c = torch.where(gt_obj > -1)
        if valid_gt_b.shape[0] == 0:
            return torch.sum(0. * pred_obj)

        num_classes = pred_obj.shape[-1]
        pred_obj = pred_obj[valid_gt_b, valid_gt_c, :].view(-1, num_classes)
        gt_obj = gt_obj[valid_gt_b, valid_gt_c].view(-1)
        return F.cross_entropy(pred_obj, gt_obj)

    @staticmethod
    def get_roi_cls_loss(pred_roi_cls, gt_roi_cls):
        if gt_roi_cls.shape[0] == 0:
            return torch.sum(0. * pred_roi_cls)

        return F.cross_entropy(pred_roi_cls, gt_roi_cls)

    @staticmethod
    def get_roi_loc_loss(pred_roi_loc, gt_roi_loc, gt_roi_cls):
        positive_idx = torch.where(gt_roi_cls > 0)[0]
        if positive_idx.shape[0] == 0:
            return torch.sum(0. * pred_roi_loc)

        gt_roi_cls = gt_roi_cls[positive_idx]
        gt_roi_loc = gt_roi_loc[positive_idx, :]
        pred_roi_loc = pred_roi_loc[positive_idx, gt_roi_cls, :]
        loc_loss = F.smooth_l1_loss(pred_roi_loc, gt_roi_loc, reduction='none')
        return torch.sum(loc_loss) / torch.sum(gt_roi_cls >= 0).float()


def get_display_boxes(output, top3=False):
    pred_roi_cls = torch.softmax(output['pred_roi_cls'], dim=1).detach().cpu().numpy()
    pred_roi_batch_idx = output['pred_roi_batch_idx'].detach().cpu().numpy()
    pred_roi_boxes = output['pred_roi_boxes'].detach().cpu().numpy()
    pred_roi_loc = output['pred_roi_loc'].detach().cpu().numpy()

    pred_roi_cls = pred_roi_cls[pred_roi_batch_idx == 0]
    pred_roi_boxes = pred_roi_boxes[pred_roi_batch_idx == 0, :]
    pred_roi_loc = pred_roi_loc[pred_roi_batch_idx == 0, :]

    pred_cls = np.argmax(pred_roi_cls, axis=1)
    pred_conf = np.max(pred_roi_cls, axis=1)
    keep_idx = np.nonzero(pred_cls > 0)[0]

    text_list = []
    rect_list = []
    if len(keep_idx) > 0:
        pred_cls = pred_cls[keep_idx]
        pred_conf = pred_conf[keep_idx]
        pred_boxes = pred_roi_boxes[keep_idx]
        pred_loc = pred_roi_loc[keep_idx, pred_cls, :]

        rects = get_boxes_from_loc(pred_boxes, pred_loc, img_width=800, img_height=800, loc_mean=np.array((0., 0., 0., 0.)), loc_std=np.array((0.1, 0.1, 0.2, 0.2)))
        post_nms = defaultdict(list)
        for rect, conf, cls in zip(rects, pred_conf, pred_cls):
            if cls > 0:
                post_nms[cls].append((rect, conf))

        post_nms_rects = []
        post_nms_confs = []
        post_nms_labels = []
        rect_list = []
        for cls in post_nms:
            cls_rects, cls_conf = zip(*post_nms[cls])
            cls_rects = np.concatenate([c.reshape(1, -1) for c in cls_rects], axis=0)
            cls_conf = np.array(cls_conf)

            cls_rects = torch.from_numpy(cls_rects).float()
            cls_conf = torch.from_numpy(cls_conf).float()
            keep_nms = ops.nms(cls_rects, cls_conf, 0.5)
            post_nms_rects.append(cls_rects[keep_nms].numpy())
            post_nms_confs.append(cls_conf[keep_nms].numpy())
            post_nms_labels.append(np.full(len(keep_nms), cls, dtype=np.int32))
        post_nms_rects = np.concatenate(post_nms_rects, axis=0)
        post_nms_confs = np.concatenate(post_nms_confs, axis=0)
        post_nms_labels = np.concatenate(post_nms_labels, axis=0)

        keep_idx = np.nonzero(post_nms_confs > 0.7)[0]

        if len(keep_idx) == 0 and top3:
            keep_idx = np.argsort(post_nms_confs)[:-4:-1]

        if len(keep_idx) > 0:
            post_nms_rects = post_nms_rects[keep_idx]
            post_nms_confs = post_nms_confs[keep_idx]
            post_nms_labels = post_nms_labels[keep_idx]

            rect_list = np.round(post_nms_rects).astype(np.int32).tolist()

            for cls_id, conf in zip(post_nms_labels, post_nms_confs):
                text_list.append('{}({:.4f}'.format(coco_id_to_name[cls_id], conf))

    return rect_list, text_list


def train_epoch(epoch, writer, model, criterion, optimizer, lr_scheduler, train_loader):
    model.train()
    optimizer.zero_grad()
    loss_avg = AverageMeter()
    rpn_obj_loss_avg = AverageMeter()
    rpn_loc_loss_avg = AverageMeter()
    roi_cls_loss_avg = AverageMeter()
    roi_loc_loss_avg = AverageMeter()
    with tqdm(total=len(train_loader), ncols=0, desc='Training Epoch {}'.format(epoch)) as pbar:
        for batch_num, (data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels, gt_count)) in enumerate(train_loader):
            data = data.cuda()
            anchor_obj = anchor_obj.cuda()
            anchor_loc = anchor_loc.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_class_labels = gt_class_labels.cuda()

            output = model(data, gt_boxes, gt_class_labels, gt_count)
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

            if batch_num % 200 == 0:
                img = tvtf.to_pil_image(torch.clamp(data[0, :, :, :] * 0.5 + 0.5, 0, 1).cpu())

                model.eval()
                test_img = data[0:1, :, :, :]
                output = model(test_img)
                model.train()

                rect_list, text_list = get_display_boxes(output)
                pred_img = draw_detections(img, rect_list, text_list)

                writer.add_image('Images/train_{}'.format(epoch), tvtf.to_tensor(pred_img))

                writer.add_scalar('Loss {}/train loss'.format(epoch), loss_avg.avg, batch_num)
                writer.add_scalar('Loss {}/train rpn_obj_loss'.format(epoch), rpn_obj_loss_avg.avg, batch_num)
                writer.add_scalar('Loss {}/train rpn_loc_loss'.format(epoch), rpn_loc_loss_avg.avg, batch_num)
                writer.add_scalar('Loss {}/train roi_cls_loss'.format(epoch), roi_cls_loss_avg.avg, batch_num)
                writer.add_scalar('Loss {}/train roi_loc_loss'.format(epoch), roi_loc_loss_avg.avg, batch_num)

            if (batch_num + 1) % args.batches_per_optimizer_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update()


def validate(writer, model, val_loader):
    # TODO
    pass


def main(writer):
    resize_width = 800
    resize_height = 800
    sub_sample = 16

    # define datasets
    train_loader, val_loader, anchor_boxes = load_datasets(sub_sample, (resize_height, resize_width))

    # define model
    model = FasterRCNN(anchor_boxes, num_classes=coco_num_obj_classes, return_rpn_output=True)
    model = torch.nn.DataParallel(model).cuda()
    # model.cuda()

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

    # train
    for epoch in range(start_epoch, args.num_epochs):
        train_epoch(epoch, writer, model, criterion, optimizer, lr_scheduler, train_loader)
        # TODO
        # validate_map = validate(writer, model, val_loader)
        # is_best = validate_map > best_map
        # if is_best:
        #     print("This is the best mAP score so far!")
        #     best_map = validate_map
        best_map = 0
        is_best = False
        save_checkpoint(model, criterion, optimizer, lr_scheduler, epoch, best_map, is_best)
        lr_scheduler.step()


if __name__ == '__main__':
    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = '../tensorboard/' + datestr
    print('Writing tensorboard output to', tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    try:
        main(writer)
    finally:
        writer.close()

