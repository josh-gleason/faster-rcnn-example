import os
from tqdm import tqdm
from datetime import datetime
import shutil
import argparse
from functools import partial
from collections import defaultdict

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tvt
import torchvision.transforms.functional as tvtf

from data.coco import CocoDetectionWithImgId, create_coco_targets
from data.voc import create_voc_targets
from data.general import faster_rcnn_collate_fn, ResizeAndPad

from utils.data_mappings import coco_num_obj_classes, coco_id_to_name, voc_num_obj_classes, voc_id_to_name
from utils.box_utils import define_anchor_boxes, get_bboxes_from_output
from utils.image_utils import draw_detections
from utils.metrics import compute_map, compute_map_coco, save_coco_style, AverageMeter

from models.faster_rcnn import FasterRCNN
from models.criterion import FasterRCNNCriterion


parser = argparse.ArgumentParser('Training code for simple Faster-RCNN implementation')
parser.add_argument('dataset', default='voc', type=str, choices={'voc', 'coco'},
                    help='Which dataset to train on')
parser.add_argument('--arch', default='vgg16', type=str,
                    choices={'vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'},
                    help='faster-rcnn backbone architecture (default vgg16)')
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


def load_datasets(sub_sample, resize_shape):
    anchor_boxes = define_anchor_boxes(sub_sample=sub_sample, height=resize_shape[0], width=resize_shape[1])

    # TODO Improve the transforms with better data augmentation
    train_transform = tvt.Compose([
        tvt.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ResizeAndPad(resize_shape),
        tvt.ToTensor(),
        tvt.Normalize([0.5] * 3, [0.5] * 3)
    ])
    val_transform = tvt.Compose([
        ResizeAndPad(resize_shape),
        tvt.ToTensor(),
        tvt.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if args.dataset == 'coco':
        print('Loading MSCOCO Detection dataset')
        base_transforms = partial(create_coco_targets, resize_shape=resize_shape, anchor_boxes=anchor_boxes)

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
        base_transforms = partial(create_voc_targets, resize_shape=resize_shape, anchor_boxes=anchor_boxes)

        train_transforms = partial(base_transforms, data_transform=train_transform)
        val_transforms = partial(base_transforms, data_transform=val_transform)

        download = not os.path.exists(os.path.join(args.voc_root, 'VOCdevkit/VOC2012'))
        train_dataset = datasets.VOCDetection(args.voc_root, year='2012', download=download,
                                              image_set='train',
                                              transforms=train_transforms)
        val_dataset = datasets.VOCDetection(args.voc_root, year='2012', download=download,
                                            image_set='val',
                                            transforms=val_transforms)

        bad_voc_train_indices = {2609}
        train_dataset = torch.utils.data.Subset(
            train_dataset, [i for i in range(len(train_dataset)) if i not in bad_voc_train_indices])
        bad_voc_val_indices = {478}
        val_dataset = torch.utils.data.Subset(
            val_dataset, [i for i in range(len(val_dataset)) if i not in bad_voc_val_indices])
        num_classes = voc_num_obj_classes
    else:
        raise ValueError

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
    out_dir = os.path.dirname(checkpoint_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_name)
    print('Finished saving checkpoint', checkpoint_name)
    if is_best:
        best_checkpoint_name = args.best_checkpoint_format.format(epoch=epoch, best_map=best_map)
        out_dir = os.path.dirname(best_checkpoint_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
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


def get_display_pred_boxes(output, resized_shapes, orig_shapes, batch_idx=0, top3=False):
    pred_boxes, batch_indices = get_bboxes_from_output(output, resized_shapes, orig_shapes)
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
            output_strs.append('{}({:.4f})'.format(
                coco_id_to_name[cls] if args.dataset == 'coco' else voc_id_to_name[cls],
                conf))

    return output_rects, output_strs


def get_display_gt_boxes(gt_boxes, gt_class_labels, gt_count, resized_shapes, orig_shapes, batch_idx=0):
    scale_x = resized_shapes[batch_idx][0] / orig_shapes[batch_idx][0]
    scale_y = resized_shapes[batch_idx][1] / orig_shapes[batch_idx][1]
    scale = min(scale_x, scale_y)

    rect_list_gt = gt_boxes[batch_idx][:gt_count[batch_idx]].cpu().numpy()
    rect_list_gt /= scale
    text_list_gt = [coco_id_to_name[lid.item()] if args.dataset == 'coco'
                    else voc_id_to_name[lid.item()]
                    for lid in gt_class_labels[batch_idx][:gt_count[batch_idx]]]

    rect_list_gt = rect_list_gt.astype(np.int32).tolist()

    return rect_list_gt, text_list_gt


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

                rect_list, text_list = get_display_pred_boxes(output, resized_shapes, orig_shapes, top3=True)
                pred_img = draw_detections(img, rect_list, text_list)

                rect_list_gt, text_list_gt = get_display_gt_boxes(gt_boxes, gt_class_labels, gt_count, resized_shapes,
                                                                  orig_shapes)
                gt_img = draw_detections(img, rect_list_gt, text_list_gt)

                writer.add_images('Train {}/example_gt_pred'.format(epoch),
                                  torch.stack((tvtf.to_tensor(gt_img), tvtf.to_tensor(pred_img)), dim=0))

                writer.add_scalar('Loss {}/train loss'.format(epoch), loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train rpn_obj_loss'.format(epoch), rpn_obj_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train rpn_loc_loss'.format(epoch), rpn_loc_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train roi_cls_loss'.format(epoch), roi_cls_loss_avg.average, batch_num)
                writer.add_scalar('Loss {}/train roi_loc_loss'.format(epoch), roi_loc_loss_avg.average, batch_num)

            pbar.update()


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
            batch_preds, pred_batch_indices = get_bboxes_from_output(output, resized_shapes, orig_shapes)

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
        validate_map = compute_map(gt, preds)
    print('mAP score @ 0.5 IoU: {:.5}'.format(validate_map))
    return validate_map


def eval_examples(writer, model, val_loader, num_shown_examples=10):
    model.eval()
    with torch.no_grad():
        total = min(int(num_shown_examples // args.test_batch_size), len(val_loader))
        with tqdm(total=total, ncols=0, desc='Val Examples') as pbar:
            shown_examples = 0
            for batch_num, (data, (anchor_obj, anchor_loc, gt_boxes, gt_class_labels,
                                   gt_count, data_batch_idx, gt_image_ids, ignore, imgs)) in enumerate(val_loader):
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

                    rect_list_pred, text_list_pred = get_display_pred_boxes(output, resized_shapes, orig_shapes, batch_idx)
                    pred_img = draw_detections(img, rect_list_pred, text_list_pred)

                    rect_list_gt, text_list_gt = get_display_gt_boxes(gt_boxes, gt_class_labels, gt_count, resized_shapes,
                                                                      orig_shapes, batch_idx)
                    gt_img = draw_detections(img, rect_list_gt, text_list_gt)

                    writer.add_images('Validate/example_{}_gt_pred'.format(shown_examples),
                                      torch.stack((tvtf.to_tensor(gt_img), tvtf.to_tensor(pred_img)), dim=0))

                    shown_examples += 1

                pbar.update()


def main(writer):
    resize_width = 800
    resize_height = 800
    sub_sample = 16

    # define datasets
    train_loader, val_loader, anchor_boxes, num_classes = load_datasets(sub_sample, (resize_height, resize_width))

    # define model
    model = FasterRCNN(anchor_boxes, num_classes=num_classes, return_rpn_output=True, arch=args.arch)
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
        if not args.quick_validate:
            validate_map = validate(writer, model, val_loader)
        else:
            validate_map = 0.0

        writer.add_scalar('Validate/mAP', validate_map, epoch)

        is_best = validate_map > best_map
        if is_best:
            print("This is the best mAP score so far!")
            best_map = validate_map

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

