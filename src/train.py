import os
from tqdm import tqdm
from datetime import datetime
import shutil
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tvt
from torchvision.datasets.vision import StandardTransform
import torchvision.transforms.functional as tvtf

from data.coco import CocoDetectionWithImgId, FormatCOCOLabels
from data.voc import FormatVOCLabels
from data.general import faster_rcnn_collate_fn, DynamicResize, PadToShape, ComposeTransforms, RandomHorizontalFlip, \
    CreateRPNLabels

from utils.data_mappings import coco_id_to_name, voc_id_to_name, num_obj_classes_dict
from utils.box_utils import get_boxes_from_output, get_display_gt_boxes, get_display_pred_boxes
from utils.image_utils import draw_detections
from utils.metrics import compute_map, compute_map_coco, save_coco_style, AverageMeter

from models.faster_rcnn import FasterRCNN
from models.criterion import FasterRCNNCriterion

from third_party.eval_tool import compute_map_voc

# TODO use config files instead
# TODO compute anchors only once you actually know the shape of the feature map

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
parser.add_argument('--test-batch-size', '-tb', default=1, type=int,
                    help='Test batch size')
parser.add_argument('--num-workers', '-j', default=8, type=int,
                    help='Number of workers')
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float,
                    help='Initial learning rate')
parser.add_argument('--momentum', '-m', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float,
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


def load_datasets(min_shape=(600, 600), max_shape=(1000, 1000), sub_sample=16, ceil_mode=False,
                  pad_to_max=True, stretch_to_max=False):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]
    train_transforms_list = [
        RandomHorizontalFlip(0.5),
        DynamicResize(min_shape, max_shape, stretch_to_max=stretch_to_max),
        StandardTransform(tvt.ToTensor()),
        StandardTransform(tvt.Normalize(mean_val, std_val)),
    ]
    val_transforms_list = [
        DynamicResize(min_shape, max_shape, stretch_to_max=stretch_to_max),
        StandardTransform(tvt.ToTensor()),
        StandardTransform(tvt.Normalize(mean_val, std_val)),
    ]
    if pad_to_max:
        train_transforms_list.append(StandardTransform(PadToShape(max_shape)))
        val_transforms_list.append(StandardTransform(PadToShape(max_shape)))
    train_transforms_list.append(CreateRPNLabels(sub_sample=sub_sample, ceil_mode=ceil_mode))
    val_transforms_list.append(CreateRPNLabels(sub_sample=sub_sample, ceil_mode=ceil_mode))

    if args.dataset == 'coco':
        print('Loading MSCOCO Detection dataset')
        train_transforms_list.insert(0, FormatCOCOLabels())
        val_transforms_list.insert(0, FormatCOCOLabels())

        train_transforms = ComposeTransforms(train_transforms_list)
        val_transforms = ComposeTransforms(val_transforms_list)

        train_dataset = CocoDetectionWithImgId(args.coco_root, image_set='train', download=True,
                                               transforms=train_transforms)
        val_dataset = CocoDetectionWithImgId(args.coco_root, image_set='val', download=True,
                                             transforms=val_transforms)
    elif args.dataset == 'voc':
        print('Loading Pascal VOC 2007 Detection dataset')
        train_transforms_list.insert(0, FormatVOCLabels(use_difficult=False))
        val_transforms_list.insert(0, FormatVOCLabels(use_difficult=True))

        train_transforms = ComposeTransforms(train_transforms_list)
        val_transforms = ComposeTransforms(val_transforms_list)

        download = not os.path.exists(os.path.join(args.voc_root, 'VOCdevkit/VOC2007'))
        train_dataset = torch.utils.data.ConcatDataset([
            datasets.VOCDetection(args.voc_root, year='2007', download=download,
                                  image_set='train', transforms=train_transforms),
            datasets.VOCDetection(args.voc_root, year='2007', download=download,
                                  image_set='val', transforms=train_transforms)])

        val_dataset = datasets.VOCDetection(args.voc_root, year='2007', download=download,
                                            image_set='test', transforms=val_transforms)
    else:
        raise ValueError

    train_sampler = torch.utils.data.RandomSampler(train_dataset)

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

    return train_loader, val_loader


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


def get_optimizer(model, lr=0.01, weight_decay=5e-4, use_adam=True):
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
        return torch.optim.SGD(params, lr=lr, momentum=0.9)


def train_epoch(epoch, writer, model, criterion, optimizer, train_loader, average_meters):
    model.train()
    optimizer.zero_grad()

    id_to_name_map = coco_id_to_name if args.dataset == 'coco' else voc_id_to_name
    with tqdm(total=len(train_loader), ncols=0, desc='Training Epoch {}'.format(epoch)) as pbar:
        for batch_num, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            gt_boxes = labels['gt_boxes'].cuda()
            gt_class_labels = labels['gt_class_labels'].cuda()
            gt_count = labels['gt_count'].cuda()
            rpn_obj_labels = labels['rpn_obj_label'].cuda()
            rpn_loc_labels = labels['rpn_loc_label'].cuda()
            initial_batch_idx = labels['batch_idx'].cuda()
            valid_shapes = labels['valid_shape']
            original_imgs = labels['original_image']

            output = model(data, initial_batch_idx, gt_boxes, gt_class_labels, gt_count)
            loss, rpn_obj_loss, rpn_loc_loss, roi_cls_loss, roi_loc_loss \
                = criterion(**output, anchor_obj=rpn_obj_labels, anchor_loc=rpn_loc_labels)

            batch_size = data.shape[0]
            average_meters['loss_avg'].update(loss.item(), batch_size)
            average_meters['rpn_obj_loss_avg'].update(rpn_obj_loss.item(), batch_size)
            average_meters['rpn_loc_loss_avg'].update(rpn_loc_loss.item(), batch_size)
            average_meters['roi_cls_loss_avg'].update(roi_cls_loss.item(), batch_size)
            average_meters['roi_loc_loss_avg'].update(roi_loc_loss.item(), batch_size)

            loss_scaled = loss / float(args.batches_per_optimizer_step)
            loss_scaled.backward()

            if (batch_num + 1) % args.batches_per_optimizer_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_num % args.print_freq == 0:
                img = original_imgs[0]

                with torch.no_grad():
                    model.eval()
                    test_img = data[0:1, :, :, :]
                    output = model(test_img, torch.tensor((0,)).to(device=test_img.device))
                    model.train()

                orig_shapes = {batch_idx: (img.size[1], img.size[0]) for batch_idx, img in enumerate(original_imgs)}

                rect_list, text_list = get_display_pred_boxes(output, valid_shapes, orig_shapes, id_to_name_map,
                                                              top3=True)
                pred_img = draw_detections(img, rect_list, text_list)

                rect_list_gt, text_list_gt = get_display_gt_boxes(gt_boxes, gt_class_labels, valid_shapes, orig_shapes,
                                                                  id_to_name_map, gt_count=gt_count, batch_idx=0)
                gt_img = draw_detections(img, rect_list_gt, text_list_gt)

                writer.add_images('Train Epoch {}/example_gt_pred'.format(epoch),
                                  torch.stack((tvtf.to_tensor(gt_img), tvtf.to_tensor(pred_img)), dim=0),
                                  global_step=batch_num // args.print_freq)

                writer.add_scalar('Loss/train loss'.format(epoch), average_meters['loss_avg'].average,
                                  average_meters['loss_avg'].count)
                writer.add_scalar('Loss/train rpn_obj_loss'.format(epoch), average_meters['rpn_obj_loss_avg'].average,
                                  average_meters['rpn_obj_loss_avg'].count)
                writer.add_scalar('Loss/train rpn_loc_loss'.format(epoch), average_meters['rpn_loc_loss_avg'].average,
                                  average_meters['rpn_loc_loss_avg'].count)
                writer.add_scalar('Loss/train roi_cls_loss'.format(epoch), average_meters['roi_cls_loss_avg'].average,
                                  average_meters['roi_cls_loss_avg'].count)
                writer.add_scalar('Loss/train roi_loc_loss'.format(epoch), average_meters['roi_loc_loss_avg'].average,
                                  average_meters['roi_loc_loss_avg'].count)

            pbar.update()


def validate(epoch, writer, model, val_loader, save_pred_filename='', save_gt_filename='', save_coco_filename=''):
    model.eval()
    with torch.no_grad():
        preds = []
        gt = []
        image_ids = []
        for batch_num, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader), ncols=0, desc='Validation'):
            data = data.cuda()
            data_batch_idx = labels['batch_idx'].cuda()
            gt_boxes = labels['gt_boxes']
            gt_class_labels = labels['gt_class_labels']
            gt_count = labels['gt_count']
            valid_shapes = labels['valid_shape']
            original_imgs = labels['original_image']
            difficult = labels['difficult']
            gt_image_ids = labels['image_id']
            ignore = labels['ignore']
            output = model(data, data_batch_idx)

            # get predicted boxes
            orig_shapes = {batch_idx: (img.size[1], img.size[0]) for batch_idx, img in enumerate(original_imgs)}
            batch_preds, pred_batch_indices = get_boxes_from_output(output, valid_shapes, orig_shapes)

            batch_image_ids = []
            batch_gt = []
            for batch_idx in pred_batch_indices:
                scale_x = orig_shapes[batch_idx][1] / float(valid_shapes[batch_idx][1])
                scale_y = orig_shapes[batch_idx][0] / float(valid_shapes[batch_idx][0])
                batch_gt.append(defaultdict(lambda: {'rects': np.zeros((0, 4), dtype=np.float32),
                                                     'ignore': np.zeros((0,), dtype=np.bool),
                                                     'difficult': np.zeros((0,), dtype=np.bool)}))
                batch_gt_labels = gt_class_labels[batch_idx][:gt_count[batch_idx]].cpu().numpy().reshape(-1)
                for label in np.unique(batch_gt_labels):
                    label_idx = np.nonzero(batch_gt_labels == label)[0]
                    batch_gt[-1][label]['rects'] = gt_boxes[batch_idx][label_idx].numpy()
                    batch_gt[-1][label]['rects'][:, ::2] *= scale_x
                    batch_gt[-1][label]['rects'][:, 1::2] *= scale_y
                    batch_gt[-1][label]['ignore'] = ignore[batch_idx][label_idx]
                    batch_gt[-1][label]['difficult'] = difficult[batch_idx][label_idx]
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
        # TODO explore differences between my implementation (compute_map) and third party (compute_map_voc)
        # validate_map = compute_map(gt, preds)
        validate_map = compute_map_voc(gt, preds, use_07_metric=True)

    print('mAP score @ 0.5 IoU: {:.5}'.format(validate_map))
    writer.add_scalar('Validate/mAP', validate_map, epoch)
    return validate_map


def eval_examples(epoch, writer, model, val_loader, num_shown_examples=10):
    id_to_name_map = coco_id_to_name if args.dataset == 'coco' else voc_id_to_name

    model.eval()
    with torch.no_grad():
        total = min(int(np.ceil(num_shown_examples / args.test_batch_size)), len(val_loader))
        with tqdm(total=total, ncols=0, desc='Val Examples') as pbar:
            shown_examples = 0
            for batch_num, (data, labels) in enumerate(val_loader):
                data = data.cuda()
                data_batch_idx = labels['batch_idx'].cuda()
                output = model(data, data_batch_idx)
                batch_size = data.shape[0]

                gt_boxes = labels['gt_boxes']
                gt_class_labels = labels['gt_class_labels']
                gt_count = labels['gt_count']
                valid_shapes = labels['valid_shape']
                original_imgs = labels['original_image']

                orig_shapes = {batch_idx: (img.size[1], img.size[0]) for batch_idx, img in enumerate(original_imgs)}
                for batch_idx in range(batch_size):
                    img = original_imgs[batch_idx]

                    rect_list_pred, text_list_pred = get_display_pred_boxes(output, valid_shapes, orig_shapes,
                                                                            id_to_name_map, batch_idx)
                    pred_img = draw_detections(img, rect_list_pred, text_list_pred)

                    rect_list_gt, text_list_gt = get_display_gt_boxes(gt_boxes, gt_class_labels, valid_shapes,
                                                                      orig_shapes, id_to_name_map, gt_count, batch_idx)
                    gt_img = draw_detections(img, rect_list_gt, text_list_gt)

                    writer.add_images('Validate/example_{}_gt_pred'.format(shown_examples),
                                      torch.stack((tvtf.to_tensor(gt_img), tvtf.to_tensor(pred_img)), dim=0),
                                      global_step=epoch)

                    shown_examples += 1
                    if shown_examples >= num_shown_examples:
                        break

                pbar.update()
                if shown_examples >= num_shown_examples:
                    break


def train(writer):
    min_height = 600
    min_width = 600
    max_height = 1000
    max_width = 1000

    num_classes = num_obj_classes_dict[args.dataset]

    # define model
    model = FasterRCNN(num_classes=num_classes, return_rpn_output=True, arch=args.arch)
    model = torch.nn.DataParallel(model).cuda()

    # define datasets
    # TODO if pad_to_max==False and stretch_to_max==False then don't allow train/test batch size > 1
    train_loader, val_loader = load_datasets((min_height, min_width), (max_height, max_width),
                                             sub_sample=model.module.get_sub_sample(),
                                             ceil_mode=model.module.get_ceil_mode(),
                                             pad_to_max=False, stretch_to_max=False)

    # define optimizer
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay, args.use_adam)

    # define criterion
    criterion = FasterRCNNCriterion()

    # define lr schedule
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_freq, args.lr_step_gamma)

    # load checkpoint
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        start_epoch, best_map = load_checkpoint(model, criterion, optimizer, lr_scheduler)

    # validate only if indicated
    if args.validate:
        eval_examples(start_epoch, writer, model, val_loader)
        if not args.quick_validate:
            validate(start_epoch, writer, model, val_loader)
        return

    # train
    average_meters = {
        'loss_avg': AverageMeter(0.998),
        'rpn_obj_loss_avg': AverageMeter(0.998),
        'rpn_loc_loss_avg': AverageMeter(0.998),
        'roi_cls_loss_avg': AverageMeter(0.998),
        'roi_loc_loss_avg': AverageMeter(0.998)}
    for epoch in range(start_epoch, args.num_epochs):
        train_epoch(epoch, writer, model, criterion, optimizer, train_loader, average_meters)
        lr_scheduler.step()

        # save a checkpoint now, will overwrite with real one after validation (useful if validation crashes)
        save_checkpoint(model, criterion, optimizer, lr_scheduler, epoch, -1.0, False)

        eval_examples(epoch, writer, model, val_loader)
        if not args.quick_validate:
            validate_map = validate(epoch, writer, model, val_loader)
        else:
            validate_map = 0.0

        is_best = validate_map > best_map
        if is_best:
            print("This is the best mAP score so far!")
            best_map = validate_map

        save_checkpoint(model, criterion, optimizer, lr_scheduler, epoch, best_map, is_best)


def main():
    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = '../tensorboard/' + args.dataset + '_' + args.arch + '_' + datestr
    print('Writing tensorboard output to', tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    try:
        train(writer)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
