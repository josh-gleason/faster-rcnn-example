import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, vgg16
from torchvision.models.utils import load_state_dict_from_url
import torchvision.ops as ops
import numpy as np
from utils.box_utils import compute_iou, choose_anchor_subset, get_loc_labels, get_boxes_from_loc_batch
import matplotlib.pyplot as plt


def show_feat_map(x, title='feats'):
    x_np = x[0, ...].detach().cpu().numpy()
    ch = x.shape[1]
    r = ch // 2**int(np.ceil(np.log2(ch) / 2))
    c = ch // r
    # r = c = int(np.ceil(ch**0.5))
    h, w = x.shape[-2:]
    res = np.zeros((r * h, c * w))
    for row in range(r):
        for col in range(c):
            ch_id = row * c + col
            if ch_id < ch:
                x_ch = x_np[ch_id, :, :]
                mn = np.min(x_ch)
                mx = np.max(x_ch)
                if mx - mn > 0:
                    res[row * h:(row + 1) * h, col * w:(col + 1) * w] = (x_ch - mn) / (mx - mn)
                else:
                    res[row * h:(row + 1) * h, col * w:(col + 1) * w] = x_ch * 0.0
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.subplots()
    ax.imshow(res)
    ax.set_title(title + ' mine')
    fig.show()
    # breakpoint()


class ResNetWrapper(resnet.ResNet):
    arch_kwargs = {
        'resnet18': dict(block=resnet.BasicBlock, layers=[2, 2, 2, 2]),
        'resnet34': dict(block=resnet.BasicBlock, layers=[3, 4, 6, 3]),
        'resnet50': dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3]),
        'resnet101': dict(block=resnet.Bottleneck, layers=[3, 4, 23, 3]),
        'resnet152': dict(block=resnet.Bottleneck, layers=[3, 8, 36, 3])}

    layer3_out_channels = {
        'resnet18': 256,
        'resnet34': 256,
        'resnet50': 1024,
        'resnet101': 1024,
        'resnet152': 1024}

    layer4_out_channels = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048}

    def __init__(self, arch='resnet18'):
        super().__init__(**self.arch_kwargs[arch])
        self.arch = arch
        self.load_state_dict(load_state_dict_from_url(resnet.model_urls[arch]))


class FeatureNetResNet(ResNetWrapper):
    def __init__(self, arch='resnet18'):
        super().__init__(arch)
        del self.avgpool
        del self.layer4
        del self.fc

    def get_out_channels(self):
        return self.layer3_out_channels[self.arch]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class HeadResNet(ResNetWrapper):
    def __init__(self, num_classes, arch='resnet18'):
        super().__init__(arch)
        self.roi_align_size = (7, 7)
        self.spatial_scale = 1.0 / 16.0
        self.feature_size = self.layer4_out_channels[arch]

        del self.conv1
        del self.bn1
        del self.relu
        del self.maxpool
        del self.layer1
        del self.layer2
        del self.layer3
        del self.fc

        self.fc_loc = nn.Linear(self.feature_size, num_classes * 4)
        self.fc_cls = nn.Linear(self.feature_size, num_classes)

        self.fc_loc.weight.data.normal_(0, 0.01)
        self.fc_loc.bias.data.zero_()
        self.fc_cls.weight.data.normal_(0, 0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, x, pred_boxes, pred_batch_idx):
        num_regions = len(pred_batch_idx)

        pred_indices_and_boxes = np.concatenate((pred_batch_idx.reshape(-1, 1), pred_boxes), axis=1)
        pred_indices_and_boxes = torch.from_numpy(pred_indices_and_boxes).to(x)

        # TODO should I use roi_align?
        # regions = ops.roi_align(x, pred_indices_and_boxes, self.roi_align_size, self.spatial_scale)
        regions = ops.roi_pool(x, pred_indices_and_boxes, self.roi_align_size, self.spatial_scale)
        y = self.avgpool(self.layer4(regions))
        y = torch.flatten(y, start_dim=1)

        pred_roi_cls = self.fc_cls(y)
        pred_roi_loc = self.fc_loc(y).view(num_regions, -1, 4)

        return pred_roi_cls, pred_roi_loc


class FeatureNetVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        model = vgg16(pretrained=True)
        features = list(model.features)[:30]
        # freeze top4 conv
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.features = nn.Sequential(*features)

    def get_out_channels(self):
        return 512

    def forward(self, x):
        x = self.features(x)
        return x


class HeadVGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.roi_align_size = (7, 7)
        self.spatial_scale = 1.0 / 16.0

        model = vgg16(pretrained=True)
        classifier = model.classifier
        classifier = list(classifier)
        # remove final fc layer
        del classifier[6]
        # remove drop-out
        del classifier[5]
        del classifier[2]
        self.classifier = nn.Sequential(*classifier)

        self.fc_loc = nn.Linear(4096, num_classes * 4)
        self.fc_cls = nn.Linear(4096, num_classes)

        self.fc_loc.weight.data.normal_(0, 0.001)
        self.fc_loc.bias.data.zero_()
        self.fc_cls.weight.data.normal_(0, 0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, x, pred_boxes, pred_batch_idx):
        num_regions = len(pred_batch_idx)

        pred_indices_and_boxes = np.concatenate((pred_batch_idx.reshape(-1, 1).astype(np.float32), pred_boxes), axis=1)
        pred_indices_and_boxes = torch.from_numpy(pred_indices_and_boxes).to(x)

        # TODO should I use roi_align?
        # regions = ops.roi_align(x, pred_indices_and_boxes, self.roi_align_size, self.spatial_scale)
        regions = ops.roi_pool(x, pred_indices_and_boxes, self.roi_align_size, self.spatial_scale)
        y = self.classifier(torch.flatten(regions, start_dim=1))

        pred_roi_loc = self.fc_loc(y).view(num_regions, -1, 4)
        pred_roi_cls = self.fc_cls(y)

        return pred_roi_cls, pred_roi_loc


class RegionProposalNetwork(nn.Module):
    def __init__(self, num_anchors, in_channels, mid_channels=512):
        super().__init__()
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.conv_obj = nn.Conv2d(mid_channels, num_anchors * 2, 1, 1, 0)
        self.conv_loc = nn.Conv2d(mid_channels, num_anchors * 4, 1, 1, 0)

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        self.conv_obj.weight.data.normal_(0, 0.01)
        self.conv_obj.bias.data.zero_()
        self.conv_loc.weight.data.normal_(0, 0.01)
        self.conv_loc.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # show_feat_map(x, 'rpn feats')
        # show_feat_map(self.conv_loc(x), 'conv_loc')
        # show_feat_map(self.conv_obj(x), 'conv_obj')
        pred_loc = self.conv_loc(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)
        pred_obj = self.conv_obj(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 2)
        return pred_loc, pred_obj


class PreprocessHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_size = 16
        self.nms_threshold = 0.7

    def __call__(self, img_shape, anchor_boxes, pred_loc, pred_obj):
        img_height, img_width = img_shape
        # currently batch size 1 supported only
        assert anchor_boxes.shape[0] == 1
        anchor_boxes = anchor_boxes[0]
        batch_size = pred_loc.shape[0]

        pred_loc = pred_loc.detach()
        pred_obj = pred_obj.detach()
        pred_obj_sm = torch.softmax(pred_obj, dim=2)[:, :, 1]

        pred_boxes = get_boxes_from_loc_batch(anchor_boxes, pred_loc, img_width, img_height)

        pred_boxes_post_nms = []
        for batch_idx in range(batch_size):
            batch_pred_boxes = pred_boxes[batch_idx, :, :]
            batch_pred_obj = pred_obj_sm[batch_idx, :]

            valid_boxes = (batch_pred_boxes[:, 2:] - batch_pred_boxes[:, :2] >= self.min_size).all(dim=1)
            batch_pred_boxes = batch_pred_boxes[valid_boxes, :]
            batch_pred_obj = batch_pred_obj[valid_boxes]

            pred_indices = torch.flip(torch.argsort(batch_pred_obj), (0,))
            if self.training:
                pred_indices = pred_indices[:self.n_train_pre_nms]
                n_post_nms = self.n_train_post_nms
            else:
                pred_indices = pred_indices[:self.n_test_pre_nms]
                n_post_nms = self.n_test_post_nms
            batch_pred_boxes = batch_pred_boxes[pred_indices, :]
            batch_pred_obj = batch_pred_obj[pred_indices]

            post_nms_indices = ops.nms(batch_pred_boxes, batch_pred_obj, self.nms_threshold)[:n_post_nms]
            pred_boxes_post_nms.append(batch_pred_boxes[post_nms_indices])

        pred_boxes = torch.cat(pred_boxes_post_nms, axis=0)

        pred_batch_idx = np.repeat(np.arange(batch_size), [b.shape[0] for b in pred_boxes_post_nms])
        pred_boxes = pred_boxes.cpu().numpy()

        return pred_boxes, pred_batch_idx


class TrainingProposalSelector:
    def __init__(self, num_samples=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_low=0.0, loc_mean=(0.0, 0.0, 0.0, 0.0), loc_std=(0.1, 0.1, 0.2, 0.2)):
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_low = neg_iou_thresh_low
        self.loc_mean = np.array(loc_mean, dtype=np.float32).reshape(1, 4)
        self.loc_std = np.array(loc_std, dtype=np.float32).reshape(1, 4)

    def __call__(self, pred_boxes, pred_batch_idx, gt_boxes, gt_class_labels, gt_count):
        gt_boxes = gt_boxes.cpu().detach().numpy()
        gt_class_labels = gt_class_labels.cpu().detach().numpy()
        gt_count = gt_count.cpu().detach().numpy()

        final_pred_boxes = list()
        final_pred_batch_idx = list()
        final_class_labels = list()
        final_loc_labels = list()
        num_batches = gt_boxes.shape[0]
        for batch in range(num_batches):
            batch_pred_boxes = pred_boxes[pred_batch_idx == batch, :]
            batch_gt_boxes = gt_boxes[batch, :gt_count[batch], :]
            batch_gt_class_labels = gt_class_labels[batch, :gt_count[batch]]

            # include ground truth boxes
            batch_pred_boxes = np.concatenate((batch_pred_boxes, batch_gt_boxes), axis=0)

            if gt_count[batch] > 0:
                iou = compute_iou(batch_pred_boxes, batch_gt_boxes)
                max_iou_gt_idx = np.argmax(iou, axis=1)
                max_iou = iou[range(iou.shape[0]), max_iou_gt_idx]

                pred_positive_idx = np.nonzero(max_iou > self.pos_iou_thresh)[0]
                # pred_positive_idx = np.array([7, 9, 10, 21, 37, 51, 100, 161, 184, 208, 212,
                #        268, 277, 311, 315, 443, 464, 636, 820, 873, 973, 1108,
                #        1218, 1319, 1340, 1562, 1581, 1698, 1699, 1910, 1912, 2000, 2001,
                #        2002])
                gt_positive_idx = max_iou_gt_idx[pred_positive_idx]

                pred_negative_idx = np.nonzero((max_iou < self.neg_iou_thresh_hi) &
                                               (max_iou >= self.neg_iou_thresh_low))[0]
                # pred_negative_idx = np.load('/home/gleason/temp.npy', allow_pickle=True)
            else:
                pred_positive_idx = np.zeros((0,), dtype=np.int32)
                gt_positive_idx = np.zeros((0,), dtype=np.int32)
                pred_negative_idx = np.arange(batch_pred_boxes.shape[0])

            positive_choice, negative_choice = choose_anchor_subset(
                self.num_samples, self.pos_ratio, len(pred_positive_idx), len(pred_negative_idx))

            pred_positive_idx = pred_positive_idx[positive_choice]
            gt_positive_idx = gt_positive_idx[positive_choice]
            pred_negative_idx = pred_negative_idx[negative_choice]

            gt_positive_boxes = batch_gt_boxes[gt_positive_idx, :]
            pred_positive_boxes = batch_pred_boxes[pred_positive_idx, :]

            pred_positive_class_labels = batch_gt_class_labels[gt_positive_idx]
            pred_positive_loc_labels = get_loc_labels(pred_positive_boxes, gt_positive_boxes,
                                                      self.loc_mean, self.loc_std)

            num_positive = len(pred_positive_idx)
            num_negative = len(pred_negative_idx)
            num_total = num_positive + num_negative

            pred_idx = np.concatenate((pred_positive_idx, pred_negative_idx))

            batch_pred_boxes = batch_pred_boxes[pred_idx, :]
            batch_pred_batch_idx = np.full(num_total, batch, dtype=np.int32)
            batch_class_labels = np.concatenate((
                pred_positive_class_labels, np.zeros((num_negative,), dtype=np.int32)))
            batch_loc_labels = np.concatenate((
                pred_positive_loc_labels, np.zeros((num_negative, 4), dtype=np.float32)))

            final_pred_boxes.append(batch_pred_boxes)
            final_pred_batch_idx.append(batch_pred_batch_idx)
            final_class_labels.append(batch_class_labels)
            final_loc_labels.append(batch_loc_labels)

        final_pred_boxes = np.concatenate(final_pred_boxes, axis=0)
        final_class_labels = np.concatenate(final_class_labels, axis=0)
        final_loc_labels = np.concatenate(final_loc_labels, axis=0)
        final_pred_batch_idx = np.concatenate(final_pred_batch_idx, axis=0)

        return final_pred_boxes, final_pred_batch_idx, final_class_labels, final_loc_labels


class FasterRCNN(nn.Module):
    def __init__(self, num_anchors=9, num_classes=92, return_rpn_output=False, arch='vgg16'):
        super().__init__()
        self.return_rpn_output = return_rpn_output

        self.feature_net = FeatureNetVGG16() if arch == 'vgg16' else FeatureNetResNet(arch)
        self.region_proposal_network = RegionProposalNetwork(num_anchors,
                                                             self.feature_net.get_out_channels())
        self.preprocess_head = PreprocessHead()
        self.head = HeadVGG16(num_classes) if arch == 'vgg16' else HeadResNet(num_classes, arch)
        self.training_proposal_selector = TrainingProposalSelector()

    def forward(self, x, anchor_boxes, initial_batch_idx, gt_boxes=None, gt_class_labels=None, gt_count=None):
        if self.training:
            assert gt_boxes is not None and gt_count is not None and gt_class_labels is not None

        # fig = plt.figure(figsize=(10, 10), dpi=80)
        # ax = fig.subplots()
        # ax.imshow(x[0, ...].cpu().numpy().transpose(1, 2, 0))
        # fig.show()

        input_shape = x.shape[-2:]
        x = self.feature_net(x)

        # show_feat_map(x)

        pred_loc, pred_obj = self.region_proposal_network(x)
        with torch.no_grad():
            pred_roi_boxes, pred_roi_batch_idx = self.preprocess_head(input_shape, anchor_boxes, pred_loc, pred_obj)
            if self.training:
                pred_roi_boxes, pred_roi_batch_idx, pred_roi_cls_labels, pred_roi_loc_labels \
                    = self.training_proposal_selector(pred_roi_boxes, pred_roi_batch_idx,
                                                      gt_boxes, gt_class_labels, gt_count)

                pred_roi_cls_labels = torch.from_numpy(pred_roi_cls_labels).to(dtype=torch.long, device=x.device)
                pred_roi_loc_labels = torch.from_numpy(pred_roi_loc_labels).to(dtype=torch.float32, device=x.device)

        pred_roi_cls, pred_roi_loc = self.head(x, pred_roi_boxes, pred_roi_batch_idx)

        pred_roi_boxes = torch.from_numpy(pred_roi_boxes).to(device=x.device)
        pred_roi_batch_idx = torch.from_numpy(pred_roi_batch_idx).to(device=x.device, dtype=torch.long)

        # local batch indicies may differ from original with DataParallel, need to map them
        pred_roi_batch_idx = initial_batch_idx[pred_roi_batch_idx]

        output = {
            'pred_roi_batch_idx': pred_roi_batch_idx,
            'pred_roi_boxes': pred_roi_boxes,
            'pred_roi_cls': pred_roi_cls,
            'pred_roi_loc': pred_roi_loc}
        if self.training or self.return_rpn_output:
            output['pred_loc'] = pred_loc
            output['pred_obj'] = pred_obj
        if self.training:
            output['pred_roi_cls_labels'] = pred_roi_cls_labels
            output['pred_roi_loc_labels'] = pred_roi_loc_labels

        return output
