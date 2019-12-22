import torch
import torch.nn as nn
import torch.nn.functional as F


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



