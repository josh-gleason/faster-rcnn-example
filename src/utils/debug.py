import numpy as np
import matplotlib.pyplot as plt
from utils.image_utils import draw_detections
from train import get_display_gt_boxes


def debug_dataset_view(dataset, anchor_boxes, id_to_name_map):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 12), dpi=80)
    ax1, ax2 = fig.subplots(2)
    for x, (anchor_obj_final, anchor_loc_final, gt_boxes, gt_class_labels,
            gt_image_id, gt_difficult, ignore, resize_shape, orig_img) in dataset:
        ax1.clear()
        ax2.clear()
        # draw original image with labeled gt boxes
        orig_width, orig_height = orig_img.size
        resize_height, resize_width = x.shape[-2:]
        rect_list_gt, text_list_gt = get_display_gt_boxes(gt_boxes, gt_class_labels, resize_shape,
                                                          (orig_height, orig_width), id_to_name_map)
        gt_img = draw_detections(orig_img, rect_list_gt, text_list_gt)
        ax1.imshow(np.array(gt_img))

        # draw actual network input image
        y = x.numpy().transpose(1, 2, 0)
        z = y * np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)) + np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        z = np.clip(z, 0, 1.0)
        ax2.imshow(z)

        # draw positive anchor boxes
        from utils.box_utils import get_boxes_from_loc
        pos_indices = np.nonzero(anchor_obj_final > 0)[0]
        pos_boxes = anchor_boxes[pos_indices, :]
        pos_locs = anchor_loc_final[pos_indices, :]
        # positive proposals without localization correction are green
        for box in pos_boxes:
            x1, y1, x2, y2 = box
            ax2.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], '-g', linewidth=4)
        # positive proposals with localization correction are yellow
        pos_boxes_fixed = get_boxes_from_loc(pos_boxes, pos_locs, resize_width, resize_height)
        for box in pos_boxes_fixed:
            x1, y1, x2, y2 = box
            ax2.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], '-y', linewidth=2)
        # center of negative proposals are red (too noisy to plot as boxes)
        neg_indices = np.nonzero(anchor_obj_final == 0)[0]
        for box in anchor_boxes[neg_indices, :]:
            x1, y1, x2, y2 = box
            ax2.plot(0.5 * (x1 + x2), 0.5 * (y1 + y2), ' .r')

        fig.show()
        breakpoint()


def debug_show_feat_map(x, title='feats'):
    x_np = x[0, ...].detach().cpu().numpy()
    ch = x.shape[1]
    r = ch // 2**int(np.ceil(np.log2(ch) / 2))
    c = int(np.ceil(ch / r))
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
    im_rows, im_cols = res.shape
    fig = plt.figure(figsize=(1 + int(im_rows // 80), 1.5 + int(im_cols // 80)), dpi=80)
    ax = fig.subplots()
    ax.imshow(res)
    ax.set_title(title + ' debug')
    fig.show()
    breakpoint()