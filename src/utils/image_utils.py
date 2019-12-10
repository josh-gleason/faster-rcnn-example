from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import randomcolor

try:
    _font = ImageFont.load("FreeMono.ttf")
except IOError:
    try:
        _font = ImageFont.load("arial.ttf")
    except IOError:
        _font = ImageFont.load_default()


def draw_detections(img, rects, texts=None, scale=None):
    """ Return a copy of img with detections drawn """

    def _draw_rectangle(img, rect, color="#ff0000ff"):
        drawer = ImageDraw.Draw(img, 'RGBA')
        drawer.rectangle(rect, outline=color, width=2)

    def _draw_text(img, coord_xy, text, color="#ff0000ff"):
        x, y = coord_xy
        drawer = ImageDraw.Draw(img, 'RGBA')
        # draw transparent background behind text to ensure readability
        w, h = _font.getsize(text)
        drawer.rectangle((x, y, x + w, y + h), fill="#0000007f")
        drawer.text(coord_xy, text, fill=color, font=_font)

    if scale is not None:
        new_width = int(round(img.width * scale))
        new_height = int(round(img.height * scale))
        img = img.resize((new_width, new_height), resample=Image.BILINEAR)
        rects = [[r * scale for r in rect] for rect in rects]
    else:
        # copy to avoid in place augmentation of img
        img = img.copy()

    if len(rects) == 0:
        return img

    color_generator = randomcolor.RandomColor()
    colors = color_generator.generate(count=len(rects), luminosity="light")
    for rect, color in zip(rects, colors):
        _draw_rectangle(img, rect, color)

    # draw text on top of rectangles
    if texts is not None:
        for rect, text, color in zip(rects, texts, colors):
            _draw_text(img, rect[:2], text, color)

    return img


def test_draw_detections(writer):
    print("Drawing some detections to tensorboard")
    from utils.data_mappings import coco_id_to_name
    from torchvision import datasets
    from torchvision import transforms

    def convert_bbox_labels(labels):
        """ convert bounding boxes from left-upper-width-height format to
            left-upper-right-down. """
        for label in labels:
            x, y, w, h = labels['bbox']
            label['bbox'] = [x, y, x + w, y + h]
        return labels

    dataset = datasets.CocoDetection(
        '../coco/images/val2017',
        '../coco/annotations/instances_val2017.json',
        transform=None,
        target_transform=convert_bbox_labels
    )

    img, labels = dataset[0]

    label_bboxs = [label['bbox'] for label in labels]
    label_names = [coco_id_to_name[label['category_id']] for label in labels]

    img = draw_detections(img, label_bboxs, label_names, scale=2.0)

    writer.add_image("test_draw_detections", transforms.functional.to_tensor(img))


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('../tensorboard')
    try:
        test_draw_detections(writer)
    finally:
        writer.close()
