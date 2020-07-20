import os
import tempfile
import zipfile
from collections import namedtuple
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import datasets
from urllib.parse import urlparse
from torch.hub import download_url_to_file

urls = {
    'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip'
}

output_dir = {
    'annotations_trainval2017.zip': 'annotations',
    'train2017.zip': 'images/train2017',
    'val2017.zip': 'images/val2017'
}

image_set_files = {
    'train': ('annotations_trainval2017.zip', 'train2017.zip'),
    'val': ('annotations_trainval2017.zip', 'val2017.zip')
}

_CocoTargetType = namedtuple('_CocoTargetType', 'annotations image_id')


def unzip_file_from_url(output_dir, url):
    output_dir = os.path.realpath(output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        temp_path = os.path.join(temp_dir, filename)
        if not os.path.exists(temp_path):
            print('Downloading: "{}" to {}\n'.format(url, temp_path))
            download_url_to_file(url, temp_path)

        assert zipfile.is_zipfile(temp_path)

        with zipfile.ZipFile(temp_path) as temp_zipfile:
            for member in tqdm(temp_zipfile.infolist(), desc='Extracting ', ncols=0):
                temp_zipfile.extract(member, output_dir)


class CocoDetectionWithImgId(datasets.CocoDetection):
    def __init__(self, root, image_set='train', download=False, transform=None, target_transform=None, transforms=None):
        if download:
            for required_file in image_set_files[image_set]:
                if not os.path.exists(os.path.join(root, output_dir[required_file])):
                    result_dir = os.path.dirname(os.path.realpath(os.path.join(root, output_dir[required_file])))
                    unzip_file_from_url(result_dir, urls[required_file])
        annFile = os.path.join(root, f'annotations/instances_{image_set}2017.json')
        imgDir = os.path.join(root, f'images/{image_set}2017')
        super().__init__(imgDir, annFile, transform, target_transform, transforms)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        labels = _CocoTargetType(coco.loadAnns(ann_ids), img_id)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, labels = self.transforms(img, labels)

        return img, labels


class FormatCOCOLabels(object):
    def __init__(self):
        pass

    def __call__(self, data, labels):
        annotations = labels.annotations
        # matches annotations[0]['image_id'] if annotations is not empty but need image_id even if annotations is empty
        image_id = labels.image_id

        if len(annotations) > 0:
            gt_class_labels = np.array([label['category_id'] for label in annotations], dtype=np.int64)
            gt_boxes = np.array([label['bbox'] for label in annotations], dtype=np.float32)
            gt_boxes[:, 2:] += gt_boxes[:, 0:2]

            ignore = np.array([('ignore' in label and label['ignore']) or ('iscrowd' in label and label['iscrowd'])
                               for label in annotations], dtype=np.bool)
        else:
            gt_class_labels = np.zeros((0,), dtype=np.int64)
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            ignore = np.zeros((0,), dtype=np.bool)

        # difficult not used for coco, set to all false
        difficult = np.zeros(gt_class_labels.shape, np.bool)

        labels_dict = {
            'gt_boxes': gt_boxes,
            'gt_class_labels': gt_class_labels,
            'difficult': difficult,
            'ignore': ignore,
            'image_id': image_id,
            'original_image': data,
        }

        return data, labels_dict

    def __repr__(self):
        return f'{self.__class__.__name__}()'
