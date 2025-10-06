from idlelib.pyparse import trans

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from loguru import logger
import os
from PIL import Image
import pandas as pd

class COCOClassification(Dataset):
    def __init__(self, image_path, ann_path, is_train):
        logger.info('Building MSCOCO Classification Dataset.')
        self.coco = COCO(annotation_file=ann_path)


    def __getitem__(self, item):
        pass

# read one image and one caption
class COCOCaptionTrain(Dataset):
    def __init__(self, image_path, ann_path, transform):
        logger.info('Building MSCOCO Caption Dataset.')
        self.root_path = image_path
        self.coco = COCO(annotation_file=ann_path)
        self.image_ids = self.coco.getImgIds()
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())
        self.transform = transform
        logger.info(f'Building Training Dataset.')
        logger.info(f'Training Image Number: {len(self.image_ids)}')
        logger.info(f'Training Pairs Number: {len(self.annotations)}')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        image_id = self.annotations[item]['image_id']
        caption = self.annotations[item]['caption']
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.root_path, image_filename)
        # image_path = Image.open(os.path.join(self.root_path), image_id, 'jpg')
        return image_path, caption

# read one image and its corresponding captions
class COCOCaptionVal(Dataset):
    def __init__(self, image_path, ann_path, transform):
        logger.info('Building MSCOCO Caption Dataset.')
        self.root_path = image_path
        self.coco = COCO(annotation_file=ann_path)
        self.image_ids = self.coco.getImgIds()
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())
        self.transform = transform

        logger.info(f'Building Validation Dataset.')
        logger.info(f'Validation Image Number: {len(self.image_ids)}')
        logger.info(f'Validation Pairs Number: {len(self.annotations)}')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        ann_ids = self.coco.getAnnIds(image_id)
        captions = self.coco.loadAnns(ann_ids)
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.root_path, image_filename)
        # image_path = Image.open(os.path.join(self.root_path), image_id, 'jpg')
        return image_path, captions
    
class Flickr30KCaptionVal(Dataset):
    def __init__(self, image_path='/root/autodl-tmp/DATA/flickr30k/flickr30k-images', ann_path='/root/autodl-tmp/DATA/flickr30k/results_20130124.token', split_paths=['/root/autodl-tmp/DATA/flickr30k/test.txt'], transform=None):
        self.root_path = image_path
        # 打开文件并读取内容
        self.image_ids = []
        for split_path in split_paths:
            with open(split_path, 'r') as file:
                # 读取每行并转换为整数（或浮点数），然后保存到列表中
                self.image_ids.extend([int(line.strip()) for line in file])
        annotations = pd.read_table(ann_path, sep='\t', header=None, names=['image', 'caption'])
        self.ann = {}
        for index, row in annotations.iterrows():
            image_name = row['image']
            caption = row['caption']
            
            # 如果图片名不在字典中，添加一个空列表
            if image_name[:-2] not in self.ann:
                self.ann[image_name[:-2]] = []
            
            # 将 caption 添加到对应图片名的列表中
            self.ann[image_name[:-2]].append({'caption': caption})
        # print(self.ann)


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        image_name = f'{self.root_path}/{image_id}.jpg'
        captions = self.ann[f'{image_id}.jpg']
        return image_name, captions
