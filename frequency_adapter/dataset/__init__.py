from .COCODataset import COCOClassification, COCOCaptionTrain, COCOCaptionVal, Flickr30KCaptionVal
from .collate import collate_fn_one_image_multiple_captions
from torchvision.datasets import CIFAR10, CIFAR100
import os




def build_dataset(dataset_name, data_type):
    all = {
        'cifar10': {
            'train': CIFAR10(root=os.path.expanduser("/root/autodl-tmp/DATA"), download=True, train=True),
            'test': CIFAR10(root=os.path.expanduser("/root/autodl-tmp/DATA"), download=True, train=False)
        },
        'cifar100': {
            'train': CIFAR100(root=os.path.expanduser("/root/autodl-tmp/DATA"), download=True, train=True),
            'test': CIFAR100(root=os.path.expanduser("/root/autodl-tmp/DATA"), download=True, train=False)
        },
        'coco2017_retrieval': {
            'train': COCOCaptionTrain(
                image_path='/root/autodl-tmp/DATA/coco/COCO2017/train2017',
                ann_path='/root/autodl-tmp/DATA/coco/COCO2017/annotations/captions_train2017.json',
                transform=processor.feature_extractor),
            'test': COCOCaptionVal(
                image_path='/root/autodl-tmp/DATA/coco/COCO2017/val2017',
                ann_path='/root/autodl-tmp/DATA/coco/COCO2017/annotations/captions_val2017.json',
                transform=processor.feature_extractor)
        },
        'coco2017_classification': {
            'train': COCOClassification(
                image_path='/root/autodl-tmp/DATA/coco/COCO2017/train2017',
                ann_path='/root/autodl-tmp/DATA/coco/COCO2017/annotations/instances_train2017.json'),
            'test': COCOClassification(
                image_path='/root/autodl-tmp/DATA/coco/COCO2017/val2017',
                ann_path='/root/autodl-tmp/DATA/coco/COCO2017/annotations/instances_val2017.json')
        }
    }
    return all[dataset_name][data_type]