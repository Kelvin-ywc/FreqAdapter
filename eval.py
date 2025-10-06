import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from frequency_adapter.dataset import COCOClassification, COCOCaptionVal, collate_fn_one_image_multiple_captions, Flickr30KCaptionVal
from frequency_adapter.eval import eval_fn, base_eval_fn
from frequency_adapter.utils import compute_retrieval_acc

from loguru import logger
from tqdm import tqdm
from datetime import datetime

import argparse
import yaml

import wandb
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def prepare_model(model_name_or_path, peft_type, ckpt_path, device, fp16):
    if peft_type is None:
        # load base model and return
        model = CLIPModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16 if fp16 else torch.float32).to(device)
        return model
    else:
        # TODO
        return None
    
def prepare_dataset(processor):
    # load data
    logger.info('loading dataset')
    train_image_path = '/home/DATA/coco/train2017'
    train_ann_path = '/home/DATA/coco/annotations/captions_train2017.json'
    val_image_path = '/home/DATA/coco/val2017'
    val_ann_path = '/home/DATA/coco/annotations/captions_val2017.json'

    train_batch_size = 128
    val_batch_size = 128
    train_dataset = COCOCaptionTrain(
        image_path=train_image_path,
        ann_path=train_ann_path,
        transform=processor.feature_extractor)
    valid_dataset = COCOCaptionVal(
        image_path=val_image_path,
        ann_path=val_ann_path,
        transform=processor.feature_extractor)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False)

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_one_image_multiple_captions,
        drop_last=False)

    
    del train_dataset, valid_dataset

    image_path = '/home/DATA/flickr30k/flickr30k-images'
    ann_path = '/home/DATA/flickr30k/results_20130124.token'
    split_val_path = ['/home/DATA/flickr30k/val.txt']
    split_test_path = ['/home/DATA/flickr30k/test.txt']
    transform = None
    flickr30k_val_dataset = Flickr30KCaptionVal(image_path=image_path, ann_path=ann_path, split_paths=split_val_path, transform=transform)
    flickr30k_test_dataset = Flickr30KCaptionVal(image_path=image_path, ann_path=ann_path, split_paths=split_test_path, transform=transform)

    flickr30k_val_dataloader = torch.utils.data.DataLoader(
        dataset=flickr30k_val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_one_image_multiple_captions,
        drop_last=False)
    
    flickr30k_test_dataloader = torch.utils.data.DataLoader(
        dataset=flickr30k_test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_one_image_multiple_captions,
        drop_last=False)
    
    return train_dataloader, valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader


def main(args):

    model_name_or_path = args.model_name_or_path
    peft_type = args.peft_type
    ckpt_path = args.ckpt_path
    dataset_name = args.dataset_name

    # for log
    task_type = 'retrieval'
    log_dir = './OUTPUT'
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name_for_log = model_name_or_path.replace('/', '-')
    logger.add(f'{log_dir}/eval/{model_name_for_log}_{dataset_name}_{peft_type}_{task_type}_{time}.log', rotation='10 MB')
    logger.info('loading model')

    # settings
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    fp16 = True
        
    # load model
    model = prepare_model(model_name_or_path, peft_type, ckpt_path, device, fp16)
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    logger.info(model)
    logger.info(processor)

    # load dataset
    valid_dataloader, test_dataloader = prepare_dataset(dataset_name, processor)

    # test
    if peft_type in [None]:
        if valid_dataloader is not None:
            logger.info(f'Begin Evaluation {dataset_name} val')
            base_eval_fn(model, processor, valid_dataloader, device)
        if test_dataloader is not None:
            logger.info(f'Begin Evaluation {dataset_name} test')
            base_eval_fn(model, processor, test_dataloader, device)
    else:
        # cross modality interaction
        topk_list = [1, 5, 10]
        for topk in topk_list:
            if valid_dataloader is not None:
                logger.info(f'Begin Evaluation {dataset_name} val topk={topk}')
                eval_fn(model, processor, valid_dataloader, device, peft_type, topk=topk)
            if test_dataloader is not None:
                logger.info(f'Begin Evaluation {dataset_name} test topk={topk}')
                eval_fn(model, processor, test_dataloader, device, peft_type, topk=topk)


def eval_model(
        model, processor, 
        valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader, 
        device, peft_type
    ):
    if peft_type in ['freq_adapter', 'org_freq_adapter']:
        logger.info('Begin evaluating coco val')
        eval_fn(model, processor, valid_dataloader, device, topk=5)
        logger.info('Begin evaluating flickr30k val')
        eval_fn(model, processor, flickr30k_val_dataloader, device, topk=5)
        logger.info('Begin evaluating flickr30k test')
        eval_fn(model, processor, flickr30k_test_dataloader, device, topk=5)
    elif peft_type in ['clip', 'clip_adapter']:
        logger.info('Begin evaluating coco val')
        base_eval_fn(model, processor, valid_dataloader, device)
        logger.info('Begin evaluating flickr30k val')
        base_eval_fn(model, processor, flickr30k_val_dataloader, device)
        logger.info('Begin evaluating flickr30k test')
        base_eval_fn(model, processor, flickr30k_test_dataloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument('--peft_type', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='coco2017', choices=['coco2017', 'flickr30k'])

    args, unknown = parser.parse_known_args()
    if False:
        try:
            import debugpy
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            print("Debugger attached")
        except:
            pass
    main(args)