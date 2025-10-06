from numpy import dtype
from sympy import false
from torch.amp import GradScaler, autocast

from frequency_adapter.dataset import COCOCaptionTrain, COCOCaptionVal, collate_fn_one_image_multiple_captions, Flickr30KCaptionVal
from frequency_adapter.eval import eval_fn, base_eval_fn
from frequency_adapter.utils import log_trainable_parameters

import torch
from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image

import argparse
import yaml

from datetime import datetime
import os
import wandb
import random
import numpy as np

from frequency_adapter.models import FreqApdapterModel, CLIPAdapterCLIPModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed_value: int):
    """固定 Python / NumPy / PyTorch 的随机种子"""
    random.seed(seed_value)  # Python 内置随机
    np.random.seed(seed_value)  # NumPy 随机
    torch.manual_seed(seed_value)  # PyTorch CPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)            # 当前 GPU 的随机种子
        torch.cuda.manual_seed_all(seed_value)        # 所有 GPU 的随机种子（多卡情况）
    
    # 一些与 PyTorch 的计算确定性相关的配置
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    
    print(f"Random seed set to {seed_value}")

def prepare_model(model_name_or_path, peft_type, fp16, device):
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    if peft_type == 'freq_adapter':
        model = FreqApdapterModel.from_pretrained(model_name_or_path).to(device)
        return model, processor
    elif peft_type == 'clip_adapter':
        model = CLIPAdapterCLIPModel.from_pretrained(model_name_or_path).to(device)
        return model, processor
    elif peft_type == 'clip':
        model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        return model, processor
    elif peft_type == 'org_freq_adapter':
        model = FrequencyAdaptationCLIP.from_pretrained(model_name_or_path).to(device)
        return model, processor
    else:
        raise ValueError(f'peft_type {peft_type} not supported')

def set_model(model, peft_type):
    # model.init_weights()
    if peft_type == 'freq_adapter':
        logger.info('Using Frequency Adapter')
        model.freq_adapter.init_weights()
        # model.freq_adapter.mgfa.init_weights()
    elif peft_type == 'clip_adapter':
        logger.info('Using CLIP Adapter')
        model.adapter.init_weights()

    for name, param in model.named_parameters():
        if 'adapter' in name or 'freq' in name:
            param.requires_grad = True
            logger.info(f'param: {name} not frozen, number of parameters: {param.numel()}, param shape: {param.shape}')
        else:
            param.requires_grad = False
            logger.info(f'param: {name} frozen')
    
    total_params = sum(p.numel() for p in model.parameters())  # numel() 返回参数的元素总数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 只有requires_grad=True的参数才是可训练的

    # 打印结果
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info('{}% of the parameters are trainable', trainable_params / total_params * 100)
    

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

def main(args):
    set_seed(42)

    model_name_or_path = args.model_name_or_path
    peft_type = args.peft_type

    # args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for logger
    dataset_name = 'coco2017'
    task_type = 'retrieval'
    log_dir = './OUTPUT'
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name_for_log = model_name_or_path.replace('/', '-')
    fp16 = True
    logger.add(f'{log_dir}/train/{model_name_for_log}_{dataset_name}_{peft_type}_{task_type}_{time}.log', rotation='10 MB')
    
    # load model
    model, processor = prepare_model(model_name_or_path, peft_type, fp16, device)
    logger.info(f'model: {model_name_or_path} loaded')
    logger.info(model)

    set_model(model, peft_type)

    # load dataset
    train_dataloader, valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader = prepare_dataset(processor)

    # train and eval model
    # prepare training params
    epochs = args.epochs
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    scaler = GradScaler()
    model.train()


    model.set_adapter(args.enable_mcfa, args.enable_mgfa, args.use_freq_adapter)
    # eval_model(model, processor, valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader, device, peft_type)

    for epoch in range(epochs):
        logger.info(f'epoch: {epoch}')
        
        running_loss = 0.0

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")

        for idx, (image_paths, captions) in pbar:
            model.train()
            
            optimizer.zero_grad()
            
            images = [Image.open(image_path) for image_path in image_paths]
            inputs = processor(
                images=images, 
                text=captions, 
                return_tensors='pt', 
                max_length=77, 
                padding='max_length', 
                truncation=True,
            ).to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16 if fp16 else torch.float32):
                outputs = model(**inputs, return_loss=True)
                loss = outputs['loss']
            
            running_loss += loss.item()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr']) # 直接从优化器获取当前lr

            # --- 完整的 GradScaler 更新流程 ---
            # 1. 放大 loss，并计算梯度
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 2. 更新模型权重
            scaler.step(optimizer)
            
            # 3. 更新 scaler 的缩放因子
            scaler.update()
            
            if (idx + 1) % 300 == 0:
                # 计算过去300步的平均损失
                avg_loss = running_loss / 300
                logger.info(f'epoch: {epoch}, idx:{idx+1}, avg_loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]}')
                
                # 重置累加器，为下一个300步做准备
                running_loss = 0.0
                
                # eval_model(model, processor, valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader, device, peft_type)

        # ✅ 修改点3: 在每个 epoch 结束后，更新学习率
        # scheduler.step()
        
        model.save_pretrained(f'./CKPT/{peft_type}_{epoch}_{time}')
        ## eval flickr30k
        logger.info('Begin Evaluating flickr30k val')
        eval_model(model, processor, valid_dataloader, flickr30k_val_dataloader, flickr30k_test_dataloader, device, peft_type)


if __name__ == '__main__':
    if False:
        try:
            import debugpy
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            print("Debugger attached")
        except Exception as e:
            print("Debugging not enabled:", e)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--peft_type', type=str, default='freq_adapter') # options: ['freq_adapter', 'clip_adapter']
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--enable_mcfa', action='store_true', help='Enable MCFA module')
    parser.add_argument('--enable_mgfa', action='store_true', help='Enable MGFA module')
    parser.add_argument('--use_freq_adapter', action='store_true', help='Use Frequency Adapter')
    args = parser.parse_args()
    logger.info(args)
    main(args)