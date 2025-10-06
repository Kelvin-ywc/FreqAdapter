from tqdm import tqdm
import torch
from ..utils import compute_retrieval_acc
from ..models.FrequencyAdaptationCLIP import FrequencyAdaptationCLIP
from loguru import logger
from datetime import datetime
import wandb

'''
Two evaluation functions:
1. eval_fn: with cross modality interaction, i.e., using top-k text features to adapt image features
should rewrite get_all_text_features and get_all_image_features in model to return the last second hidden states and features after projection
and get_adapted_image_features to adapt image features

2. base_eval_fn: no cross modality interaction, i.e., directly compute similarity between image and text features
should rewrite get_text_features and get_image_features in model to return all hidden states and features after projection
'''
# no cross modality interaction
def base_eval_fn(model, processor, valid_dataloader, device):
    logger.info('Begin Evaluation')

    # get all text features
    image_feature_list = []
    text_feature_list = []
    caption_lens = []
    model.eval()
    logit_scale = model.logit_scale.exp().cpu()
    logger.info('Extracting text features')

    for images, captions, caption_len in tqdm(valid_dataloader): # image_paths and its captions

        text_inputs = processor(text=captions, return_tensors='pt', padding=True, truncation=True).to(device)
        caption_lens.extend(caption_len)
        with torch.no_grad():
            text_feature = model.get_text_features(**text_inputs)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        text_feature_list.append(text_feature)
    logger.info(f'caption length: {len(caption_lens)}')

    text_features = torch.cat(text_feature_list, dim=0)
    logger.info('Extracting image features')
    for images, captions,caption_len in tqdm(valid_dataloader): # image_paths and its captions
        vision_inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
        # text_inputs = processor(text=captions, return_tensors='pt', padding=True).to(device)
        # caption_lens.extend(caption_len)
        with torch.no_grad():
            image_feature = model.get_image_features(**vision_inputs)
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
            image_feature_list.append(image_feature)
            # text_features.append(text_feature)
    image_features = torch.cat(image_feature_list, dim=0)
    logger.info(f'image_features shape: {image_features.shape}')
    logger.info(f'text_features shape: {text_features.shape}')

    # construct retrieval labels
    labels_i2t = {}
    labels_t2i = {}
    start = 0
    for i, caption_len in enumerate(caption_lens):
        labels_i2t[i] = list(range(start, start+caption_len))
        for j in range(caption_len):
            labels_t2i[start+j] = [i]
        start += caption_len

    logger.info(f'labels length: {len(labels_i2t)}')

    logits_per_text = torch.matmul(text_features, image_features.t())*logit_scale
    logits_per_image = logits_per_text.t()

    logits_per_text = logits_per_text.detach().cpu().numpy()
    logits_per_image = logits_per_image.detach().cpu().numpy()
    # compute loss using logits
    loss_label = torch.zeros(logits_per_text.shape[0], dtype=torch.long)
    for i, label in labels_t2i.items():
        loss_label[i] = labels_t2i[i][0]
    print(loss_label)

    val_loss = torch.nn.CrossEntropyLoss()(torch.from_numpy(logits_per_text), loss_label)
    logger.info(f'Val Loss: {val_loss}')

    acc_1_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 1)
    acc_5_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 5)
    acc_10_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 10)
    
    logger.info(f'image2text: acc@1: {acc_1_i2t}, acc@5: {acc_5_i2t}, acc@10: {acc_10_i2t}')
    acc_1_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 1)
    acc_5_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 5)
    acc_10_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 10)
    logger.info(f'text2image: acc@1: {acc_1_t2i}, acc@5: {acc_5_t2i}, acc@10: {acc_10_t2i}')


def eval_fn(model, processor, valid_dataloader, device, topk=20):
    logger.info('Begin Evaluation')
    logger.info(f'Select top k related text features: {topk}')
    image_feature_list = []
    text_feature_list = [] # after projection
    text_outputs_list = [] # all hidden states
    attention_mask_list = []
    caption_lens = []

    model.eval()
    # model.logit_scale.requires_grad = False
    logit_scale = model.logit_scale.exp().cpu()
    # get all text features
    logger.info('Extracting text features')
    debug = True
    for images, captions, caption_len in tqdm(valid_dataloader): # image_paths and its captions
        text_inputs = processor(text=captions, return_tensors='pt', padding='max_length', max_length=77, truncation=True).to(device)
        caption_lens.extend(caption_len)
        with torch.no_grad():
            text_outputs, text_feature = model.get_all_text_features(**text_inputs) # [倒数第二层hidden states, features after projection]
        text_outputs_list.append(text_outputs)
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
        text_feature_list.append(text_feature)
        attention_mask_list.append(text_inputs['attention_mask'])
    logger.info(f'caption length: {len(caption_lens)}')

    text_features = torch.cat(text_feature_list, dim=0)
    text_outputs = torch.cat(text_outputs_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    # topk = 20
    logger.info(f'top k: {topk}')
    logger.info('Extracting image features')
    for images, captions,caption_len in tqdm(valid_dataloader): # image_paths and its captions
        vision_inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            org_image_output, org_image_feature = model.get_all_image_features(**vision_inputs)
            org_image_feature = org_image_feature / org_image_feature.norm(p=2, dim=-1, keepdim=True)
            # org_sim = (logit_scale * org_image_feature @ text_features.T).softmax(dim=-1)
            org_sim = logit_scale * org_image_feature @ text_features.T
            values, indices = org_sim.topk(topk)
            # values softmax
            scores = torch.nn.functional.softmax(values, dim=-1) # [batch_size, topk]
            # print(torch.mean(text_outputs[indices], dim=1))
            text_feature_to_fuse = text_outputs[indices] # [128, 5, 77, 768]
            attention_mask_to_fuse = attention_mask[indices] # [128, 5, 77]
            text_feature_to_fuse = text_feature_to_fuse * attention_mask_to_fuse.unsqueeze(-1) # [128, 5, 77, 768]
            text_feature_to_fuse = torch.sum(text_feature_to_fuse, dim=2) / torch.sum(attention_mask_to_fuse, dim=2, keepdim=True) # [128, 5, 768]
            
            # text_feature_to_fuse = torch.mean(text_feature_to_fuse, dim=1) # [128, 768]
            text_feature_to_fuse = torch.sum(text_feature_to_fuse * scores.unsqueeze(-1), dim=1) # [128, 768]

            # print(f'text_feature_to_fuse shape after sum: {text_feature_to_fuse.shape}')

            image_feature = model.get_adapted_image_features(org_image_output, text_feature_to_fuse)
            # print(f'image_feature shape: {image_feature.shape}')
            # print(image_feature)
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)

            image_feature_list.append(image_feature)
            # text_features.append(text_feature)
    image_features = torch.cat(image_feature_list, dim=0)
    logger.info(f'image_features shape: {image_features.shape}')
    logger.info(f'text_features shape: {text_features.shape}')
    # construct retrieval labels
    labels_i2t = {}
    labels_t2i = {}
    start = 0
    for i, caption_len in enumerate(caption_lens):
        labels_i2t[i] = list(range(start, start+caption_len))
        for j in range(caption_len):
            labels_t2i[start+j] = [i]
        start += caption_len

    logger.info(f'labels length: {len(labels_i2t)}')

    logits_per_text = torch.matmul(text_features, image_features.t())*logit_scale
    logits_per_image = logits_per_text.t()

    logits_per_text = logits_per_text.detach().cpu().numpy()
    logits_per_image = logits_per_image.detach().cpu().numpy()
    # compute loss using logits
    loss_label = torch.zeros(logits_per_text.shape[0], dtype=torch.long)
    for i, label in labels_t2i.items():
        loss_label[i] = labels_t2i[i][0]
    print(loss_label)
    val_loss = torch.nn.CrossEntropyLoss()(torch.from_numpy(logits_per_text), loss_label)
    logger.info(f'Val Loss: {val_loss}')
    acc_1_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 1)
    acc_5_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 5)
    acc_10_i2t = compute_retrieval_acc(labels_i2t, logits_per_image, 10)
    
    logger.info(f'image2text: acc@1: {acc_1_i2t}, acc@5: {acc_5_i2t}, acc@10: {acc_10_i2t}')
    acc_1_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 1)
    acc_5_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 5)
    acc_10_t2i = compute_retrieval_acc(labels_t2i, logits_per_text, 10)
    logger.info(f'text2image: acc@1: {acc_1_t2i}, acc@5: {acc_5_t2i}, acc@10: {acc_10_t2i}')

def extract_text_inputs(processor, text_inputs, device):
    # text_inputs: caption list
    text_inputs = processor(text=text_inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    return text_inputs

def extract_text_features(model, processor, text_inputs, device):
    text_inputs = extract_text_inputs(processor, text_inputs, device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

