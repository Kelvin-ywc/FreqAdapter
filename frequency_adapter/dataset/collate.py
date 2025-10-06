import torch
from PIL import Image

def collate_fn_one_image_multiple_captions(batch):
    image_paths, caption_list = zip(*batch)
    images = [Image.open(image_path) for image_path in image_paths]
    all_captions = [caption['caption'] for captions in caption_list for caption in captions]
    caption_len = [len(captions) for captions in caption_list]
    return images, all_captions, caption_len
