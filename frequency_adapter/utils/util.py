from loguru import logger

def log_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'name: {name}, param cnt: {param.numel()}')
