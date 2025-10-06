from torch import nn
from transformers import Trainer

class FrequencyAdaptationCLIPTrainer(Trainer):
    def get_train_dataloader(self):
        return self.train_dataloader