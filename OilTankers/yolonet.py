"""
    Author: julij
    Date: 30/12/2022
    Description: 
"""

import torch
import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection


class YoloNet(pl.LightningModule):

    def __init__(self, lr, weight_decay, train_dataloader, valid_dataloader):
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained('hustvl/yolos-tiny',
                                                                 num_labels=1,
                                                                 ignore_mismatched_sizes=True)
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.train_dataloader = train_dataloader
        # self.valid_dataloader = valid_dataloader
        # self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    # def common_step(self, batch, batch_idx):
    #     pixel_values = batch['pixel_values']
    #     labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
    #     outputs = self.model(pixel_values=pixel_values, labels=labels)
    #     loss = outputs.loss
    #     loss_dict = outputs.loss_dict
    #     return loss, loss_dict
    #
    # def training_step(self, batch, batch_idx):
    #     loss, loss_dict = self.common_step(batch, batch_idx)
    #     self.log("train_loss", loss)
    #     for k, v in loss_dict.items():
    #         self.log('train_' + k, v.item())
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     loss, loss_dict = self.common_step(batch, batch_idx)
    #     self.log("validation_loss", loss)
    #     for k, v in loss_dict.items():
    #         self.log("validation_" + k, v.item())
    #     return loss
    #
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer
    #
    # def train_dataloader(self):
    #     return self.train_dataloader
    #
    # def val_dataloader(self):
    #     return self.valid_dataloader