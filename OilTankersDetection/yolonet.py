"""
    Author: julij
    Date: 30/12/2022
    Description: 
"""

import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection


class YoloNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForObjectDetection.from_pretrained('hustvl/yolos-tiny',
                                                                 num_labels=1,
                                                                 ignore_mismatched_sizes=True)
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs