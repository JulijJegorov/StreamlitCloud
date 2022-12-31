"""
    Author: julij
    Date: 31/12/2022
    Description: 
"""

from torchvision.datasets import CocoDetection


class CustomDataset(CocoDetection):
    def __init__(self, imgage_folder, annotation_file, feature_extractor):
        super(CustomDataset, self).__init__(imgage_folder, annotation_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CustomDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()
        target = encoding['labels'][0]
        return pixel_values, target