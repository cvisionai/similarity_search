import logging
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class FathomnetDataset(Dataset):
    """Data Set for loading Fathomnet ROIs"""
    def __init__(self, csv_file, root_dir, transform=None, start_idx=None):
        self.root_dir = root_dir
        self.rois = pd.read_csv(csv_file)
        if start_idx is not None:
            self.rois = self.rois.loc[start_idx:]
        self.transform = transform

    def __len__(self):
        return (len(self.rois))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.rois.iloc[idx,0])
        x1,y1,x2,y2 = self.rois.iloc[idx,1:5]
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Bounds checking
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > image.width:
            x2 = image.width
        if y2 > image.height:
            y2 = image.height

        assert(x1 < x2 and y1 < y2), f"Bad ROI dimensions: {x1,y1,x2,y2} in {img_name}"
        image = image.crop((x1,y1,x2,y2))
        image = self.transform(image)
        sample = {"image" : image, "name" : self.rois.iloc[idx,0], "label" : self.rois.iloc[idx,5], "roi" : {"x1" : x1,"y1" : y1, "x2" : x2, "y2" : y2}}

        return sample

if __name__ == "__main__":
    logger.info("Creating dataset and loader...")
    dataset = FathomnetDataset(
                    csv_file = '/mnt/md0/Projects/Fathomnet/Training_Files/2022-03-11-Detectron/train_file_v2_df.csv',
                    root_dir = '/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download',
                    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))

    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=False, num_workers=8,pin_memory=True)

    logger.info(f"Complete. Number of batches: {len(dataset) / 8}")
    logger.info("Creating model...")
    model = torchvision.models.mobilenetv3.mobilenet_v3_large(pretrained=True)
    model = nn.Sequential(*list(model.features),model.avgpool)
    model.eval()
    model.to('cuda')

    logger.info("Starting inference")
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            batch = sample_batched.get("image").to('cuda')
            names = sample_batched.get("name")
            labels = sample_batched.get("label")
            rois = sample_batched.get("roi")
            output = np.squeeze(model(batch).to("cpu").numpy())
            dummy_idx=0

            x1s = rois.get('x1')
            x2s = rois.get('x2')
            y1s = rois.get('y1')
            y2s = rois.get('y2')

            for features,fname,label,x1,y1,x2,y2 in zip(output,names,labels,x1s,y1s,x2s,y2s):
                pickle.dump(
                    {"features" : features, "fname" : fname, "label" : label, "roi" : {"x1" : x1.item(), "y1" : y1.item(), "x2" : x2.item(), "y2" : y2.item()}}, 
                    open(os.path.join(
                        '/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download-ROI-Features',
                        f'batch_{i_batch}_num_{dummy_idx}_{label.replace(" ", "-")}_features.pkl'),'wb')
                )
                dummy_idx += 1
            if i_batch % 100 == 0:
                logger.info(f"Processing batch {i_batch}...")
            if i_batch >= 21190:
                break