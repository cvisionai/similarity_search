import os
import argparse
import multiprocessing
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from grafit_pytorch import Grafit
import pytorch_lightning as pl

from grafit_pytorch.util import FathomnetDataset

# test model, a resnet 50

resnet = models.resnet50(pretrained=True)
resnet.to("cuda")

# arguments

parser = argparse.ArgumentParser(description='grafit-lightning-test')

parser.add_argument('--csv_file', type=str, required = True,
                       help='path to your csv_file containing fathomnet image information')

args = parser.parse_args()

# constants

BATCH_SIZE = 16
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = Grafit(net, **kwargs)

    def forward(self, sample):
        img = sample.get('image')
        idx = sample.get('index')
        return self.learner(img,idx)

    def training_step(self, sample, _):
        loss = self.forward(sample)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

# main

if __name__ == '__main__':
    img_transforms = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    ds = FathomnetDataset(args.csv_file, root_dir='/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download/',transform=img_transforms)
    labels = ds.get_labels()
    label_set = list(set(labels))
    label_set.sort()
    label_dict = {label : i for i,label in enumerate(label_set)}
    labels_numeric = [label_dict.get(label) for label in labels]
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        dataset_size = len(labels),
        dataset_labels = labels_numeric
    )

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader)