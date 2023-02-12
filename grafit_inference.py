import os
import argparse
import multiprocessing
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import pickle

from grafit_pytorch import Grafit
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from grafit_pytorch.util import FathomnetDataset

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = Grafit(net, **kwargs)

    def forward(self, sample):
        img = sample.get('image')
        idx = sample.get('index')
        return self.learner(img,idx)

    def predict_step(self, batch, batch_idx):
        img = batch.get('image')
        idx = batch.get('index')
        proj, _ = self.learner(img, idx, return_embedding=True)
        return proj 


    def training_step(self, sample, _):
        loss = self.forward(sample)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

if __name__ == "__main__":
    
    BATCH_SIZE = 24
    EPOCHS     = 10
    LR         = 0.001
    NUM_GPUS   = 1
    IMAGE_SIZE = 256
    IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
    NUM_WORKERS = multiprocessing.cpu_count()
    TRAIN_DATA_SIZE = 169516

    img_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])
    '''
    img_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
            transforms.ToTensor()
        ])
    '''
    train_ds = FathomnetDataset('/mnt/md0/Projects/Fathomnet/Training_Files/2022-03-11-Detectron/train_file_v2_df.csv', root_dir='/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download/',transform=img_transforms)
    ds = FathomnetDataset('/mnt/md0/Projects/Fathomnet/Training_Files/2022-03-11-Detectron/val_file_v2_df.csv', root_dir='/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download/',transform=img_transforms)
    labels = train_ds.get_labels()
    label_set = list(set(labels))
    label_set.sort()
    label_dict = {label : i for i,label in enumerate(label_set)}
    labels_numeric = [label_dict.get(label) for label in labels]

    resnet = models.resnet50(pretrained=True)
    resnet.to("cuda")

    model = SelfSupervisedLearner(
            resnet,
            image_size = IMAGE_SIZE,
            hidden_layer = 'avgpool',
            projection_size = 256,
            projection_hidden_size = 4096,
            moving_average_decay = 0.99,
            dataset_size = TRAIN_DATA_SIZE,
            dataset_labels = labels_numeric
        )

    checkpoint = torch.load('/home/ben/code/similarity_search/grafit_pytorch/examples/lightning/lightning_logs/version_1/checkpoints/epoch=299-step=2119200.ckpt')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    trainer = Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloaders=train_loader)
    pickle.dump(predictions,open('/home/ben/code/similarity_search/grafit_inference_results_val-epoch299.pkl','wb'))
