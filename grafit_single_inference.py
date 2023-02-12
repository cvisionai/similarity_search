import torch
from torchvision import models, transforms
import pickle
import argparse
from grafit_pytorch import Grafit
import pytorch_lightning as pl
from grafit_pytorch.util import FathomnetDataset
from grafit_inference import SelfSupervisedLearner
from PIL import Image

NUM_GPUS   = 1
IMAGE_SIZE = 256
TRAIN_DATA_SIZE = 169516

def parse_args():
    parser = argparse.ArgumentParser(description='Run Grafit inference on input images.')
    parser.add_argument('--image', '-i', type=str, default='',
                        help="Image to run inference on.")
    parser.add_argument('--model', '-m', type=str, default='',
                        help="Grafit model file")
    parser.add_argument('--out_file',type=str, default='',
                        help="Output file location")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    img_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])

    resnet = models.resnet50(pretrained=True)
    resnet.to("cuda")

    # This is only constructed to get the dataset labels
    train_ds = FathomnetDataset('/mnt/md0/Projects/Fathomnet/Training_Files/2022-03-11-Detectron/train_file_v2_df.csv', root_dir='/mnt/md0/Projects/Fathomnet/Data_Files/2021-06-04-Download/',transform=img_transforms)

    # Because we don't use these here, can just replace with a default value of length of dataset
    labels = train_ds.get_labels()
    label_set = list(set(labels))
    label_set.sort()
    label_dict = {label : i for i,label in enumerate(label_set)}
    labels_numeric = [label_dict.get(label) for label in labels]
    model = SelfSupervisedLearner(resnet,image_size=IMAGE_SIZE,hidden_layer='avgpool',projection_size=256,projection_hidden_size=4096,moving_average_decay=0.99,dataset_size=TRAIN_DATA_SIZE,dataset_labels = labels_numeric)

    if args.model == '':
        checkpoint = torch.load('/home/ben/scratch/similarity_search/epoch99.ckpt')
    else:
        checkpoint = torch.load(args.model)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    img = Image.open(args.image)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img_transforms(img)
    results = model.predict_step(batch={'image' : torch.unsqueeze(img.cuda(),dim=0), 'index' : 0}, batch_idx=0).cpu().detach()
    if args.out_file == '':
        out_file = 'inference_results.pkl'
    else:
        out_file = args.out_file
    pickle.dump(results,open(out_file,'wb'))


