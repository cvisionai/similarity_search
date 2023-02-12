# import the necessary packages
import os
import logging

import torch
from torchvision import models, transforms
import pickle
import argparse
from grafit_pytorch import Grafit
import pytorch_lightning as pl
from grafit_pytorch.util import FathomnetDataset
from grafit_inference import SelfSupervisedLearner
from PIL import Image

import numpy as np
import helpers
import redis
import time
import json

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# connect to Redis server
db = redis.StrictRedis(host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))

class DictToDotNotation:
    '''Useful class for getting dot notation access to dict'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def parse_args():
    parser = configargparse.ArgumentParser(description="Testing script for testing video data.",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add("-c", "--config", required=True, is_config_file=True, help="Default config file path")
    parser.add_argument('--model', '-m', type=str, default='',
                        help="Grafit model file")
    parser.add_argument('--out_file',type=str, default='',
                        help="Output file location")

    return parser.parse_args()

def load_model(args):
    '''Load model architecture and weights
    '''
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
    return model

def process_nms(model_outputs):
    '''Seperate function so that we can substitute more complex logic (e.g. inter vs intra class NMS)
    '''
    nms_op = torchvision.ops.nms
    model_outputs[0]["instances"] = model_outputs[0]["instances"][nms_op(model_outputs[0]["instances"].pred_boxes.tensor, model_outputs[0]["instances"].scores, 0.45).to("cpu").tolist()]

    return model_outputs


def classify_process():

    img_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(256),transforms.ToTensor()])
    img = Image.open(args.image)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img_transforms(img)
    results = model.predict_step(batch={'image' : torch.unsqueeze(img.cuda(),dim=0), 'index' : 0}, batch_idx=0).cpu().detach()


    args = {}
    args['model_config_file'] = 'fathomnet_config_v2_1280.yaml'
    args['model_weights_file'] = 'model_final.pth'
    args['score_threshold'] = 0.3
    args = DictToDotNotation(args)

    logger.info("Loading model...")
    model = load_model(args)
    logger.info("Loading complete!")

    augs = create_augmentations()
    label_map = create_labelset()
    # continually pool for new images to classify
    while True:
        # monitor queue for jobs and grab one when present
        q = db.blpop(os.getenv("IMAGE_QUEUE_DETECTRON2"))
        logger.info(q[0])
        q = q[1]
        imageIDs = []
        batch = None

        # deserialize the object and obtain the input image
        q = json.loads(q.decode("utf-8"))
        img_width = q["width"]
        img_height = q["height"]
        image = helpers.base64_decode_image(q["image"],
            os.getenv("IMAGE_DTYPE"),
            (1, img_height, img_width,
                int(os.getenv("IMAGE_CHANS"))))

        # check to see if the batch list is None. Currently
        # only batch size of 1 is supported, future growth.
        if batch is None:
            batch = image
        # otherwise, stack the data
        else:
            batch = np.vstack([batch, image])

        # update the list of image IDs
        imageIDs.append(q["id"])

        # check to see if we need to process the batch. 
        # Currently only batch size of 1 is supporeted,
        # future growth.
        if len(imageIDs) > 0:
            logger.info(imageIDs)
            batch_results = []
            # Create augmented images based on list of augmentations
            logger.info(f"Augmenting {len(batch)} images")
            aug_imgs = augment_images(augs, batch)
            # Run model
            for aug_img_batch in aug_imgs:
                logger.info(f"Running model")
                model_outputs = process_model(model, aug_img_batch, img_width, img_height)
                # Process model outputs
                logger.info(f"Processing outputs")
                output_boxes, output_scores, output_classes = process_model_outputs(model_outputs)

                results = []
                for box,scores,label in zip(output_boxes, output_scores, output_classes):
                    image_result = {
                            'category_id' : label_map[label],
                                'scores'      : [scores],
                                'bbox'        : box,
                            }
                
                    results.append(image_result)
                batch_results.append(results)
            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, batch_results):
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(resultSet))


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    classify_process()
