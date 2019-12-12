"""
Mask R-CNN
Train on the toy gta5 dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 gta5.py train --dataset=/path/to/gta5/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 gta5.py train --dataset=/path/to/gta5/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 gta5.py train --dataset=/path/to/gta5/dataset --weights=imagenet

    # Apply color splash to an image
    python3 gta5.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 gta5.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from utilities import *
import pandas as pd
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class gta5Config(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "gta5"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + gta5

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Resize image according to dataset
    IMAGE_MAX_DIM = 2048
    IMAGE_SHAPE = [1914 1052 3]




############################################################
#  Dataset
############################################################

class gta5DataSet(utils.Dataset):
    def load_gta5(self, Root_Path='/content/drive/My Drive/rob535_perception/images/', ifTest:bool=False, start_idx: int=0, end_idx: int=599, ifInfere: bool=False):
        #additional initilization source named as "ROB535"
        self.source  = "gta5"
        self.root_path = Root_Path
        tr_data = np.loadtxt(Root_Path.replace('images','data') + 'training_data.csv', skiprows=1, dtype=str, delimiter=',')
        np.random.shuffle(tr_data)
        relative_img_path = tr_data[:,0]
        bbox = tr_data[:,1:10].astype(np.float32)
        labels = tr_data[:,-1]
        self.img_list = [self.root_path + 'train/' + img for img in relative_img_path]
        self.train_img_list = self.img_list[end_idx:]
        #self.test_img_list = glob(self.root_path+'test/*/*_image.jpg')
        # self.train_img_list = glob(self.root_path+'train/*/*_image.jpg')
        if ifTest:
            self.test_img_list = self.img_list[start_idx:end_idx]
        if ifInfere:
            self.test_img_list = glob(self.root_path+'test/*/*_image.jpg')

        self.add_class(self.source,1,'Cars')
        self.add_class(self.source,2,'Trucks')
        self.add_class(self.source,3,'Bikes')

        if ifTest:
            for i,path in enumerate(self.test_img_list):
                self.add_image(self.source, i, path)
        else:
            for i,path in enumerate(self.train_img_list):
                self.add_image(self.source, i, path, "bbox"=bbox[i,:], "label"=labels[i])

    def image_reference(self, image_id):
        """Return path of the image."""
        info = self.image_info[image_id]
        return info["path"]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        proj = get_proj_mtx(image_reference(self, image_id))
        info = self.image_info[image_id]
        # size = self.bbox[6:9]
        # p0 = -size / 2
        # p1 = size / 2
        # [bbox_3d_v, bbox_3d_e] = get_bbox(self, p0,p1)
        # bbox_3d_v = np.vstack([bbox_3d_v, np.ones_like(bbox_3d_v[0,:])])
        # bbox_2d_v = proj.dot(bbox_3d_v)
        # bbox_2d_v /= bbox_2d_v[-1,:]
        # V,C = get_2d_bbox(bbox_2d_v)
        [bbox_3d_v,bbox_3d_e]  = get_3d_bbox(info["bbox"])
        bbox_3d_v = np.vstack([bbox_3d_v, np.ones_like(bbox_3d_v[0,:])])
        bbox_2d_v = proj.dot(bbox_3d_v)
        bbox_2d_v /= bbox_2d_v[-1,:]
        V = get_2d_bbox(bbox_2d_v)
        mask = get_mask(V)
        label = info["label"]
        return mask, label

    def get_bbox(self, p0, p1):
        """
        Input:
        *   p0, p1
            (3)
            Corners of a bounding box represented in the body frame.
        """
        b_box = self.bbox
        v = np.array([
                     [p0[0],p0[0],p0[0],p0[0],p1[0],p1[0],p1[0],p1[0]],
                     [p0[1],p0[1],p1[1],p1[1],p0[1],p0[1],p1[1],p1[1]],
                     [p0[2],p1[2],p0[2],p1[2],p0[2],p1[2],p0[2],p1[2]]
                    ], dtype=np.int32)
        e = np.array([
                     [2,3,0,0,3,3,0,1,2,3,4,4,7,7],
                     [7,6,1,2,1,2,4,5,6,7,5,6,5,6]
                     ], dtype=np.uint8)
        R = rot(b_box[0:3])
        t = b_box[3:6]
        vertices = R.dot(vertices)
        vertices[0,:] += t[0]
        vertices[1,:] += t[1]
        vertices[2,:] += t[2]
        return vertices, edges
        """
        Output:
        *   v
            (3, 8)
            Vertices of the bounding box represented in the body frame.
        *   e
            (2, 14)
            Edges of the bounding box. The first 2 edges indicate the `front` side
            of the box.
        """

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = gta5Dataset()
    dataset_train.load_gta5(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = gta5Dataset()
    dataset_val.load_gta5(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect gta5s.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/gta5/dataset/",
                        help='Directory of the gta5 dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = gta5Config()
    else:
        class InferenceConfig(gta5Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
