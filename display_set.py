import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


import voc.transforms as transforms
from encoder import DataEncoder
from voc.datasets import VocLikeDataset

parser = argparse.ArgumentParser(description='PyTorch Dataset Viewer')
parser.add_argument('--exp', required=True, help='experiment name')
args = parser.parse_args()

# Load the config file params from the exps directory specified in args
sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg

trainset_disp = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.train_imageset_fn,
                          image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=transforms.Compose([transforms.ToTensor()]))

def add_boxes(image, boxes, labels):
    """Show image with landmarks"""
    # Display the image
    ax.imshow(np.swapaxes(np.swapaxes(image.numpy(),0,2),0,1))
    for box in boxes:
        # Create a Rectangle patch
        bx = box.numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)


fig,ax = plt.subplots(1)
for i in range(1):
    sample = trainset_disp[i]

    print(i, sample['image'].shape, sample['boxes'].shape)

    plt.tight_layout()
    add_boxes(**sample)

plt.show()
