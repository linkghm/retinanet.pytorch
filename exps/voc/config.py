import os

root = os.path.join('/media/hayden/Storage21/DATASETS/IMAGE/VOC')
image_dir = os.path.join(root, 'IMAGES', '2012')
annotation_dir = os.path.join(root, 'ANNOTATIONS', '2012', 'XML')
train_imageset_fn = os.path.join(root, 'SPLITS', '2012', 'Main', 'trainval.txt')
val_imageset_fn = os.path.join(root, 'SPLITS', '2012', 'Main', 'val.txt')
image_ext = '.jpg'

# backbone = 'resnet50'
# # backbone = 'resnet34'
# backbone = 'resnet18'
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
scale = None

batch_size = 1
lr = 0.01
# lr = 0.001
momentum = 0.8
# momentum = 0.5
weight_decay = 1e-4
num_epochs = 1000
lr_decay_epochs = [1000]#[83, 110]
num_workers = 8

eval_while_training = True
eval_every = 1
