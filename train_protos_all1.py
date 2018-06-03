import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import voc.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss, ProtosLoss, ProtosLossOne
from retinanet import RetinaNet
from voc.datasets import VocLikeDataset, VocLikeProtosDataset, OmniglotDetectDataset


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Load the config file params from the exps directory specified in args
sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg

# Check for Cuda
assert torch.cuda.is_available(), 'Error: CUDA not found!'

# Initialise vars
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

# Set up the transforms for each set
print('Preparing data..')
train_transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
train_transform_list = [transforms.ToTensor()]
if cfg.scale is not None:
    train_transform_list.insert(0, transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

n_way = 5
n_support = 5
n_query = 5
emb_size = 2
n_episodes = 10
# Load the sets, and loaders
# trainset = VocLikeProtosDataset(image_dir=cfg.image_dir,
#                                 annotation_dir=cfg.annotation_dir,
#                                 imageset_fn=cfg.train_imageset_fn,
#                                 image_ext=cfg.image_ext,
#                                 classes=cfg.classes,
#                                 n_way=n_way,
#                                 n_support=n_support,
#                                 n_query=n_query,
#                                 encoder=DataEncoder(),
#                                 transform=train_transform)

trainset = OmniglotDetectDataset(base_dir="/media/hayden/Storage21/DATASETS/IMAGE/OMNIGLOT/",
                                 split="val",
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query,
                                 n_classes_p_i=1,
                                 n_objects_p_i=1,
                                 encoder=DataEncoder(),
                                 transform=train_transform)

# trainset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.train_imageset_fn,
#                         image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=train_transform)
valset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.val_imageset_fn,
                        image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=val_transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
#                                           num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)



# Setup the model
print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes), emb_size=emb_size)

# If we loading from a checkpoint, load it
if args.resume:
    print('Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join('ckpts', args.exp, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    lr = checkpoint['lr']

# 'Parallelise' the net
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# Use the GPU
net.cuda()
cudnn.benchmark = True

# Setup loss and optimizer
# criterion = FocalLoss(len(cfg.classes))
criterion = ProtosLossOne(n_way=n_way,
                       n_support=n_support,
                       n_query=n_query,
                       emb_size=emb_size)

# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

# def train(epoch):
#     print('\nTrain Epoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
#         inputs = Variable(inputs.cuda())
#         loc_targets = Variable(loc_targets.cuda())
#         cls_targets = Variable(cls_targets.cuda())
#
#         optimizer.zero_grad()
#         loc_preds, cls_preds = net(inputs)
#         loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
#         loss.backward()
#         nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
#         optimizer.step()
#
#         train_loss += loss.data[0]
#         print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
#     save_checkpoint(train_loss, len(trainloader))

def train(episode):
    print('\nTrain Episode: %d' % episode)
    net.train()
    # trainset.generate_episode()
    loss = 0
    sum_acc = 0
    clss = [[],[]]
    optimizer.zero_grad()
    graph_vecs = []

    for episode_idx in range(n_episodes):  # episodes at once
        inputs, loc_targets, cls_targets = trainset.load_episode()

        torch.cuda.empty_cache()
        input = Variable(inputs.cuda())

        loc_target = Variable(loc_targets.cuda())
        cls_target = Variable(cls_targets.cuda())

        loc_pred, cls_pred = net(input) # this builds mem on querys as we store for loss tracing the path to the loss?

        loc_loss, cls_loss, acc, vectors = criterion(loc_pred, loc_target, cls_pred, cls_target)

        loss += loc_loss + cls_loss  # avg loss over num queries
        print('E: %02d | loc_loss: %.3f | cls_loss: %.3f | tot_loss: %.3f | acc: %.3f' % (episode_idx, loc_loss, cls_loss, loss/(episode_idx+1), acc))

    loss = loss/n_episodes
    loss.backward()
    nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
    optimizer.step()

    graph('/media/hayden/Storage21/MODELS/PROTINANET/vis/' + str(episode) + '.png', vectors)

    # train_loss += loss.data[0]
    # print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss / (batch_idx + 1)))
    # save_checkpoint(train_loss, len(trainloader))
    return loss.data[0]

def graph(path, vectors):
    # sups = criterion.supports.cpu().data.numpy().reshape((n_support*n_way, emb_size))

    # graph_vecs = np.array(graph_vecs).squeeze()
    if vectors.shape[1] > 2:
        try:
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(vectors)
        except ValueError:
            print('h')
    else:
        tsne_results = vectors

    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, label in enumerate(vectors):
        if i < n_way * n_support:
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], c=colors[int(i / n_support)], label=int(i / n_support))
        else:
            plt.scatter(tsne_results[i, 0], tsne_results[i, 1], facecolors='none',
                        edgecolors=colors[int((i - (n_way * n_support)) / n_query)],
                        label=int((i - (n_way * n_support)) / n_query))
    # plt.legend()
    # plt.show()
    plt.savefig(path)
    # plt.savefig('/media/hayden/Storage21/MODELS/PROTINANET/vis/'+str(episode)+'.pdf')

def val(epoch):
    net.eval()
    val_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        val_loss += loss.data[0]
        print('val_loss: %.3f | avg_loss: %.3f' % (loss.data[0], val_loss/(batch_idx+1)))
    save_checkpoint(val_loss, len(valloader))

def save_checkpoint(loss, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'lr': lr
        }
        ckpt_path = os.path.join('ckpts', args.exp)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path, 'ckpt.pth'))
        best_loss = loss

for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    if epoch in cfg.lr_decay_epochs:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    train(epoch)

    # if cfg.eval_while_training and epoch % cfg.eval_every == 0:
    #     val(epoch)
