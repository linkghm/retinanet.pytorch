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
from loss import Memory
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
# train_transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
train_transform_list = [transforms.ToTensor()]
if cfg.scale is not None:
    train_transform_list.insert(0, transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

n_way = 5
n_support = 6  # max nshot ability -1
episode_length = n_way*n_support
# n_query = 5
emb_size = 128
memory_size = 512 # 1024  # 8192
# n_episodes = 10
batch_size = 8
validation_frequency = 25
delete_mem_every_episode = False
delete_mem_every_validation = False

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
                                 batch_size=batch_size,
                                 n_classes_p_i=1,
                                 n_objects_p_i=1,
                                 encoder=DataEncoder(),
                                 transform=train_transform)


valset = OmniglotDetectDataset(base_dir="/media/hayden/Storage21/DATASETS/IMAGE/OMNIGLOT/",
                                 split="val",
                                 n_way=n_way,
                                 n_support=n_support,
                                 batch_size=1,
                                 n_classes_p_i=1,
                                 n_objects_p_i=1,
                                 encoder=DataEncoder(),
                                 transform=train_transform)

# trainset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.train_imageset_fn,
#                         image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=train_transform)
# valset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.val_imageset_fn,
#                         image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=val_transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
#                                           num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
# valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
#                                         num_workers=cfg.num_workers, collate_fn=valset.collate_fn)



# Setup the model
print('Building model...')
mem = Memory(memory_size, emb_size)
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes), emb_size=emb_size, memory=True)

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

# Setup optimizer

# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, eps=1e-4)

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

# def train(episode):
#     print('\nTrain Episode: %d' % episode)
#     net.train()
#     # trainset.generate_episode()
#     loss = 0
#     sum_acc = 0
#     clss = [[],[]]
#     optimizer.zero_grad()
#     graph_vecs = []
#
#     for episode_idx in range(n_episodes):  # episodes at once
#         inputs, loc_targets, cls_targets = trainset.load_episode()
#
#         torch.cuda.empty_cache()
#         input = Variable(inputs.cuda())
#
#         loc_target = Variable(loc_targets.cuda())
#         cls_target = Variable(cls_targets.cuda())
#
#         loc_pred, cls_pred = net(input) # this builds mem on querys as we store for loss tracing the path to the loss?
#
#         loc_loss, cls_loss, acc, vectors = criterion(loc_pred, loc_target, cls_pred, cls_target)
#
#         loss += loc_loss + cls_loss  # avg loss over num queries
#         print('E: %02d | loc_loss: %.3f | cls_loss: %.3f | tot_loss: %.3f | acc: %.3f' % (episode_idx, loc_loss, cls_loss, loss/(episode_idx+1), acc))
#
#     loss = loss/n_episodes
#     loss.backward()
#     nn.utils.clip_grad_norm(net.parameters(), max_norm=1.2)
#     optimizer.step()
#
#     graph('/media/hayden/Storage21/MODELS/PROTINANET/vis/' + str(episode) + '.png', vectors)
#
#     # train_loss += loss.data[0]
#     # print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss / (batch_idx + 1)))
#     # save_checkpoint(train_loss, len(trainloader))
#     return loss.data[0]

def train(e):
    # TODO make sure can learn by deleting mem every episode, it generally can but just slower...
    if delete_mem_every_episode:
        mem.build()

    cummulative_loss = [0, 0]
    counter = 0
    correct = []

    inputs, loc_targets, cls_targets = trainset.load_mem_episode(e, view=True)
    for s in range(episode_length):  # goes across episode length xx is batch size len
        xx = inputs[s]
        # for b in range(batch_size):
        #     xx[b] = xx[b]*(b/batch_size)
        optimizer.zero_grad()
        xx_cuda = Variable(xx.cuda())
        loc_preds, cls_preds = net(xx_cuda)  # embed: (batch_size, key_dim)

        yy, yy_hat, softmax_embed, cls_loss, loc_loss, vectors = mem.query(loc_preds.cuda(), cls_preds.cuda(),
                                                Variable(loc_targets[s]).cuda(),
                                                Variable(cls_targets[s]).cuda(),
                                                predict=False)
        loss = cls_loss + loc_loss
        cummulative_loss[0] += cls_loss.data[0]
        cummulative_loss[1] += loc_loss.data[0]
        # cc = float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy.data, dim=1).cpu()))
        # correct.append(float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy.data, dim=1).cpu())))

        # if n_other > 0 and other_loss.data > .001:
        #     loss += other_loss
        loss.backward()
        optimizer.step()
        counter += 1

    # graph('/media/hayden/Storage21/MODELS/PROTINANET/vis/mem/' + str(e) + '.png',
    #       vectors.data.cpu().numpy(), yy.data.cpu().numpy(),
    #       mem, mean=False)
    print("episode batch: {0:d} average cls loss: {1:.6f} average loc loss: {2:.6f}".format(e, (cummulative_loss[0] / (counter)), (cummulative_loss[1] / (counter))))
    # print("episode batch: {0:d} average cls loss: {1:.6f} average loc loss: {2:.6f} : average acc: {3: .6f}".format(e, (cummulative_loss[0] / (counter)), (cummulative_loss[1] / (counter)), np.mean(correct)))

    # filter = net.conv1.weight.data.numpy()
    # (1/(2*(maximum negative value)))*filter+0.5 === you need to normalize the filter before plotting.
    # filter = (1 / (2 * 3.69201088)) * filter + 0.5  # Normalizing the values to [0,1]

    if e % validation_frequency == 0:
        # validation
        correct = []
        correct_by_k_shot = dict((k, list()) for k in range(n_way + 1))
        for i in range(50):
            inputs, loc_targets, cls_targets = valset.load_mem_episode(e)

            # erase memory before validation episode
            if delete_mem_every_validation:
                mem.build()

            y_hat = []
            y = []
            for s in range(episode_length):
                xx = inputs[s]

                xx_cuda = Variable(xx.cuda())
                loc_preds, cls_preds = net(xx_cuda)

                yy, yy_hat, embed, cls_loss, loc_loss, vectors = mem.query(loc_preds.cuda(),
                                                                  cls_preds.cuda(),
                                                                  Variable(loc_targets[s]).cuda(),
                                                                  Variable(cls_targets[s]).cuda(),
                                                                  predict=True)
                yy = yy.data.cpu()
                y_hat.append(yy_hat)
                y.append(yy)
                correct.append(float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy, dim=1))))

            # graph(zp, zq, name='TE_' + str(e) + "_" + str(counter))

            # compute per_shot accuracies
            seen_cls_reg = [int(y[i]) for i in range(n_way)]
            seen_count = [0 for idx in range(n_way)]
            # loop over episode steps
            for yy, yy_hat in zip(y, y_hat):
                # count = seen_count[yy[0] % n_way]
                count = seen_count[seen_cls_reg.index(yy[0])]
                if count < (n_way + 1):
                    correct_by_k_shot[count].append(float(torch.equal(yy_hat.cpu(), torch.unsqueeze(yy, dim=1))))
                seen_count[seen_cls_reg.index(yy[0])] += 1
                # seen_count[(yy[0]-1) % n_way] += 1

        # print("episode batch: {0:d} average loss: {1:.6f} average 'other' loss: {2:.6f}".format(e, (
        # cummulative_loss / (counter)), (cummulative_other_loss / (counter))))
        print("validation overall accuracy {0:f}".format(np.mean(correct)))

        for idx in range(n_way + 1):
            print("{0:d}-shot: {1:.3f}".format(idx, np.mean(correct_by_k_shot[idx])))
        # cummulative_loss = 0
        # counter = 0

def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def graph(path, vectors, labels, mem, mean=True):
    mks = mem.keys.cpu().numpy()
    mvs = mem.values.cpu().numpy().squeeze()

    mkms = []
    clss = []
    for i in range(len(labels)):
        inds = np.where(mvs == labels[i])
        mk = mks[tuple(inds)]
        if mean:
            mkm = mk.mean(0)
            mkms.append(mkm)
            clss.append(i)
        else:
            for mki in mk:
                mkms.append(mki)
                clss.append(i)

    mkms = np.array(mkms)
    mem_len = len(clss)
    clss += list(range(len(labels)))

    vectors = np.concatenate((mkms, vectors))
    # graph_vecs = np.array(graph_vecs).squeeze()
    if vectors.shape[1] > 2:
        try:
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
            vectors = tsne.fit_transform(vectors)
        except ValueError:
            print('Value Error')

    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

    for i in range(len(clss)):
        if i >= mem_len:
            plt.scatter(vectors[i, 0], vectors[i, 1], c=colors[clss[i]], label=clss[i])
        else:
            plt.scatter(vectors[i, 0], vectors[i, 1], facecolors='none', edgecolors=colors[clss[i]], label=clss[i])
    # plt.legend()
    # plt.show()
    plt.savefig(path)
    # plt.savefig('/media/hayden/Storage21/MODELS/PROTINANET/vis/'+str(episode)+'.pdf')


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
