import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import voc.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss
from retinanet import RetinaNet
from voc.datasets import VocLikeDataset


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Prediction')
parser.add_argument('--exp', required=True, help='experiment name')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

sys.path.insert(0, os.path.join('exps', args.exp))
import config as cfg

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

print('Preparing data..')
train_transform_list = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    train_transform_list.insert(0, transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std)
])

# trainset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.train_imageset_fn,
#                           image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=train_transform)
valset = VocLikeDataset(image_dir=cfg.image_dir, annotation_dir=cfg.annotation_dir, imageset_fn=cfg.val_imageset_fn,
                        image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=val_transform, val=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
#                                           num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)

print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
# if args.resume:
print('Loading from checkpoint..')
checkpoint = torch.load(os.path.join('ckpts', args.exp, 'ckpt.pth'))
net.load_state_dict(checkpoint['net'])


net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

# criterion = FocalLoss(len(cfg.classes))
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#
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

def predict():
    net.eval()
    val_loss = 0
    loc_preds_lst = []
    cls_preds_lst = []
    bxs = []
    lbls = []
    for batch_idx, (inputs, img_sizes, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)

        loc_preds_npy = loc_preds.data.cpu().numpy()
        cls_preds_npy = cls_preds.data.cpu().numpy()

        max_h = max([list(im)[0] for im in img_sizes])
        max_w = max([list(im)[1] for im in img_sizes])


        for i in range(8):
            boxes, labels = DataEncoder().decode(loc_preds[i], cls_preds[i], input_size=(max_w, max_h))
            if boxes is not None:
                bxs.append(boxes)
                lbls.append(labels)
    print('h')
    #     # boxes[i], labels[i], input_size = (max_w, max_h)
    #     batch_size, num_boxes = cls_targets.size()
    #     pos = cls_targets > 0
    #     num_pos = pos.data.long().sum()
    #
    #     mask = pos.unsqueeze(2).expand_as(loc_preds)
    #     masked_loc_preds = loc_preds[mask].view(-1, 4)
    #     masked_loc_targets = loc_targets[mask].view(-1, 4)
    #     # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
    #
    #     pos_neg = cls_targets > -1
    #     mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
    #     masked_cls_preds = cls_preds[mask].view(-1, len(cfg.classes) + 1)
    #     # cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
    #
    #     # loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
    #     # val_loss += loss.data[0]
    #     # print('val_loss: %.3f | avg_loss: %.3f' % (loss.data[0], val_loss/(batch_idx+1)))
    # # save_checkpoint(val_loss, len(valloader))

# def save_checkpoint(loss, n):
#     global best_loss
#     loss /= n
#     if loss < best_loss:
#         print('Saving..')
#         state = {
#             'net': net.module.state_dict(),
#             'loss': loss,
#             'epoch': epoch,
#             'lr': lr
#         }
#         ckpt_path = os.path.join('ckpts', args.exp)
#         if not os.path.isdir(ckpt_path):
#             os.makedirs(ckpt_path)
#         torch.save(state, os.path.join(ckpt_path, 'ckpt.pth'))
#         best_loss = loss

for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    # if epoch in cfg.lr_decay_epochs:
    #     lr *= 0.1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # train(epoch)

    predict()
