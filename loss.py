from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)

class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        y = one_hot(y.cpu(), x.size(-1)).cuda()
        logit = F.softmax(x)
        # logit = F.sigmoid(x)
        logit = logit.clamp(1e-7, 1. - 1e-7)

        loss = -1 * y.float() * torch.log(logit)
        loss = loss * (1 - logit) ** 2
        return loss.sum()
 
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        mask = pos.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].view(-1,4)
        masked_loc_targets = loc_targets[mask].view(-1,4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes + 1)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss


class ProtosLoss(nn.Module):
    def __init__(self, n_way, n_support, n_query, emb_size, other_alpha=1):
        super(ProtosLoss, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.emb_size = emb_size
        self.other_alpha = other_alpha
        self.reset()  # setup mem and counter

    def reset(self):
        """
        used to reset the memory (protos and counter per episode)
        :return:
        """
        self.s_count = 0  # use this to know what sample we are up to for sup/query eval
        self.q_count = 0  # use this to know what sample we are up to for sup/query eval
        # self.protos = torch.zeros(n_way, emb_size).long().cuda()  # this stores our proto sums, will be avgd after all support done
        self.supports = Variable(torch.zeros(self.n_way, self.n_support, self.emb_size).float().cuda(), requires_grad=False)  # this stores our supports, rather than summing we hold for 'other' bound calcs
        self.proto_bounds = Variable(torch.zeros(self.n_way), requires_grad=False)

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def proto_loss(self, x, y):
        when_avg = False  # when to apply avg, default is false, there is a difference in the values of these losses, think more about it

        pos = y > 0  # all of the positions of the classes for this query sample
        num_pos = pos.data.long().sum()  # number of positive anchors that match the class in this query sample

        if when_avg:
            # calc dists between every 'other' and protos
            dists = self.euclidean_dist(x, self.protos)
            log_p_y = F.log_softmax(-dists)

            # this gets the loss by averaging the pos softmax's per class and then taking the appropriate gt class indx
            # here we do the mean on the SMs after distance calc
            loss = -log_p_y[pos.unsqueeze(1)].view(-1, self.n_way).mean(0)[int(self.q_count / self.n_query)]
        else:
            # here we do the mean of the query embeddings then the distance calc
            mean_query_anchs = x[pos.unsqueeze(1)].view(-1, self.emb_size).mean(0)
            dists2 = self.euclidean_dist(mean_query_anchs.unsqueeze(0), self.protos)
            log_p_y = F.log_softmax(-dists2)
            loss = -log_p_y.squeeze()[int(self.q_count / self.n_query)]
        print('d')

        # y = one_hot(y.cpu(), x.size(-1)).cuda()
        # logit = F.softmax(x)
        # # logit = F.sigmoid(x)
        # logit = logit.clamp(1e-7, 1. - 1e-7)
        #
        # loss = -1 * y.float() * torch.log(logit)
        # loss = loss * (1 - logit) ** 2
        return loss#loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        print(self.s_count)

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()  # the number of gt anchors for the class/es we interested in for a single input image

        # if support hold data for mean in memory and have to hold queries too? no we need to pass all support first hold them then queries one by one to calc loss

        if self.s_count < self.n_support * self.n_way:  # is support sample
            # mask out 'ignore' and 'other' boxes to not affect
            if num_pos > 0:  # todo no samples in this.. take avg of prev or keep as 0s or dont avg over boxes per img but fill sup until filled then skip
                mask = pos.unsqueeze(2).expand_as(cls_preds)
                masked_cls_preds = cls_preds[mask].view(-1, self.emb_size)

                # get mean of all boxes in this image to add as a support example
                support = masked_cls_preds.mean(0)
                self.supports[int(self.s_count / self.n_support), self.s_count % self.n_support] = support.data
            else:

                if self.s_count % self.n_support > 0:
                    # take mean of already passed vectors
                    sd = self.supports[int(self.s_count / self.n_support), :self.s_count % self.n_support].mean(0).data
                    self.supports[int(self.s_count / self.n_support), self.s_count % self.n_support] = sd
                else:
                    pass # have to leave as 0s for now
                print("not enough samples : class %d" % (int(self.s_count / self.n_support)))
            # when building the support / protos mem we have no loss
            cls_loss = 0 #TODO change to a tensor of zeros -- Variable(torch.zeros(1).float().cuda(), requires_grad=False)
            loc_loss = 0
            self.s_count += 1
        else:
            if self.q_count == 0:  # is first query sample, we need to mean the protos, and create the bounds
                self.protos = self.supports.mean(1)
                for i in range(self.n_way):
                    self.proto_bounds[i] = self.euclidean_dist(self.supports[i], self.protos[i].unsqueeze(0)).max().data

            mask = pos.unsqueeze(2).expand_as(loc_preds)  # mask 'other' and 'ignore' boxes out to not affect
            masked_loc_preds = loc_preds[mask].view(-1, 4)
            masked_loc_targets = loc_targets[mask].view(-1, 4)
            loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)


            pos_neg = cls_targets > -1  # mask out 'ignore' boxes to not affect
            mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
            masked_cls_preds = cls_preds[mask].view(-1, self.emb_size)
            cls_loss = self.proto_loss(masked_cls_preds, cls_targets[pos_neg])
            self.q_count += 1

        if num_pos > 0:
            loss = cls_loss + (loc_loss / num_pos)
            # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0] / num_pos, cls_loss.data[0]),
            print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss / num_pos, cls_loss),
                  end=' | ')
        else:
            loss = 0
        return loss