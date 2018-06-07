from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as ag

import math
import functools

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
        if num_pos <= 0:
            return 0, None, 0
        if when_avg:
            # calc dists between every 'other' and protos
            dists = self.euclidean_dist(x, self.protos)
            log_p_y = F.log_softmax(-dists)

            # this gets the loss by averaging the pos softmax's per class and then taking the appropriate gt class indx
            # here we do the mean on the SMs after distance calc
            loss = -log_p_y[pos.unsqueeze(1)].view(-1, self.n_way).mean(0)[int(self.q_count / self.n_query)]
        else:
            # here we do the mean of the query embeddings then the distance calc
            mean_query_anchs = x[pos.unsqueeze(1)].view(-1, self.emb_size)[0]#.mean(0) # take mean or just one anchor
            vectors = mean_query_anchs
            dists2 = self.euclidean_dist(mean_query_anchs.unsqueeze(0), self.protos)
            log_p_y = F.log_softmax(-dists2)
            i = Variable(torch.zeros(self.n_way), requires_grad=False).cuda()
            i[(int(self.q_count / self.n_query))]=1
            loss = (-log_p_y.squeeze()*i).sum() # todo these indecies might not allow loss to flow correctly as not torch vars

        _, y_hat = log_p_y.max(1)
        _, yb = i.max(0)
        acc_val = torch.eq(y_hat, yb).float().mean()
        # y = one_hot(y.cpu(), x.size(-1)).cuda()
        # logit = F.softmax(x)
        # # logit = F.sigmoid(x)
        # logit = logit.clamp(1e-7, 1. - 1e-7)
        #
        # loss = -1 * y.float() * torch.log(logit)
        # loss = loss * (1 - logit) ** 2
        return loss, vectors, acc_val#loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()  # the number of gt anchors for the class/es we interested in for a single input image

        # is support sample
        if self.s_count < self.n_support * self.n_way:

            # mask out 'ignore' and 'other' boxes to not affect
            if num_pos > 0:  # todo no samples in this.. take avg of prev or keep as 0s or dont avg over boxes per img but fill sup until filled then skip
                mask = pos.unsqueeze(2).expand_as(cls_preds)
                masked_cls_preds = cls_preds[mask].view(-1, self.emb_size)

                # get mean of all boxes in this image to add as a support example
                support = masked_cls_preds[0]#.mean(0) #mean or first element
                self.supports[int(self.s_count / self.n_support), self.s_count % self.n_support] = support.data
            else:

                if self.s_count % self.n_support > 0:
                    # take mean of already passed vectors
                    support = self.supports[int(self.s_count / self.n_support), :self.s_count % self.n_support][0].data#.mean(0).data # mean or first element
                    self.supports[int(self.s_count / self.n_support), self.s_count % self.n_support] = support
                else:
                    pass # have to leave as 0s for now
                print("not enough samples : class %d" % (int(self.s_count / self.n_support)))

            # when building the support / protos mem we have no loss
            cls_loss = 0 #TODO change to a tensor of zeros -- Variable(torch.zeros(1).float().cuda(), requires_grad=False)
            loc_loss = 0
            acc = 0
            vector = self.supports[int(self.s_count / self.n_support), self.s_count % self.n_support]
            self.s_count += 1
        else:  # is query sample
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
            cls_loss, vector, acc = self.proto_loss(masked_cls_preds, cls_targets[pos_neg])
            self.q_count += 1

        if num_pos > 0:
            return (loc_loss / num_pos), (cls_loss/ num_pos), acc, vector
            # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0] / num_pos, cls_loss.data[0]),
            # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss / num_pos, cls_loss),
            #       end=' | ')
        else:
            return 0, 0, 0, None

class ProtosLossOne(nn.Module):
    def __init__(self, n_way, n_support, n_query, emb_size, other_alpha=1):
        super(ProtosLossOne, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.emb_size = emb_size
        self.other_alpha = other_alpha

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


    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        cosine = False
        normalise = False
        if cosine:
            normalise = True
        target_inds = torch.arange(0, self.n_way).view(self.n_way, 1, 1).expand(self.n_way, self.n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).cuda()

        # batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0

        # num_pos = pos.data.long().sum(1)#pos.sum(1)#.float()  # the number of gt anchors for the class/es we interested in for a single input image
        num_pos = pos.float().sum(1)#pos.sum(1)#.float()  # the number of gt anchors for the class/es we interested in for a single input image

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # mask 'other' and 'ignore' boxes out to not affect
        masked_loc_preds = loc_preds[mask].view(-1, 4)
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        loc_loss = loc_loss/num_pos.sum()
        loc_loss.data[loc_loss.data == float("inf")] = 0

        pos_neg = cls_targets > -1  # mask out 'ignore' boxes to not affect

        # mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        maskb = pos.unsqueeze(2).expand_as(cls_preds)  # get a 0,1 mask to get the interesting anchors
        # samples = (cls_preds*mask.float()).data.sum(1)  # multiply mask by input then sum per image x
        samples = (cls_preds*maskb.float()).sum(1)  # multiply mask by input then sum per image x
        samples = samples/num_pos.float().unsqueeze(1).expand_as(samples)  # lastly divide to get the average emb per x

        if normalise:
            samples = F.normalize(samples, p=2, dim=1)
        zs = samples[:self.n_way*self.n_support].view(self.n_way, self.n_support, -1)
        zq = samples[self.n_way*self.n_support:]#.view(self.n_way, self.n_query, -1)
        zp = zs.mean(1)

        if cosine:
            # zp = zp.repeat(1, self.n_query).view(self.n_way*self.n_query,self.emb_size)  # need to use repeat for cosine sim
            zp2 = zp.repeat(1, self.n_query*self.n_way).view(self.n_way*self.n_way*self.n_query,self.emb_size)  # need to use repeat for cosine sim
            zq2 = zq.repeat(1, self.n_way).transpose(0,1).contiguous().view(self.n_way*self.n_way*self.n_query,self.emb_size)
        if cosine:
            # dists = F.cosine_similarity(zq2, zp2)
            dists = F.cosine_similarity(zq2, zp2).view(self.n_way*self.n_query, self.n_way).abs()
        else:
            dists = self.euclidean_dist(zq, zp)#*100000 # when dist values are small (.00nnn) then the log_p_y values all tend to 1.099
        # dd = F.softmax(-dists)
        # dd2 = F.softmax(-dists,dim=0)
        # dd3 = F.softmax(-dists,dim=1)
        log_p_y = F.log_softmax(-dists).view(self.n_way, self.n_query, -1)
        # log_p_y = -dists.view(self.n_way, self.n_query, -1) #

        # Push the others away loss (rest_loss)
        maskc = 1 - torch.eye(self.n_way, self.n_way).repeat(1, self.n_query).view(self.n_way, self.n_query, self.n_way)
        maskc = Variable(maskc, requires_grad=False).cuda()

        # vv = (log_p_y*maskc).sum(1).sum(0)/(self.n_query)
        # vvv = vv
        # vvvv = vvv
        rest_loss = -1000/(log_p_y*maskc).mean()

        origin_loss = self.euclidean_dist(zp, Variable(torch.zeros(1,self.emb_size)).cuda()).mean()

        # log_p_y = -dists.view(self.n_way, self.n_query, -1)
        # print(log_p_y.data)
        cls_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() # gather fails on cosine...

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        # cls_loss += rest_loss
        # if float(origin_loss.data.cpu()) > 1:
        #     cls_loss += origin_loss

        #
        # n_other = int(list(support_flags.size())[0]) - (n_way * (n_support + n_query))
        # if n_other:
        #     z_sup = z[sf]
        #     ps_dists = self.euclidean_dist(z_sup, z_proto).view(n_way, n_support, n_way)
        #     identity = torch.autograd.Variable(torch.eye(n_way), requires_grad=False).cuda()
        #     ps_dists = ps_dists.max(1)[0] * identity
        #     ps_dists = ps_dists.max(0)[0] * alpha
        #     ps_dists = ps_dists.repeat(n_other, 1)
        #
        #     zo = z[of]
        #
        #     distso = euclidean_dist(zo, z_proto)
        #
        #     inds = (distso < ps_dists).float()
        #
        #     other_loss = (inds * (distso / ps_dists)).sum() / inds.sum()


        return loc_loss, cls_loss, acc_val, samples.data.cpu().numpy()


class Memory(nn.Module):
    def __init__(self, memory_size, key_dim, top_k = 256, inverse_temp = 40, age_noise=8.0, margin = 0.1):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.top_k = min(top_k, memory_size)
        self.softmax_temperature = max(1.0, math.log(0.2 * top_k) / inverse_temp)
        self.age_noise = age_noise
        self.margin = margin

        # Parameters
        self.build()
        # self.query_proj = nn.Linear(key_dim, key_dim)

    # def forward(self, x):
    #     return self.query_proj(x)

    def build(self):
        self.keys = F.normalize(self.random_uniform((self.memory_size, self.key_dim), -0.001, 0.001, cuda=True), dim=1)
        self.keys_var = ag.Variable(self.keys, requires_grad=False)
        self.values = torch.zeros(self.memory_size, 1).long().cuda()
        self.age = torch.zeros(self.memory_size, 1).cuda()

    def predict(self, x):
        batch_size, dims = x.size()
        # query = F.normalize(self.query_proj(x), dim=1)
        query = F.normalize(x, dim=1)

        # Find the k-nearest neighbors of the query
        scores = torch.matmul(query, torch.t(self.keys_var))
        cosine_similarity, topk_indices_var = torch.topk(scores, self.top_k, dim=1)

        # softmax of cosine similarities - embedding
        softmax_score = F.softmax(self.softmax_temperature * cosine_similarity)

        # retrive memory values - prediction
        y_hat_indices = topk_indices_var.data[:, 0]
        y_hat = self.values[y_hat_indices]

        return y_hat, softmax_score

    # def query(self, x, y, predict=False):
    def query(self, loc_preds, cls_preds, loc_targets, cls_targets, predict=False):
        """
        Compute the nearest neighbor of the input queries.

        Arguments:
            x: A normalized matrix of queries of size (batch_size x key_dim)
            y: A matrix of correct labels (batch_size x 1)
        Returns:
            y_hat, A (batch-size x 1) matrix
		        - the nearest neighbor to the query in memory_size
            softmax_score, A (batch_size x 1) matrix
		        - A normalized score measuring the similarity between query and nearest neighbor
            loss - average loss for memory module
        """
        y, _ = cls_targets.max(1)
        pos = cls_targets > 0

        num_pos = pos.float().sum(1) # the number of gt anchors for the class/es we interested in for a single input image

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # mask 'other' and 'ignore' boxes out to not affect
        masked_loc_preds = loc_preds[mask].view(-1, 4)
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        tt = masked_loc_preds.max() # TODO why does the preds.max value jump from a decimal to into the 100's
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        loc_loss = loc_loss/num_pos.sum()
        # loc_loss.data[loc_loss.data == float("inf")] = 0

        pos_neg = cls_targets > -1  # mask out 'ignore' boxes to not affect

        # mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        maskb = pos.unsqueeze(2).expand_as(cls_preds)  # get a 0,1 mask to get the interesting anchors
        # samples = (cls_preds*mask.float()).data.sum(1)  # multiply mask by input then sum per image x
        samples = (cls_preds*maskb.float()).sum(1)  # multiply mask by input then sum per image x
        samples = samples/num_pos.float().unsqueeze(1).expand_as(samples)  # lastly divide to get the average emb per x

        batch_size, dims = samples.size()
        # query = F.normalize(self.query_proj(samples), dim=1)
        query = F.normalize(samples, dim=1)
        #query = F.normalize(torch.matmul(x, self.query_proj), dim=1)

        # Find the k-nearest neighbors of the query
        scores = torch.matmul(query, torch.t(self.keys_var))
        cosine_similarity, topk_indices_var = torch.topk(scores, self.top_k, dim=1)

        # softmax of cosine similarities - embedding
        softmax_score = F.softmax(self.softmax_temperature * cosine_similarity)

        # retrive memory values - prediction
        topk_indices = topk_indices_var.detach().data
        y_hat_indices = topk_indices[:, 0]
        y_hat = self.values[y_hat_indices]

        cls_loss = None
        if not predict:
            # Loss Function
            # topk_indices = (batch_size x topk)
            # topk_values =  (batch_size x topk x value_size)

            # collect the memory values corresponding to the topk scores
            batch_size, topk_size = topk_indices.size()
            flat_topk = self.flatten(topk_indices)
            flat_topk_values = self.values[topk_indices]
            topk_values = flat_topk_values.resize_(batch_size, topk_size)

            correct_mask = torch.eq(topk_values, torch.unsqueeze(y.data, dim=1)).float()
            correct_mask_var = ag.Variable(correct_mask, requires_grad=False)

            pos_score, pos_idx = torch.topk(torch.mul(cosine_similarity, correct_mask_var), 1, dim=1)
            neg_score, neg_idx = torch.topk(torch.mul(cosine_similarity, 1-correct_mask_var), 1, dim=1)

            # zero-out correct scores if there are no correct values in topk values
            mask = 1.0 - torch.eq(torch.sum(correct_mask_var, dim=1), 0.0).float()
            pos_score = torch.mul(pos_score, torch.unsqueeze(mask, dim=1))

            #print(pos_score, neg_score)
            cls_loss = self.MemoryLoss(pos_score, neg_score, self.margin)

        # Update memory
        self.update(query, y, y_hat, y_hat_indices)

        return y, y_hat, softmax_score, cls_loss, loc_loss, query


    def update(self, query, y, y_hat, y_hat_indices):
        batch_size, dims = query.size()

        # 1) Untouched: Increment memory by 1
        self.age += 1

        # Divide batch by correctness
        result = torch.squeeze(torch.eq(y_hat, torch.unsqueeze(y.data, dim=1))).float()
        incorrect_examples = torch.squeeze(torch.nonzero(1-result))
        correct_examples = torch.squeeze(torch.nonzero(result))

        incorrect = len(incorrect_examples.size()) > 0
        correct = len(correct_examples.size()) > 0

        # 2) Correct: if V[n1] = v
        # Update Key k[n1] <- normalize(q + K[n1]), Reset Age A[n1] <- 0
        if correct:
            correct_indices = y_hat_indices[correct_examples]
            correct_keys = self.keys[correct_indices]
            correct_query = query.data[correct_examples]

            new_correct_keys = F.normalize(correct_keys + correct_query, dim=1)
            self.keys[correct_indices] = new_correct_keys
            self.age[correct_indices] = 0

        # 3) Incorrect: if V[n1] != v
        # Select item with oldest age, Add random offset - n' = argmax_i(A[i]) + r_i
        # K[n'] <- q, V[n'] <- v, A[n'] <- 0
        if incorrect:
            incorrect_size = incorrect_examples.size()[0]
            incorrect_query = query.data[incorrect_examples]
            incorrect_values = y.data[incorrect_examples]

            age_with_noise = self.age.cuda() + self.random_uniform((self.memory_size, 1), -self.age_noise, self.age_noise, cuda=True)
            topk_values, topk_indices = torch.topk(age_with_noise, incorrect_size, dim=0)
            oldest_indices = torch.squeeze(topk_indices)

            self.keys[oldest_indices] = incorrect_query
            self.values[oldest_indices] = incorrect_values
            self.age[oldest_indices] = 0

    def where(self, cond, x_1, x_2):
        return (cond * x_1) + ((1 - cond) * x_2)

    def random_uniform(self, shape, low, high, cuda):
        x = torch.rand(*shape)
        result_cpu = (high - low) * x + low
        if cuda:
            return result_cpu.cuda()
        else:
            return result_cpu

    def multiply(self, x):
        return functools.reduce(lambda x, y: x * y, x, 1)

    def flatten(self, x):
        """ Flatten matrix into a vector """
        count = self.multiply(x.size())
        return x.resize_(count)

    def index(self, batch_size, x):
        idx = torch.arange(0, batch_size).long()
        idx = torch.unsqueeze(idx, -1)
        return torch.cat((idx, x), dim=1)

    def MemoryLoss(self, positive, negative, margin):
        """
            Calculate Average Memory Loss Function
            positive - positive cosine similarity
            negative - negative cosine similarity
            margin
        """
        assert (positive.size() == negative.size())
        dist_hinge = torch.clamp(negative - positive + margin, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss
