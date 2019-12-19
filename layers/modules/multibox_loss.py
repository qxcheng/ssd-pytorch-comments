# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu                            # True/False
        self.num_classes = num_classes                    # 21
        self.threshold = overlap_thresh                   # 0.5
        self.background_label = bkg_label                 # 0
        self.encode_target = encode_target                # False
        self.use_prior_for_matching = prior_for_matching  # True
        self.do_neg_mining = neg_mining                   # True
        self.negpos_ratio = neg_pos                       # 3
        self.neg_overlap = neg_overlap                    # 0.5
        self.variance = cfg['variance']                   # [0.1, 0.2]

    def forward(self, predictions, targets):
        # targets : [tensor([[x_min,y_min,x_max,y_max,label], targets2, ...]), batch2, ...]
        loc_data, conf_data, priors = predictions   # (1, 8732, 4)  (1, 8732, 21)  (8732, 4)
        num = loc_data.size(0)                      # batch=1
        priors = priors[:loc_data.size(1), :]       # (8732, 4)
        num_priors = (priors.size(0))               # 8732
        num_classes = self.num_classes              # 21

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)    # (1, 8732, 4)
        conf_t = torch.LongTensor(num, num_priors)  # (1, 8732)
        for idx in range(num):
            truths = targets[idx][:, :-1].data      # (num_tar, 4)
            labels = targets[idx][:, -1].data       # (num_tar)
            defaults = priors.data                  # (8732, 4)
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)                 # (1, 8732, 4)   每个预测框匹配到的真实坐标
        conf_t = Variable(conf_t, requires_grad=False)               # (1, 8732)      每个预测框匹配到的真实标签,0为背景

        pos = conf_t > 0                                             # (1, 8732)
        num_pos = pos.sum(dim=1, keepdim=True)                       # int:num_pos 12

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)       # (1, 8732, 4)
        loc_p = loc_data[pos_idx].view(-1, 4)                        # (num_pos, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)                           # (num_pos, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # float 27.9045

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)                            # (8732, 21)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # (8732, 1)

        # Hard Negative Mining
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])                           # add line  (1, 8732)
        loss_c[pos] = 0                                                              # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)                                                # (1, 8732)
        _, loss_idx = loss_c.sort(1, descending=True)                                # _, (1, 8732) 最大值的索引...
        _, idx_rank = loss_idx.sort(1)                                               # _, (1, 8732) 原数据按降序排序时的标签
        num_pos = pos.long().sum(1, keepdim=True)                                    # (1, 1)  tensor([[12]])
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)          # (1, 1)  tensor([[36]])
        neg = idx_rank < num_neg.expand_as(idx_rank)                                 # (1, 8732)    小于36的留下，即损失最大的36个值

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)                              # (1, 8732, 21)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)                              # (1, 8732, 21)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)       # (48, 21) 48=12+36
        targets_weighted = conf_t[(pos+neg).gt(0)]                                   # (48)
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)       # tensor(277.3131, grad_fn=...)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()                                                       # tensor(12)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
