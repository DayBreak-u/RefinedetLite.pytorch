# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco_refinedet as cfg
from ..box_utils import match, log_sum_exp, refine_match ,refine_ioumatch
from ..box_utils import match_ious, bbox_overlaps_iou, bbox_overlaps_giou, bbox_overlaps_diou, bbox_overlaps_ciou, decode


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        print(self.gamma)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class IouLoss(nn.Module):

    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss


class RefineDetMultiBoxLoss(nn.Module):
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
                 use_gpu=True, theta=0.01, use_ARM=False , loss_name = 'SmoothL1'):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.theta = theta
        self.use_ARM = use_ARM
        self.loss_name = loss_name
        self.focalloss = FocalLoss(self.num_classes, gamma=2, size_average=False)
        self.gious = IouLoss(pred_mode='Center', size_sum=True, variances=self.variance, losstype=self.loss_name)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors = predictions
        # print(arm_loc_data.size(), arm_conf_data.size(),
        #      odm_loc_data.size(), odm_conf_data.size(), priors.size())
        #input()
        if self.use_ARM:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        #print(loc_data.size(), conf_data.size(), priors.size())

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if num_classes == 2:
                labels = labels >= 0
            defaults = priors.data
            if self.use_ARM:
                if self.loss_name == 'SmoothL1':
                    refine_match(self.threshold, truths, defaults, self.variance, labels,
                        loc_t, conf_t, idx, arm_loc_data[idx].data)
                else:
                    refine_ioumatch(self.threshold, truths, defaults, self.variance, labels,
                                 loc_t, conf_t, idx, arm_loc_data[idx].data)
            else:
                if self.loss_name == 'SmoothL1':
                    refine_match(self.threshold, truths, defaults, self.variance, labels,
                        loc_t, conf_t, idx)
                else:
                    refine_ioumatch(self.threshold, truths, defaults, self.variance, labels,
                                 loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        #loc_t = Variable(loc_t, requires_grad=False)
        #conf_t = Variable(conf_t, requires_grad=False)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        #print(loc_t.size(), conf_t.size())

        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:,:,1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.data] = 0
        else:
            pos = conf_t > 0
        #print(pos.size())
        #num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        if self.loss_name == "SmoothL1":
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        else:
            giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
            loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print(loss_c.size())

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        #print(num_pos.size(), num_neg.size(), neg.size())

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(pos_idx.size(), neg_idx.size(), conf_p.size(), targets_weighted.size())
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        #N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        #print(N, loss_l, loss_c)
        return loss_l, loss_c
