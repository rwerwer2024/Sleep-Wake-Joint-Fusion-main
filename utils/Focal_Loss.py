# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

def LabelSmooth(onehot, num_classes, delta=0.01):
    return onehot * (1 - delta) + delta * 1.0 / num_classes

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, reduction='mean'):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi) 可以单独拎出来用,替代cross_entropy
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C]
        :param labels:  实际类别. size:[B,N]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        alpha = 0.25
        targets = one_hot_embedding(labels.data.cpu(), 4).to(preds.device)  # [N,21]
        # targets = LabelSmooth(targets, preds.shape[1])
        targets = targets.view(-1, targets.size(-1))
        preds = preds.view(-1, preds.size(-1))

        # pos_mask = (labels > 0)
        # neg_mask = (labels == 0)
        # mask = pos_mask | neg_mask
        # targets = torch.Tensor(torch.ones(preds.shape)*(-1)).to(preds.device)
        # targets[mask,:]=0
        # targets[pos_mask,labels[pos_mask]]=1

        alpha_factor = torch.Tensor(torch.ones(targets.shape) * alpha).to(preds.device)

        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        # classes = torch.argmax(preds_softmax, dim=-1)
        alpha_factor = torch.where(targets==1, alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(targets==1, 1. - preds_softmax, preds_softmax)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        bce = -torch.where(targets==1, torch.log(preds_softmax), torch.log(1.0 - preds_softmax))
        # bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

        bce = -(targets * torch.log(preds_softmax) + (1. - targets) * torch.log(1. - preds_softmax))
        loss = focal_weight * bce

        # loss = torch.where(torch.ne(targets, -1.0), loss, torch.zeros(loss.shape).to(preds.device))

        # loss = loss.sum()/pos_mask.sum()
        # loss = loss.sum()
        # if self.reduction== 'mean':
        #     loss = loss.mean()
        # elif self.reduction== 'sum':
        #     loss = loss.sum()
        return loss


    def forward_org(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C]
        :param labels:  实际类别. size:[B,N]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.reduction== 'mean':
            loss = loss.mean()
        elif self.reduction== 'sum':
            loss = loss.sum()
        return loss
