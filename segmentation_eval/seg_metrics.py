### Functions modified from torch_geometric.utils.metrics

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add


def get_loss_fn(criteria):
    def loss_fn(outs, y):
        loss = criteria(outs, y)
        return loss.item()
    return loss_fn


def accuracy(pred, target, weights = None):
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    matches = (pred == target)
    
    if weights is None:
        return  matches.sum().item() / target.numel()
    else:
        matches = matches.float()
        matches *= weights
        return matches.sum().item() / weights.sum().item()
        

def intersection_and_union(pred, target, num_classes, weights=None, batch=None):
    r"""Computes intersection and union of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)
    
    i_result = pred & target
    u_result = pred | target
    
    if weights is not None:
        i_result = i_result.float()
        u_result = u_result.float()
        
        i_result *= weights.unsqueeze(1)
        u_result *= weights.unsqueeze(1)
    
    if batch is None:
        i = (i_result).sum(dim=0)
        u = (u_result).sum(dim=0)
    else:
        i = scatter_add(i_result, batch, dim=0)
        u = scatter_add(u_result, batch, dim=0)

    return i, u



def mean_iou(pred, target, num_classes, weights=None, batch=None):
    r"""Computes the mean intersection over union score of predictions.

    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.

    :rtype: :class:`Tensor`
    """
    i, u = intersection_and_union(pred, target, num_classes, weights, batch)
    iou = i.to(torch.float) / u.to(torch.float)
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou


# SMOOTH = 1e-6
#def mean_iou(outputs: torch.Tensor, labels: torch.Tensor):
#    outputs = torch.argmax(outputs, 1)
#    if len(outputs.shape) == 4:
#        outputs = outputs.squeeze(1)
#    intersection = (outputs & labels).float().sum((1, 2))
#    union = (outputs | labels).float().sum((1, 2))
#    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

#    return iou.mean().item()

