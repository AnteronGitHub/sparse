import torch

def compress_with_pruneFilter(pred, prune_filter, budget, serverFlag = False):
    if serverFlag:
        mask = prune_filter
    else:
        mask = torch.square(torch.sigmoid(prune_filter.squeeze()))
    topk = torch.topk(mask, budget)
    compressedPred = torch.index_select(pred, 1, topk.indices.sort().values)

    return compressedPred, mask

def decompress_with_pruneFilter(pred, mask, budget, device):
    a = torch.mul(mask.repeat([128,1]).t(), torch.eye(128).to(device))
    b = a.index_select(1, mask.topk(budget).indices.sort().values)
    b = torch.where(b>0.0, 1.0, 0.0).to(device)
    decompressed_pred = torch.einsum('ij,bjlm->bilm', b, pred)

    return decompressed_pred

def prune_loss_fn(loss_fn, pred, y, prune_filter, budget, delta = 0.1, epsilon=1000):
    prune_filter_squeezed = prune_filter.squeeze()
    prune_filter_control_1 = torch.exp( delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
    prune_filter_control_2 = torch.exp(-delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
    prune_filter_control = prune_filter_control_1 + prune_filter_control_2
    entropyLoss = loss_fn(pred,y)
    diff = entropyLoss + epsilon * prune_filter_control
    return diff, entropyLoss

