import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2_distance(a, b):
    return torch.norm(a.unsqueeze(1) - b.unsqueeze(0), dim=2)

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    """Calculate similarity between two sets of vectors."""
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    return h1 @ h2.t()

def pairwise_distances(anchor, sample, pos_mask, neg_mask):
    sample = sample.to(device)
    anchor = anchor.to(device)
    sample = sample.squeeze(0)

    positive_samples = sample[pos_mask[0].bool()]
    negative_samples = sample[neg_mask[0].bool()]

    if positive_samples.size(0) == 0:
        raise ValueError("No positive samples found.")

    if anchor.shape[0] > 1 and positive_samples.size(0) > 0:
        sim = _similarity(anchor, positive_samples)
        max_sim_idx = torch.argmax(sim.max(dim=1).values)
        anchor = anchor[max_sim_idx].unsqueeze(0)

    positive_distances = l2_distance(anchor, positive_samples)
    negative_distances = l2_distance(anchor, negative_samples)
    return positive_distances, negative_distances


class InfoNCE(nn.Module):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        if len(anchor) > 1:
            sim, _ = torch.max(sim, dim=0)

        exp_sim = torch.exp(sim)
        pos_loss = torch.log((exp_sim * pos_mask).sum(dim=1))
        neg_loss = torch.log((exp_sim * (pos_mask + neg_mask)).sum(dim=1))

        loss = pos_loss - neg_loss
        return -loss.mean()

class Triplet(nn.Module):
    def __init__(self, margin, topk):
        super(Triplet, self).__init__()
        self.margin = margin
        self.topk = topk

    def forward(self, positive_distances, negative_distances):
        num_positive = positive_distances.size(1)
        num_negative = negative_distances.size(1)
        k_ = min(self.topk, max(0, min(num_positive, num_negative)))

        if k_ == 0:
            return F.relu(self.margin).mean()

        hardest_positive_dist = torch.topk(positive_distances, k=k_, dim=1)[0].mean(dim=1)
        hardest_negative_dist = torch.topk(negative_distances, k=k_, dim=1, largest=False)[0].mean(dim=1)
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return triplet_loss.mean()
