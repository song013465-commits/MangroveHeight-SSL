import torch
from torch import nn
from torch.nn import functional as F


class SigLoss(nn.Module):
    
    def __init__(self, valid_mask=True, loss_weight=10.0, max_depth=70,
                 warm_up=False, warm_iter=100, loss_name="sigloss", lambda_value=0.85):
        super().__init__()
        self.valid_mask, self.loss_weight, self.max_depth = valid_mask, loss_weight, max_depth
        self.warm_up, self.warm_iter, self.loss_name = warm_up, warm_iter, loss_name
        if not 0 <= lambda_value < 1:
            raise ValueError("SigLoss lambda must be in [0, 1)")
        self.lambda_value = float(lambda_value)
        self.register_buffer("warm_up_counter", torch.zeros((), dtype=torch.long))

    def forward(self, prediction, target, sample_mask=None, per_tile=False):
        mask = torch.isfinite(target) & (target > 0)
        if self.max_depth is not None:
            mask &= target <= self.max_depth
        if sample_mask is not None:
            mask &= sample_mask
        if per_tile:
            groups = zip(prediction.float(), target.float(), mask)
        else:
            groups = [(prediction.float(), target.float(), mask)]
        warming = self.training and self.warm_up and self.warm_up_counter.item() < self.warm_iter
        losses = []
        for pred, ref, valid in groups:
            p, t = pred[valid], ref[valid]
            if not p.numel():
                continue
            g = torch.log(p.clamp_min(0) + 0.001) - torch.log(t + 0.001)
            value = (1.0 - self.lambda_value) * g.mean().square()
            if not warming:
                value = value + g.var(unbiased=False)
            losses.append(self.loss_weight * (torch.sqrt(value + 1e-12) - 1e-6))
        if not losses:
            return prediction.sum() * 0.0
        if self.training:
            self.warm_up_counter.add_(1)
        return torch.stack(losses).mean()


def soft_histogram(values, bins=70, maximum=70.0):
    
    position = (values.float().reshape(-1) / maximum * bins - 0.5).clamp(0, bins - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(bins - 1)
    fraction = position - lower.float()
    counts = values.new_zeros(bins, dtype=torch.float32)
    counts = counts.scatter_add(0, lower, 1 - fraction)
    counts = counts.scatter_add(0, upper, fraction)
    return (counts + 1e-6) / (counts.sum() + bins * 1e-6)


class JointCanopyLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0), max_depth=70.0, bins=70,
                 warm_iter=100, sig_alpha=10.0, sig_lambda=0.85, sig_warm_up=False):
        super().__init__()
        self.weights, self.max_depth, self.bins = tuple(weights), max_depth, bins
        self.sig = SigLoss(max_depth=max_depth, loss_weight=sig_alpha,
                           lambda_value=sig_lambda, warm_up=sig_warm_up, warm_iter=warm_iter)

    def forward(self, outputs, target, sample_mask, input_mask):
        prediction = outputs["prediction"].float()
        p, t = prediction[sample_mask], target.float()[sample_mask]
        # Compute within each tile, then average tiles with valid observations.
        loss0 = self.sig(prediction, target.float(), sample_mask, per_tile=True)
        if not p.numel():
            loss1 = loss2 = prediction.sum() * 0.0
        else:
            loss1 = F.mse_loss(p, t)
            distribution = soft_histogram(p, self.bins, self.max_depth)

            with torch.no_grad():
                target_counts = torch.histc(t.clamp(0, self.max_depth), bins=self.bins, min=0, max=self.max_depth)
                target_distribution = target_counts / target_counts.sum().clamp_min(1)
            loss2 = -(target_distribution * distribution.log()).sum()
        total = self.weights[0] * loss0 + self.weights[1] * loss1 + self.weights[2] * loss2
        return total, {"sig_loss": loss0, "mse_loss": loss1, "hist_loss": loss2}
