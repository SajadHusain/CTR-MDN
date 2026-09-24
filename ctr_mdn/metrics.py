import math

import numpy as np
import torch

from .data import denorm_y, norm_x, norm_y
from .model import CTRForwardMDN, mdn_nll

ROBOT_LENGTH = 210.0  # mm

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _unit(q):
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def pose_errors(y_true, y_pred):
    e_pos = torch.linalg.norm(y_true[:, :3] - y_pred[:, :3], dim=1)
    dot = (_unit(y_true[:, 3:7]) * _unit(y_pred[:, 3:7])).sum(1).abs().clamp(0, 1)
    return e_pos, 2 * torch.acos(dot)


@torch.no_grad()
def evaluate(model, loader, stats, device, point_estimate="map"):
    """NLL (normalised targets) and tip pose errors in mm / deg."""
    model.eval()
    nll, n = 0.0, 0
    pos, ang, top1, keff = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, mu, U = model(norm_x(x, stats))
        nll += mdn_nll(logits, mu, U, norm_y(y, stats)).item() * len(x)
        n += len(x)

        if point_estimate == "mean":
            y_hat = (torch.softmax(logits, -1).unsqueeze(-1) * mu).sum(1)
        else:  # mean of the most probable component
            y_hat = mu[torch.arange(len(x), device=device), logits.argmax(-1)]
        y_hat = denorm_y(y_hat, stats)
        y_hat[:, 3:7] = _unit(y_hat[:, 3:7])

        ep, ea = pose_errors(y, y_hat)
        pos.append(ep.cpu())
        ang.append(ea.cpu())
        pi = torch.softmax(logits, -1)
        top1.append(pi.max(-1).values.cpu())
        keff.append(torch.exp(-(pi * pi.clamp_min(1e-12).log()).sum(-1)).cpu())

    pos = torch.cat(pos).double()
    ang = torch.cat(ang).double()
    return {
        "nll": nll / n,
        "pos_mean_mm": pos.mean().item(),
        "pos_median_mm": pos.median().item(),
        "pos_mean_pct": 100 * pos.mean().item() / ROBOT_LENGTH,
        "ang_mean_deg": math.degrees(ang.mean().item()),
        "ang_median_deg": math.degrees(ang.median().item()),
        "top1_weight": torch.cat(top1).mean().item(),
        "k_eff": torch.cat(keff).mean().item(),
        "n": n,
    }


@torch.no_grad()
def position_uncertainty(model, loader, stats, device):
    """Position errors and sqrt(trace(Sigma)) [mm] for the MAP component and for the full mixture."""
    model.eval()
    out = {"err_map": [], "err_mix": [], "u_map": [], "u_mix": []}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, mu, U = model(norm_x(x, stats))
        pi = torch.softmax(logits, -1)

        cov = CTRForwardMDN.covariance(U)[..., :3, :3]
        S = torch.diag_embed(stats["y_pos_std"]).expand_as(cov)
        cov = S @ cov @ S
        mu = denorm_y(mu, stats)[..., :3]

        b = torch.arange(len(x), device=device)
        k = logits.argmax(-1)
        mu_mix = (pi.unsqueeze(-1) * mu).sum(1)
        d = mu - mu_mix.unsqueeze(1)
        cov_mix = (pi[..., None, None] * (cov + d.unsqueeze(-1) @ d.unsqueeze(-2))).sum(1)

        def spread(C):
            return torch.diagonal(C, dim1=-2, dim2=-1).sum(-1).clamp_min(0).sqrt()

        out["err_map"].append(torch.linalg.norm(y[:, :3] - mu[b, k], dim=1).cpu())
        out["err_mix"].append(torch.linalg.norm(y[:, :3] - mu_mix, dim=1).cpu())
        out["u_map"].append(spread(cov[b, k]).cpu())
        out["u_mix"].append(spread(cov_mix).cpu())
    return {k: torch.cat(v).double().numpy() for k, v in out.items()}


def risk_coverage(err, unc, min_coverage=0.01):
    e = err[np.argsort(unc)]
    n = len(e)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    s = np.searchsorted(coverage, min_coverage)
    return coverage[s:], risk[s:], float(_trapz(risk[s:], coverage[s:]))
