# gaze_vector_loss.py
import math
import torch
import torch.nn.functional as F

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def logC3_vmf(kappa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    log normalizer for vMF on S^2.
    C_3(k) = k / (4π sinh k)  ->  logC = log k - log(4π) - log(sinh k)
    Stable for large/small k using log1p.
    """
    k = kappa.clamp_min(0.0)
    # log sinh k = k + log(1 - exp(-2k)) - log 2
    two_k = (2.0 * k).clamp_max(50.0)  # avoid underflow in exp(-2k)
    log_sinh = k + torch.log1p((-torch.exp(-two_k)).clamp_min(-0.999999)) - math.log(2.0)
    return torch.log(k + eps) - math.log(4.0 * math.pi) - log_sinh

def vmf_nll(mu: torch.Tensor, kappa: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    vMF negative log-likelihood on S^2.
    mu:     [*,3] unit
    kappa:  [*,1] >= 0
    target: [*,3] unit
    """
    mu = _normalize(mu)
    target = _normalize(target)
    dot = (mu * target).sum(dim=-1, keepdim=True)                 # [*,1]
    return -(logC3_vmf(kappa) + kappa * dot).mean()

def angular_error_rad(d_pred: torch.Tensor, d_ref: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    d_pred = _normalize(d_pred).reshape(-1, 3)
    d_ref  = _normalize(d_ref).reshape(-1, 3)
    cos = (d_pred * d_ref).sum(-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos).mean()

@torch.no_grad()
def spherical_mean_kappa(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    κ-weighted spherical mean for directions.
    mu:    [B,N,3] (unit)
    kappa: [B,N,1]
    returns m: [B,3] (unit)
    """
    m = (kappa * mu).sum(dim=1)           # [B,3]
    return _normalize(m)

def multiview_gaze_vector_vmf_losses(pred: dict,
                                     gaze_vec_gt: torch.Tensor,
                                     w_cons: float = 0.2) -> dict:
    """
    vMF accuracy + spherical consistency (multi-view).
    pred:        {"mu":[B,N,3], "kappa":[B,N,1]}
    gaze_vec_gt: [B,N,3] (unit)
    returns:     {"accuracy": nll, "consistency": w_cons * spread}
    """
    mu    = _normalize(pred["mu"])                 # [B,N,3]
    kappa = pred["kappa"].clamp_min(0.0)           # [B,N,1]
    target = _normalize(gaze_vec_gt)               # [B,N,3]

    # vMF NLL for accuracy
    nll = vmf_nll(mu.reshape(-1,3), kappa.reshape(-1,1), target.reshape(-1,3))

    # spherical consistency around κ-weighted mean direction
    m = spherical_mean_kappa(mu, kappa)            # [B,3]
    m_rep = m.unsqueeze(1).expand_as(mu)           # [B,N,3]
    spread = angular_error_rad(mu, m_rep)          # mean angular deviation (radians)

    return {"accuracy": nll, "consistency": w_cons * spread}
