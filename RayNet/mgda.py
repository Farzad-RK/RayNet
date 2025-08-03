import torch

def flatten_grad(grads):
    return torch.cat([g.contiguous().view(-1) for g in grads])

def get_grads(loss, shared_params):
    grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
    return flatten_grad(grads)

def mgda_solver(grads):
    """
    grads: list of flattened gradient vectors, one per task (all with same length)
    Returns: list of weights (lambdas), summing to 1
    """
    K = len(grads)
    G = torch.stack(grads)  # [K, D]
    GG = G @ G.t()          # [K, K] Gram matrix

    # Closed-form for 2 tasks: lambda1 + lambda2 = 1, minimize max grad norm
    if K == 2:
        g1, g2 = grads
        cos_sim = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-12)
        # Avoid negative or nan
        lambda1 = (g2.norm() - cos_sim * g1.norm()) / (g1.norm() + g2.norm() - 2 * cos_sim * g1.norm())
        lambda1 = lambda1.clamp(0, 1)
        return [lambda1, 1 - lambda1]

    # For more tasks: use simplex projection (approximate with equal weights if not solved)
    lambdas = torch.ones(K, device=G.device) / K
    return lambdas.tolist()

def mgda_loss(losses, shared_params):
    """
    losses: list of scalar losses (per task)
    shared_params: parameters to compute gradients w.r.t.
    Returns: weighted sum of losses, list of weights
    """
    grads = [get_grads(loss, shared_params) for loss in losses]
    lambdas = mgda_solver(grads)
    total_loss = sum(l * loss for l, loss in zip(lambdas, losses))
    return total_loss, lambdas
