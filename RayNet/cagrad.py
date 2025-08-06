# cagrad.py

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Iterable

class CAGrad:
    """
    CAGrad (Constrained Optimization for Multi-Task Learning)
    https://arxiv.org/abs/2103.06824
    """
    def __init__(self, optimizer: Optimizer, alpha: float = 0.5, rescale: bool = True):
        self._optim = optimizer
        self.alpha = alpha  # Interpolation between equal weighting and optimal solution
        self.rescale = rescale

    def zero_grad(self):
        self._optim.zero_grad()

    def _compute_grads(self, losses: List[Tensor], params: Iterable[Tensor], retain_graph: bool = True):
        grads_per_task = []
        for loss in losses:
            self.zero_grad()
            loss.backward(retain_graph=retain_graph, create_graph=False)
            grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            grads_per_task.append(torch.cat([g.flatten() for g in grads]))
        grads_per_task = torch.stack(grads_per_task)  # [num_tasks, total_params]
        return grads_per_task

    def cagrad_backward(self, losses: List[Tensor], params: Iterable[Tensor], retain_graph: bool = True):
        """
        Backward step with CAGrad. Sets .grad fields of params to the CAGrad direction.
        """
        grads_per_task = self._compute_grads(losses, params, retain_graph=retain_graph)  # [num_tasks, total_params]
        GG = grads_per_task @ grads_per_task.t()  # [num_tasks, num_tasks]
        g0 = grads_per_task.mean(dim=0)          # [total_params]

        # Find the optimal combination (solves a QP)
        # minimize 0.5 x^T GG x, subject to x >= 0, sum(x) = 1
        try:
            import cvxpy as cp
            x = cp.Variable(grads_per_task.size(0))
            objective = 0.5 * cp.quad_form(x, GG.cpu().numpy())
            constraints = [x >= 0, cp.sum(x) == 1]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP)
            w_star = torch.tensor(x.value, dtype=torch.float32, device=grads_per_task.device)
        except ImportError:
            # Fallback: equal weighting
            w_star = torch.ones(grads_per_task.size(0), device=grads_per_task.device) / grads_per_task.size(0)

        # CAGrad: Convex combine with mean grad for stability
        merged_grad = self.alpha * g0 + (1 - self.alpha) * (w_star @ grads_per_task)
        if self.rescale:
            merged_grad = merged_grad / (merged_grad.norm() + 1e-8) * (g0.norm() + 1e-8)

        # Write back merged_grad into .grad fields of params
        pointer = 0
        for p in params:
            numel = p.numel()
            if p.requires_grad:
                p.grad = merged_grad[pointer:pointer+numel].view_as(p).clone()
                pointer += numel

    def step(self):
        self._optim.step()
