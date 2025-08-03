import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Iterable, Optional
import random

class PCGrad:
    """
    Projected Conflicting Gradient (PCGrad) Optimizer wrapper for multitask learning.

    Arguments:
        optimizer: Standard PyTorch optimizer (e.g. Adam, SGD).
        eps: Small value for numerical stability in projection.
        grad_clip: (Optional) Max norm for gradient clipping (recommended for stability).
    """
    def __init__(self, optimizer: Optimizer, eps: float = 1e-12, grad_clip: Optional[float] = 10.0):
        self._optim = optimizer
        self.eps = eps
        self.grad_clip = grad_clip

    def zero_grad(self):
        self._optim.zero_grad()

    def _project_conflicts(self, grads: List[List[Tensor]]):
        """
        In-place projects each task’s gradient onto the normal plane of any other
        task's gradient when a conflict is detected (dot product < 0).
        Randomly shuffles task order for unbiased projection (SOTA tip).
        """
        num_tasks = len(grads)
        order = list(range(num_tasks))
        random.shuffle(order)  # Shuffle for less bias

        for i in order:
            for j in order:
                if i == j:
                    continue
                dot_ij = sum((g_i * g_j).sum() for g_i, g_j in zip(grads[i], grads[j]))
                if dot_ij < 0:
                    denom = sum((g_j**2).sum() for g_j in grads[j]) + self.eps
                    proj = dot_ij / denom
                    for k in range(len(grads[i])):
                        grads[i][k] -= proj * grads[j][k]

    def pc_backward(
        self,
        losses: List[Tensor],
        params: Iterable[Tensor],
        retain_graph: bool = True,
        verbose: bool = True,
    ):
        """
        Computes de-conflicted gradients for multitask loss.

        Arguments:
            losses: List of per-task scalar losses.
            params: Iterable of model parameters (usually shared parameters).
            retain_graph: Whether to retain the computation graph (should be True for multitask).
            verbose: If True, will print warnings when NaNs are detected.
        """
        grads_per_task: List[List[Tensor]] = []

        for idx, loss in enumerate(losses):
            self.zero_grad()
            loss.backward(retain_graph=retain_graph, create_graph=False)
            task_grads = []
            for p in params:
                if p.grad is not None:
                    g = p.grad.detach().clone()
                else:
                    g = torch.zeros_like(p)
                task_grads.append(g)
            # Check for NaN or Inf in any grad
            if any(torch.isnan(g).any() or torch.isinf(g).any() for g in task_grads):
                if verbose:
                    print(f"NaN or Inf detected in grads for task {idx}. Skipping this batch.")
                return False  # You might want to skip the batch
            grads_per_task.append(task_grads)

        # Project away conflicts
        self._project_conflicts(grads_per_task)

        # Average and write back the grads to each parameter
        self.zero_grad()
        num_tasks = len(grads_per_task)
        for idx, p in enumerate(params):
            stacked = torch.stack([grads_per_task[t][idx] for t in range(num_tasks)], dim=0)
            mean_grad = stacked.mean(dim=0)
            if torch.isnan(mean_grad).any() or torch.isinf(mean_grad).any():
                if verbose:
                    print(f"NaN or Inf in mean gradient for param {idx}. Skipping update.")
                return False
            p.grad = mean_grad

        # Optional: Gradient clipping for stability (set grad_clip=None to disable)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

        return True

    def step(self):
        self._optim.step()
