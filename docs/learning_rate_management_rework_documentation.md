# Learning Rate Management Rework Documentation

## Overview

The current training configuration uses `CosineAnnealingLR` uniformly across all curriculum phases. Although cosine annealing is a strong general-purpose schedule, the present implementation introduces severe optimization starvation due to:

- Aggressive decay toward near-zero learning rates.
- Short curriculum phase durations.
- Phase transitions with newly unfrozen parameters.
- Epoch-level scheduler stepping instead of optimizer-step scheduling.
- Uniform learning rate treatment across heterogeneous parameter groups.
- Persistent optimizer state across frozen/unfrozen transitions.

The result is insufficient optimization energy during the most critical adaptation phases, especially after unfreezing gaze-related and cross-view fusion modules.

This document proposes a complete restructuring of learning rate management for the staged training curriculum.

---

# Current Problems

## 1. Cosine Annealing Decays Too Aggressively

The current scheduler configuration:

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=phase_end - phase_start + 1,
)
```

causes the learning rate to decay nearly to zero inside each curriculum phase.

Example behavior:

| Phase | Initial LR | Final Effective LR |
|---|---|---|
| P1 | 1e-3 | ~0 |
| P2 | 3e-4 | ~0 |
| P3 | 5e-5 | ~0 |

This is especially harmful because:

- Phase durations are relatively short.
- Newly unfrozen modules require aggressive adaptation.
- Later curriculum stages are adaptation-heavy rather than convergence-only.

The optimizer spends too much time operating at ineffective learning rates.

---

## 2. Newly Unfrozen Modules Receive Decayed Learning Rates Immediately

During staged training:

- P1 trains foundational geometry.
- P2 introduces gaze adaptation.
- P3 introduces multi-view fusion.

However, each phase starts with a scheduler already configured to decay rapidly.

This creates the following issue:

1. New modules are unfrozen.
2. Optimization begins.
3. Learning rate collapses before meaningful adaptation occurs.

The model never receives sufficient optimization energy to fully adapt the newly trainable layers.

---

## 3. Epoch-Level Scheduler Stepping Is Too Coarse

Current behavior:

```python
scheduler.step()
```

is executed once per epoch.

This introduces several problems:

- Poor schedule granularity.
- Abrupt LR transitions.
- Reduced effectiveness of cosine schedules.
- Poor compatibility with gradient accumulation.
- Poor compatibility with distributed training.

Modern learning rate schedules should generally step per optimizer update.

---

## 4. Uniform Learning Rate Across All Parameter Groups

The current optimizer applies a single learning rate to:

- Pretrained backbone layers.
- Geometry heads.
- Gaze estimation branches.
- Cross-view attention modules.
- Fusion layers.

This is suboptimal because different modules have different optimization requirements.

Examples:

- Pretrained CNN stems require conservative updates.
- Newly initialized attention layers require aggressive updates.
- Fusion layers often need higher adaptation capacity.

Uniform LR scaling slows adaptation and destabilizes optimization.

---

## 5. Optimizer State Persists Across Phase Transitions

The current pipeline freezes modules using:

```python
requires_grad_(False)
eval()
```

However, optimizer parameter groups and Adam moments are preserved.

This introduces stale momentum statistics when:

- Modules are unfrozen later.
- Loss geometry changes.
- Optimization objectives shift.

As a result:

- Previously frozen modules inherit stale optimizer dynamics.
- Adaptation becomes unstable.
- Effective learning rates become distorted.

---

# Recommended Learning Rate Strategy

The learning rate schedule should be phase-aware.

Each curriculum phase represents a different optimization regime and therefore requires a dedicated strategy.

---

# Phase 1: Geometry Bootstrap

## Objective

Phase 1 is responsible for:

- Geometric representation formation.
- Stable landmark initialization.
- Initial low-level feature learning.
- Building optimization stability.

This phase requires:

- High optimization energy.
- Stable gradient flow.
- Strong parameter mobility.

Aggressive decay is undesirable during this phase.

---

## Recommended Strategy

### Preferred Option: Constant Learning Rate

Use a constant learning rate throughout the entire phase.

Recommended configuration:

```python
lr = 1e-3
```

No scheduler should be used.

Implementation:

```python
optimizer = AdamW(trainable_params, lr=1e-3)
scheduler = None
```

This preserves optimization energy across the entire geometry bootstrap stage.

---

## Alternative Option: Cosine Annealing With Minimum Floor

If a scheduler is still desired, cosine annealing must never decay near zero.

Recommended implementation:

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=3e-4,
)
```

This changes the trajectory from:

```text
1e-3 -> ~0
```

to:

```text
1e-3 -> 3e-4
```

This preserves meaningful optimization capacity throughout the phase.

---

# Phase 2: Gaze Adaptation

## Objective

Phase 2 introduces:

- Gaze-specific branches.
- Iris geometry.
- Visual axis estimation.
- Higher-level representation adaptation.

This is not a convergence phase.

It is a high-plasticity adaptation phase.

The optimizer must retain sufficient mobility throughout the entire stage.

---

## Why Cosine Annealing Fails Here

The current configuration:

```text
3e-4 -> ~0
```

causes:

- Early loss stabilization.
- Reduced adaptation capacity.
- Weak branch integration.
- Incomplete manifold restructuring.

This phase benefits far more from cyclical exploration than monotonic decay.

---

## Recommended Strategy: OneCycleLR

Recommended implementation:

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-4,
    total_steps=total_train_steps,
    pct_start=0.25,
    div_factor=10,
    final_div_factor=20,
)
```

Resulting trajectory:

```text
5e-5 -> 5e-4 -> 2.5e-5
```

Advantages:

- Strong early exploration.
- Sustained optimization energy.
- Better adaptation for newly unfrozen modules.
- Smoother stabilization.
- Improved convergence robustness.

---

## Scheduler Stepping

`OneCycleLR` must step per optimizer update.

Correct implementation:

```python
loss.backward()

if should_step:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

Do not step once per epoch.

---

# Phase 3: Multi-View Fusion Stabilization

## Objective

Phase 3 introduces:

- Cross-view attention.
- Fusion modules.
- Multi-view consistency.
- Reprojection alignment.
- Higher-order geometric refinement.

This phase already uses a substantially reduced base LR.

Current configuration:

```python
lr = 5e-5
```

Applying aggressive cosine decay on top of this creates double attenuation.

The effective learning rate frequently becomes:

```text
1e-6 to 1e-7
```

which is insufficient for meaningful optimization.

---

# Recommended Strategy

## Preferred Option: Constant LR

Recommended implementation:

```python
optimizer = AdamW(trainable_params, lr=5e-5)
scheduler = None
```

This maintains stable adaptation throughout fusion training.

---

## Alternative Option: Cosine With Floor

If cosine annealing is preferred:

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=2e-5,
)
```

This changes:

```text
5e-5 -> ~0
```

into:

```text
5e-5 -> 2e-5
```

which preserves useful optimization capacity.

---

# Parameter Group Learning Rates

Different modules should use different learning rates.

Recommended structure:

```python
optimizer = AdamW([
    {
        'params': stem.parameters(),
        'lr': base_lr * 0.1,
    },
    {
        'params': landmark_head.parameters(),
        'lr': base_lr * 0.2,
    },
    {
        'params': gaze_branch.parameters(),
        'lr': base_lr,
    },
    {
        'params': cross_view_attention.parameters(),
        'lr': base_lr * 1.5,
    },
])
```

---

## Recommended Scaling

| Module Type | LR Multiplier |
|---|---|
| Pretrained backbone | 0.1x |
| Landmark heads | 0.2x |
| Gaze branch | 1.0x |
| Cross-view attention | 1.5x |
| Newly initialized fusion layers | 1.5x to 2.0x |

This improves:

- Adaptation speed.
- Stability.
- Representation preservation.
- Fusion convergence.

---

# Optimizer Reinitialization Across Phases

When phase transitions occur, the optimizer should be rebuilt.

Do not keep stale optimizer state for frozen modules.

Recommended implementation:

```python
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = AdamW(
    trainable_params,
    lr=current_phase_lr,
)
```

This ensures:

- Fresh optimizer statistics.
- Stable adaptation.
- Correct momentum behavior.
- Cleaner curriculum transitions.

---

# Scheduler Stepping Best Practices

## Incorrect

```python
for epoch in epochs:
    train()
    validate()
    scheduler.step()
```

---

## Correct

```python
for batch in loader:
    loss.backward()

    if should_step:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

The scheduler should follow optimizer updates, not epoch boundaries.

---

# Recommended Final Configuration

## Phase 1

### Preferred

```python
lr = 1e-3
scheduler = None
```

### Alternative

```python
CosineAnnealingLR(
    eta_min=3e-4
)
```

---

## Phase 2

```python
OneCycleLR(
    max_lr=5e-4,
    div_factor=10,
    final_div_factor=20,
)
```

Per-step scheduler updates required.

---

## Phase 3

### Preferred

```python
lr = 5e-5
scheduler = None
```

### Alternative

```python
CosineAnnealingLR(
    eta_min=2e-5
)
```

---

# Expected Improvements

The proposed redesign should improve:

- Optimization stability.
- Representation adaptation.
- Gaze branch convergence.
- Multi-view fusion learning.
- Gradient utilization.
- Late-phase optimization energy.
- Cross-view alignment quality.
- Overall convergence speed.

The largest gains are expected from:

1. Preventing near-zero LR collapse.
2. Replacing P2 cosine annealing with OneCycleLR.
3. Using constant LR during P1 and optionally P3.
4. Introducing parameter-group scaling.
5. Rebuilding optimizer state between curriculum phases.

---

# Conclusion

The current training pipeline suffers from curriculum-aware optimization mismatch rather than simple scheduler misconfiguration.

The model architecture uses staged representation learning with progressive unfreezing, but the optimizer configuration behaves as though all phases are homogeneous convergence stages.

A phase-aware optimization strategy is required.

The recommended redesign preserves optimization energy during adaptation-heavy stages while still allowing controlled stabilization during later convergence.

The most important correction is eliminating cosine decay toward near-zero learning rates during short curriculum phases.

Constant learning rate training in P1 is strongly recommended.

