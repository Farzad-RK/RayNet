"""
Hardware profiles and distributed-training entry points for RayNet.

This module centralises GPU/CPU hardware configuration (batch sizes, AMP
settings, gradient accumulation, etc.) so that train.py only has to pick a
named profile. Distributed training is handled via HuggingFace Accelerate
— the AMP path in train.py (autocast + GradScaler) is left untouched.

Supported scenarios
-------------------
Single-process (no distribution needed):
    python -m RayNet.train --profile t4      ...
    python -m RayNet.train --profile a100    ...

Kaggle dual-GPU (2× Tesla T4 on one node):
    accelerate launch --multi_gpu --num_processes 2 \\
        -m RayNet.train --profile kaggle_t4x2 ...

Two machines on the same network (one GPU each, NCCL over TCP):
    # On machine 0 (main, IP = $MAIN_IP):
    accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \\
        --machine_rank 0 --main_process_ip $MAIN_IP --main_process_port 29500 \\
        -m RayNet.train --profile multi_node_t4 ...
    # On machine 1:
    accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \\
        --machine_rank 1 --main_process_ip $MAIN_IP --main_process_port 29500 \\
        -m RayNet.train --profile multi_node_t4 ...

In the distributed profiles, `batch_size` is PER-PROCESS. Global effective
batch = batch_size * num_processes * grad_accum_steps.
"""

import torch


HARDWARE_PROFILES = {
    # ---- CPU / low-end GPU (testing, debugging) ----
    # v4: 224×224 input. Batch sizes ~4× larger than v3 (448×448).
    'default': {
        'batch_size': 504,          # 56 mv_groups × 9 views
        'mv_groups': 56,
        'num_workers': 4,
        'pin_memory': True,
        'amp': False,
        'amp_dtype': 'float32',
        'grad_accum_steps': 1,
        'compile_model': False,
        'tf32': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
    },
    # ---- NVIDIA T4  (16 GB, Colab free / GCP n1-standard) ----
    # FP16 is critical — T4 has weak FP32 but decent FP16 (65 TFLOPS).
    # 16 GB VRAM comfortable at 224×224: ~16 mv_groups (144 samples).
    't4': {
        'batch_size': 144,          # 16 mv_groups × 9 views
        'mv_groups': 16,
        'num_workers': 2,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 288
        'compile_model': False,     # T4 doesn't benefit much from compile
        'tf32': False,              # T4 doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA L4  (24 GB, GCP g2-standard) ----
    # Ada Lovelace arch: good FP16/BF16 (121 TFLOPS FP16).
    # 24 GB comfortable at 224×224: ~32 mv_groups.
    'l4': {
        'batch_size': 288,          # 32 mv_groups × 9 views
        'mv_groups': 32,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',    # BF16: same range as FP32 → no exp/log overflow
        'grad_accum_steps': 1,      # effective batch = 288
        'compile_model': False,     # disabled: interacts badly with grad checkpointing
        'tf32': True,               # Ada supports TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A10G  (24 GB, AWS g5) ----
    # Ampere arch, similar to L4 in VRAM but different compute profile.
    'a10g': {
        'batch_size': 288,          # 32 mv_groups × 9 views
        'mv_groups': 32,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',    # BF16: same range as FP32 → no exp/log overflow
        'grad_accum_steps': 1,      # effective batch = 288
        'compile_model': False,     # disabled: interacts badly with grad checkpointing
        'tf32': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA V100  (16 GB / 32 GB, GCP / AWS p3) ----
    # Volta: no TF32, no torch.compile benefit. Good FP16 via Tensor Cores.
    'v100': {
        'batch_size': 144,          # 16 mv_groups × 9 views
        'mv_groups': 16,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 288
        'compile_model': False,     # Volta doesn't benefit from compile
        'tf32': False,              # Volta doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A100  (40 GB / 80 GB, GCP a2, Colab Pro+) ----
    # Ampere flagship: TF32, BF16, huge memory bandwidth (2 TB/s).
    # At 224×224: fits large batches comfortably.
    'a100': {
        'batch_size': 1152,         # 128 mv_groups × 9 views
        'mv_groups': 128,
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',
        'grad_accum_steps': 1,      # effective batch = 1152
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 10,
        'persistent_workers': True,
    },
    # ---- NVIDIA H100  (80 GB, GCP a3, Lambda Labs) ----
    # Hopper: FP8 support, Transformer Engine, 3.4 TB/s bandwidth.
    # BF16 preferred (less overflow risk than FP16 at similar speed).
    'h100': {
        'batch_size': 2304,         # 256 mv_groups × 9 views
        'mv_groups': 256,
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',
        'grad_accum_steps': 1,      # effective batch = 2304
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    },
    # ---- Kaggle 2× Tesla T4 (dual-GPU, single node) ----
    # Per-process config mirrors 't4'. Launch with:
    #   accelerate launch --multi_gpu --num_processes 2 \
    #       -m RayNet.train --profile kaggle_t4x2 ...
    # Global effective batch = 144 * 2 * grad_accum = 576.
    'kaggle_t4x2': {
        'batch_size': 144,          # PER-GPU (global: 288 samples)
        'mv_groups': 16,
        'num_workers': 2,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # per-GPU → global effective 576
        'compile_model': False,
        'tf32': False,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- Two-machine distributed (one T4 per machine, same network) ----
    # Per-process config mirrors 't4'. Launch on each node with matching
    # --machine_rank; see module docstring for full launch command.
    'multi_node_t4': {
        'batch_size': 144,          # PER-GPU (global: 288 across both nodes)
        'mv_groups': 16,
        'num_workers': 2,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # per-GPU → global effective 576
        'compile_model': False,
        'tf32': False,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
}


AMP_DTYPE_MAP = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}


def apply_hardware_profile(args):
    """Load the chosen profile and overlay any CLI overrides."""
    hw = HARDWARE_PROFILES[args.profile].copy()

    if getattr(args, 'batch_size', None) is not None:
        hw['batch_size'] = args.batch_size
    if getattr(args, 'mv_groups', None) is not None:
        hw['mv_groups'] = args.mv_groups
    if getattr(args, 'num_workers', None) is not None:
        hw['num_workers'] = args.num_workers
    if getattr(args, 'grad_accum_steps', None) is not None:
        hw['grad_accum_steps'] = args.grad_accum_steps
    if getattr(args, 'no_compile', False):
        hw['compile_model'] = False

    return hw


def setup_hardware(hw, device):
    """Apply hardware-specific knobs (TF32) and print the active GPU."""
    if hw['tf32'] and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print("  TF32 enabled for matmul and cuDNN")

    if device.type == 'cuda':
        gpu_idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        gpu_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9
        print(f"  GPU[{gpu_idx}]: {gpu_name} ({gpu_mem:.1f} GB)")


def build_accelerator():
    """Create an Accelerator for distributed training.

    `mixed_precision='no'` is intentional: this codebase already manages
    AMP manually (autocast + GradScaler in train.py). Accelerate is used
    here purely for DDP / multi-node orchestration, dataloader sharding,
    and main-process gating of I/O.

    On a single machine with no `accelerate launch`, the returned
    Accelerator transparently falls back to single-process mode.
    """
    from accelerate import Accelerator
    return Accelerator(mixed_precision='no')
