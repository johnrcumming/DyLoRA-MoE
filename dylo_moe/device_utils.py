"""
Device selection utilities for DyLoRA-MoE.

Provides consistent device selection logic across training and inference:
- CUDA (NVIDIA GPUs) - preferred for training and inference
- MPS (Apple Silicon) - good for inference on Mac
- CPU - fallback for systems without GPU acceleration
"""

import torch
from typing import Tuple, Optional


def get_device() -> str:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_map() -> Optional[str]:
    """
    Get the appropriate device_map for model loading.
    
    Returns 'auto' for CUDA (enables multi-GPU distribution),
    returns None for MPS and CPU (manual device placement).
    
    Returns:
        'auto' for CUDA, None for MPS/CPU
    """
    if torch.cuda.is_available():
        return 'auto'
    else:
        return None


def get_torch_dtype(force_dtype: Optional[str] = None) -> torch.dtype:
    """
    Get the appropriate torch dtype for the current device.
    
    - CUDA: bfloat16 (best for modern GPUs)
    - MPS: float32 (MPS has limited bfloat16 support)
    - CPU: float32 (no low-precision support)
    
    Args:
        force_dtype: Optional override ('fp16', 'bf16', 'fp32')
    
    Returns:
        torch.dtype: bfloat16 for CUDA, float32 otherwise, or forced dtype
    """
    if force_dtype:
        dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'fp32': torch.float32,
        }
        if force_dtype.lower() in dtype_map:
            return dtype_map[force_dtype.lower()]
    
    if torch.cuda.is_available():
        return torch.bfloat16
    else:
        return torch.float32


def move_model_to_device(model: torch.nn.Module, verbose: bool = True) -> torch.nn.Module:
    """
    Move a model to the best available device.
    
    Args:
        model: PyTorch model to move
        verbose: Whether to print device placement info
        
    Returns:
        Model on the selected device
    """
    device = get_device()
    
    if device == 'cuda':
        model = model.cuda()
        if verbose:
            print("✓ Model moved to CUDA")
    elif device == 'mps':
        model = model.to('mps')
        if verbose:
            print("✓ Model moved to MPS")
    else:
        if verbose:
            print("ℹ️  Model on CPU")
    
    return model


def get_device_info(force_dtype: Optional[str] = None) -> Tuple[str, Optional[str], torch.dtype]:
    """
    Get comprehensive device configuration.
    
    Args:
        force_dtype: Optional override ('fp16', 'bf16', 'fp32')
    
    Returns:
        Tuple of (device, device_map, torch_dtype)
    """
    return get_device(), get_device_map(), get_torch_dtype(force_dtype)


def print_device_info(force_dtype: Optional[str] = None):
    """Print detailed information about the selected device.
    
    Args:
        force_dtype: Optional override ('fp16', 'bf16', 'fp32')
    """
    device = get_device()
    dtype = get_torch_dtype(force_dtype)
    device_map = get_device_map()
    
    print("\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    print(f"Device: {device.upper()}")
    print(f"Dtype: {dtype}")
    print(f"Device Map: {device_map}")
    
    if device == 'cuda':
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    elif device == 'mps':
        print("Apple Silicon GPU acceleration enabled")
    else:
        print("No GPU acceleration available")
    
    print("="*60 + "\n")
