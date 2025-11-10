"""
Device selection utilities for DyLoRA-MoE.

Provides consistent device selection logic across training and inference:
- CUDA (NVIDIA GPUs) - preferred for training and inference
- MPS (Apple Silicon) - good for inference on Mac
- CPU - fallback for systems without GPU acceleration
"""

import torch
from typing import Tuple, Optional


def get_device(force_device: Optional[str] = None) -> str:
    """
    Get the best available device for PyTorch operations.
    
    Priority order (when force_device is None):
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Args:
        force_device: Optional device override ('cpu', 'cuda', 'mps', 'cuda:0', etc.)
    
    Returns:
        Device string: 'cuda', 'mps', 'cpu', or specific device like 'cuda:0'
    """
    if force_device:
        return force_device
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_map(force_device: Optional[str] = None) -> Optional[str]:
    """
    Get the appropriate device_map for model loading.
    
    Returns 'auto' for CUDA (enables multi-GPU distribution),
    returns None for MPS and CPU (manual device placement).
    
    Args:
        force_device: Optional device override ('cpu', 'cuda', 'mps', etc.)
    
    Returns:
        'auto' for CUDA, None for MPS/CPU, or specific device if forced
    """
    device = get_device(force_device)
    
    if device.startswith('cuda'):
        # For specific CUDA device (e.g., 'cuda:0'), return that device
        if ':' in device:
            return device
        # For generic 'cuda', use auto distribution
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


def move_model_to_device(model: torch.nn.Module, verbose: bool = True, force_device: Optional[str] = None) -> torch.nn.Module:
    """
    Move a model to the best available device.
    
    Args:
        model: PyTorch model to move
        verbose: Whether to print device placement info
        force_device: Optional device override ('cpu', 'cuda', 'mps', 'cuda:0', etc.)
        
    Returns:
        Model on the selected device
    """
    device = get_device(force_device)
    
    if device.startswith('cuda'):
        model = model.to(device)
        if verbose:
            print(f"✓ Model moved to {device.upper()}")
    elif device == 'mps':
        model = model.to('mps')
        if verbose:
            print("✓ Model moved to MPS")
    else:
        model = model.to('cpu')
        if verbose:
            print("ℹ️  Model on CPU")
    
    return model


def get_device_info(force_dtype: Optional[str] = None, force_device: Optional[str] = None) -> Tuple[str, Optional[str], torch.dtype]:
    """
    Get comprehensive device configuration.
    
    Args:
        force_dtype: Optional override ('fp16', 'bf16', 'fp32')
        force_device: Optional device override ('cpu', 'cuda', 'mps', etc.)
    
    Returns:
        Tuple of (device, device_map, torch_dtype)
    """
    return get_device(force_device), get_device_map(force_device), get_torch_dtype(force_dtype)


def print_device_info(force_dtype: Optional[str] = None, force_device: Optional[str] = None):
    """Print detailed information about the selected device.
    
    Args:
        force_dtype: Optional override ('fp16', 'bf16', 'fp32')
        force_device: Optional device override ('cpu', 'cuda', 'mps', etc.)
    """
    device = get_device(force_device)
    dtype = get_torch_dtype(force_dtype)
    device_map = get_device_map(force_device)
    
    print("\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    print(f"Device: {device.upper()}")
    print(f"Dtype: {dtype}")
    print(f"Device Map: {device_map}")
    
    if device.startswith('cuda'):
        print("NVIDIA GPU acceleration enabled")
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  GPU {i} free memory: {torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)} bytes") 
            
    elif device == 'mps':
        print("Apple Silicon GPU acceleration enabled")
    else:
        print("No GPU acceleration available")
    
    print("="*60 + "\n")
