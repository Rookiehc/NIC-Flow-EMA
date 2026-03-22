import torch
import sys
import os

ckpt_path = "save_dir/-20251231_131422/imm_step_15000.pth"
if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
    sys.exit(1)

try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Keys in checkpoint: {ckpt.keys()}")
    if 'optimizer' in ckpt:
        print("Optimizer state found.")
    else:
        print("Optimizer state NOT found.")
        
    state_dict = ckpt.get("state_dict", ckpt)
    
    nan_count = 0
    param_count = 0
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            param_count += 1
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"Layer {k} has NaNs or Infs!")
                nan_count += 1
                
    if nan_count > 0:
        print(f"Found {nan_count} layers with NaNs/Infs out of {param_count} layers.")
        print("The model is corrupted.")
    else:
        print("No NaNs or Infs found in the weights.")
        print("The weights are numerically valid (but might be degraded).")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
