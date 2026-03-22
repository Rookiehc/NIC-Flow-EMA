
import torch
from diffusers import StableDiffusionPipeline
import os

try:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo", cache_dir="weights", local_files_only=True)
    print(f"Prediction Type: {getattr(pipe.scheduler.config, 'prediction_type', 'default/epsilon')}")
    print(f"Timestep spacing: {getattr(pipe.scheduler.config, 'timestep_spacing', 'unknown')}")
except Exception as e:
    print(f"Error: {e}")
