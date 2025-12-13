#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Lightweight FID utilities for InvSR.

This module provides:

1) A small script-style function to pre-compute reference statistics
   (mu, sigma) for a given image folder and save them as
   `ref_stats_fid.npz`.

2) A simple online FID accumulator that can be used during inference to
   compute FID between generated images and the pre-computed reference
   statistics.

Design goals:
 - No cross-project dependencies: only use PyTorch, torchvision and
   numpy/scipy.
 - Simple API that can be called from `inference_invsr.py`.

Usage example (offline, to build reference stats for GT dataset):

	from utils.fid_utils import compute_ref_stats_for_folder
	compute_ref_stats_for_folder(
		img_dir="/data/yhc/imm_val_datasets/img512",
		save_path="/data/yhc/imm_val_datasets/ref_stats_fid.npz",
		batch_size=32,
		device="cuda"
	)

During inference (online FID):

	from utils.fid_utils import FIDOnline
	fid = FIDOnline(ref_stats_path="/data/.../ref_stats_fid.npz", device="cuda")
	...  # For each batch of generated images and GT images in [0,1]
	fid.update(gen_batch, gt_batch)
	final_fid = fid.compute()

"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import scipy.linalg


__all__ = [
	"compute_ref_stats_for_folder",
	"FIDOnline",
]


def _get_inception(device: torch.device) -> nn.Module:
	"""Load torchvision Inception v3 and return a feature extractor.

	The network outputs a 2048-D feature vector per image.
	"""

	inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
	inception.fc = nn.Identity()
	inception.eval().to(device)
	return inception


def _preprocess_images(x: torch.Tensor) -> torch.Tensor:
	"""Preprocess images for Inception.

	Args:
		x: Tensor of shape (N, C, H, W), values in [0, 1] or [0, 255].

	Returns:
		Tensor float32 in range [0, 1], resized to (299, 299).
	"""

	if x.dtype != torch.float32:
		x = x.float()
	# If in [0,255], scale to [0,1]
	if x.max() > 1.5:
		x = x / 255.0
	x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
	return x


def _images_to_features(inception: nn.Module, imgs: torch.Tensor) -> np.ndarray:
	"""Run images through Inception and return N x 2048 features (numpy)."""

	with torch.no_grad():
		imgs = _preprocess_images(imgs)
		feats = inception(imgs)
	feats = feats.cpu().numpy()
	return feats


def _compute_stats_from_features(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Compute mean and covariance of feature matrix.

	Args:
		feats: numpy array of shape (N, D)

	Returns:
		mu: (D,)
		sigma: (D, D)
	"""

	assert feats.ndim == 2, "features must be (N, D)"
	mu = np.mean(feats, axis=0)
	sigma = np.cov(feats, rowvar=False)
	return mu, sigma


def compute_ref_stats_for_folder(
	img_dir: str,
	save_path: str,
	batch_size: int = 32,
	device: str | torch.device = "cuda",
) -> None:
	"""Compute reference FID stats (mu, sigma) for a folder of images.

	Args:
		img_dir: Path to folder containing reference images (e.g. GT, 512x512).
		save_path: Path to `.npz` file to save stats (`mu`, `sigma`).
		batch_size: Batch size for Inception feature extraction.
		device: Device string or torch.device (e.g. "cuda" or "cpu").
	"""

	img_dir = Path(img_dir)
	if not img_dir.exists() or not img_dir.is_dir():
		raise FileNotFoundError(f"Image directory not found: {img_dir}")

	device = torch.device(device)
	inception = _get_inception(device)

	exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
	paths = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in exts]
	if len(paths) < 2:
		raise RuntimeError(f"Need at least 2 images to compute FID stats, got {len(paths)}")

	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.ToTensor(),  # [0,1], CxHxW
	])

	all_feats = []
	for i in range(0, len(paths), batch_size):
		batch_paths = paths[i : i + batch_size]
		imgs = []
		for p in batch_paths:
			img = Image.open(p).convert("RGB")
			img = transform(img)  # CxHxW in [0,1]
			imgs.append(img)
		imgs = torch.stack(imgs, dim=0).to(device)  # NxCxHxW
		feats = _images_to_features(inception, imgs)
		all_feats.append(feats)

	all_feats = np.concatenate(all_feats, axis=0)
	mu, sigma = _compute_stats_from_features(all_feats)

	save_path = Path(save_path)
	if not save_path.parent.exists():
		save_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(save_path, mu=mu, sigma=sigma)
	print(f"Saved FID reference stats to {save_path} (N={all_feats.shape[0]})")


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
	"""Numpy implementation of Frechet distance between two Gaussians.

	This follows the standard FID formulation.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)
	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
	assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

	diff = mu1 - mu2

	covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			raise ValueError("Imaginary component in covmean")
		covmean = covmean.real

	tr_covmean = np.trace(covmean)
	fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
	return float(fid)


class FIDOnline:
	"""Online FID accumulator.

	- Load reference stats (mu_ref, sigma_ref) from an `.npz` file.
	- For each batch of generated images, extract Inception features and
	  accumulate mean/covariance statistics.
	- Call `.compute()` at the end to get scalar FID.

	Notes:
		* Images are expected in [0,1] float format, shape (N, C, H, W).
		* This class only uses generated images vs. reference stats, i.e.,
		  it does NOT use per-image GT, which matches standard FID.
	"""

	def __init__(self, ref_stats_path: str, device: str | torch.device = "cuda") -> None:
		ref_stats_path = Path(ref_stats_path)
		if not ref_stats_path.exists():
			raise FileNotFoundError(f"ref_stats file not found: {ref_stats_path}")

		data = np.load(ref_stats_path)
		self.mu_ref = data["mu"]
		self.sigma_ref = data["sigma"]

		self.device = torch.device(device)
		self.inception = _get_inception(self.device)

		# Accumulators for generated images
		self._sum = None
		self._sum_sq = None
		self._count = 0

	def update(self, imgs: torch.Tensor) -> None:
		"""Accumulate features for a batch of generated images.

		Args:
			imgs: Tensor (N, C, H, W), values in [0,1] or [0,255].
		"""

		if imgs is None:
			return
		if not torch.is_tensor(imgs):
			imgs = torch.from_numpy(imgs)
		imgs = imgs.to(self.device)

		feats = _images_to_features(self.inception, imgs)  # (N, D)

		if self._sum is None:
			dim = feats.shape[1]
			self._sum = np.zeros(dim, dtype=np.float64)
			self._sum_sq = np.zeros((dim, dim), dtype=np.float64)

		self._sum += feats.sum(axis=0)
		self._sum_sq += feats.T @ feats
		self._count += feats.shape[0]

	def compute(self) -> float:
		"""Compute FID between accumulated features and reference stats."""

		if self._count < 2:
			raise RuntimeError("Not enough images to compute FID; call update() with more data.")

		mu_gen = self._sum / self._count
		sigma_gen = (self._sum_sq / (self._count - 1)) - np.outer(mu_gen, mu_gen) * (self._count / (self._count - 1))

		fid = _calculate_frechet_distance(mu_gen, sigma_gen, self.mu_ref, self.sigma_ref)
		return fid


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Compute FID reference stats for a folder of images.")
	parser.add_argument("img_dir", type=str, help="Path to image folder (reference dataset).")
	parser.add_argument("save_path", type=str, help="Where to save ref_stats_fid.npz.")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Inception.")
	parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. 'cuda' or 'cpu'.")
	args = parser.parse_args()

	compute_ref_stats_for_folder(
		img_dir=args.img_dir,
		save_path=args.save_path,
		batch_size=args.batch_size,
		device=args.device,
	)

