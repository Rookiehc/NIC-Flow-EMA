#!/usr/bin/env python3
"""
Inspect and optionally convert a folder of images/.npy to square images of given resolution.

Usage:
  # Just inspect first few files
  python scripts/fix_dataset_to_square.py --path /path/to/dataset --inspect

  # Convert files in-place (backup not created) to 32x32 png
  python scripts/fix_dataset_to_square.py --path /path/to/dataset --resize 32 --overwrite

This script supports .png/.jpg/.jpeg and .npy files. For .npy it expects either
  - shape (C, H, W) or (H, W, C) or (H, W)
Converted files will be written as PNG images unless --npy_out is given.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load_any(path: Path):
    ext = path.suffix.lower()
    if ext == '.npy':
        a = np.load(path)
        return a
    else:
        im = Image.open(path).convert('RGB')
        return np.array(im)


def to_chw_uint8(arr: np.ndarray):
    # arr can be HWC uint8, CHW uint8, or float
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 3:
        # HWC -> CHW
        if arr.dtype != np.uint8:
            # scale floats to 0..255
            a = (arr * 255).clip(0,255).astype(np.uint8)
        else:
            a = arr
        return a.transpose(2,0,1)
    elif arr.shape[0] == 3 and (arr.ndim == 3):
        # CHW already
        a = arr
        if a.dtype != np.uint8:
            a = (a * 255).clip(0,255).astype(np.uint8)
        return a
    else:
        # unknown layout -> try to coerce
        a = arr
        if a.dtype != np.uint8:
            a = (a * 255).clip(0,255).astype(np.uint8)
        if a.ndim == 3 and a.shape[-1] == 3:
            return a.transpose(2,0,1)
        elif a.ndim == 3 and a.shape[0] == 3:
            return a
        else:
            # fallback: convert to RGB via PIL
            im = Image.fromarray(a.astype(np.uint8))
            arr = np.array(im)
            return arr.transpose(2,0,1)


def save_as_png(chw, out_path: Path, resize=None):
    # chw -> HWC for PIL
    c,h,w = chw.shape
    img = np.transpose(chw, (1,2,0))
    pil = Image.fromarray(img)
    if resize is not None:
        pil = pil.resize((resize, resize), Image.BICUBIC)
    pil.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--resize', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--npy_out', action='store_true', help='output npy instead of png when converting')
    args = p.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(args.path)

    files = sorted([x for x in root.iterdir() if x.suffix.lower() in ['.png','.jpg','.jpeg','.npy']])
    if len(files) == 0:
        print('No supported files in', root)
        return

    for i,f in enumerate(files[:10]):
        try:
            a = load_any(f)
            if isinstance(a, np.ndarray):
                if a.ndim == 3:
                    if a.shape[0] in (1,3) and a.shape[1] == a.shape[2]:
                        print(f.name, 'shape(CHW square?)', a.shape)
                    elif a.shape[-1] in (1,3) and a.shape[-2] == a.shape[-3]:
                        print(f.name, 'shape(HWC square?)', a.shape)
                    else:
                        print(f.name, 'shape', a.shape)
                else:
                    print(f.name, 'shape', a.shape)
        except Exception as e:
            print('Error reading', f, e)

    if args.inspect:
        return

    # Convert files
    for f in files:
        try:
            a = load_any(f)
            chw = to_chw_uint8(a)
            out_path = f
            if not args.overwrite:
                out_dir = root / 'fixed'
                out_dir.mkdir(exist_ok=True)
                out_path = out_dir / (f.stem + '.png')
            if args.npy_out:
                # save as npy in CHW uint8
                np.save(out_path.with_suffix('.npy'), chw)
            else:
                save_as_png(chw, out_path, resize=args.resize)
            print('Wrote', out_path)
        except Exception as e:
            print('Failed to convert', f, '->', e)


if __name__ == '__main__':
    main()
