#!/usr/bin/env bash
set -euo pipefail

# Simple InvSR inference runner (single command, easy to edit)
# Usage: ./scripts/run_inference_simple.sh

# GPU selection (edit if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Paths (edit if necessary)
IN_PATH="/data/yhc/stableSR_testdata/StableSR_testsets/DIV2K_V2_val/test_LR"
OUT_PATH="/home/yhc/code/InvSR_EMA/output/DIV2K_V2_val"
IMM_CKPT="/home/yhc/code/InvSR_EMA/save_dir/imm-run/imm_step_140000.pth"
CFG_PATH="configs/infer-imm.yaml"
REF_DIR="/data/yhc/stableSR_testdata/StableSR_testsets/DIV2K_V2_val/test_HR"
FID_REF="/data/yhc/stableSR_testdata/StableSR_testsets/DIV2K_V2_val/ref_stats_fid_HR.npz"

# Create output dir
mkdir -p "${OUT_PATH}"

# Run inference (single GPU; for multi-GPU use scripts/run_inference.sh)
python "$(dirname "$0")/../inference_invsr.py" \
  -i "${IN_PATH}" \
  -o "${OUT_PATH}" \
  --bs 1 --num_steps 1 \
  --cfg_path "${CFG_PATH}" \
  --imm_ckpt_path "${IMM_CKPT}" \
  --ref_dir "${REF_DIR}" \
  --fid_ref "${FID_REF}"

echo "Inference finished. Results are in ${OUT_PATH}"