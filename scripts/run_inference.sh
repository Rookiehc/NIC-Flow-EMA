#!/usr/bin/env bash
set -euo pipefail

# Simple InvSR inference runner (single command, easy to edit)
# Usage: ./scripts/run_inference_simple.sh

# GPU selection (edit if needed)
export CUDA_VISIBLE_DEVICES=0

# Checkpoints+
IMM_CKPT="/home/yhc/code/InvSR_EMA/save_dir/imm_step_5000.pth"
START_CKPT="/home/yhc/code/InvSR_EMA/weights/ema_model_195000.pth"

# Config
CFG_PATH="configs/infer-imm.yaml"
REF_DIR="/data/yhc/stableSR_testdata/StableSR_testsets/DrealSRVal_crop128/test_HR"
IN_PATH="/data/yhc/stableSR_testdata/StableSR_testsets/DrealSRVal_crop128/test_LR"
OUT_PATH="/home/yhc/code/InvSR_EMA/output/DrealSRVal_crop128"

# Create output dir
mkdir -p "${OUT_PATH}"

# 1. Run inference
echo "Starting Inference..."
python "$(dirname "$0")/../inference_invsr.py" \
  -i "${IN_PATH}" \
  -o "${OUT_PATH}" \
  --bs 1 --num_steps 1 \
  --cfg_path "${CFG_PATH}" \
  --imm_ckpt_path "${IMM_CKPT}" \
  --started_ckpt_path "${START_CKPT}" \
  --ref_dir "${REF_DIR}" 

# 2. Run Metrics Evaluation
echo "Starting Evaluation..."
# Use absolute path to the evaluation script as requested
EVAL_SCRIPT="/home/yhc/code/InvSR/scripts/eval_image_metrics.py"

if [ -f "$EVAL_SCRIPT" ]; then
    python "$EVAL_SCRIPT" \
      --sr_dir "${OUT_PATH}" \
      --gt_dir "${REF_DIR}" \
      --bs 8 \
      --y_channel true \
      --niqe --pi
else
    echo "Warning: Evaluation script not found at $EVAL_SCRIPT"
fi

echo "All Finished. Results in ${OUT_PATH}"