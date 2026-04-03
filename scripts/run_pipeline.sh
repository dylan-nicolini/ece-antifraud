#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  ece-antifraud full EC2 pipeline
#
# Runs inside ~/ece-antifraud on the EC2 instance.
# Called by agent_v1.py after the CSV has been downloaded to ./data/
#
# Usage:
#   bash run_pipeline.sh --experiment-name exp01_gtan_sffsd [--dataset sffsd|ieee]
#
# Steps:
#   1. Feature engineering
#   2. GTAN training (with Comet logging)
#   3. Archive experiment (zip + S3 upload + cleanup)
# =============================================================================
set -Eeuo pipefail

PYTHON="/home/ubuntu/miniconda3/envs/fraud_env/bin/python"
export PYTHONNOUSERSITE=1

export COMET_API_KEY="iU27xMQWN5Wi4rc3VLC8E34Az"
export COMET_PROJECT_NAME="ece-thesis-fraud"
export COMET_WORKSPACE="dylan-nicolini"

DATASET="sffsd"
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-name)
      EXPERIMENT_NAME="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:-sffsd}"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Usage: $0 --experiment-name <name> [--dataset sffsd|ieee]"
      exit 1
      ;;
  esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "ERROR: --experiment-name is required"
  exit 1
fi

# Pick the right feature engineering script based on dataset
if [[ "$DATASET" == "ieee" ]]; then
  FEATURE_ENG_SCRIPT="feature_engineering/data_process_v3.py"
else
  FEATURE_ENG_SCRIPT="feature_engineering/data_process.py"
fi

echo "============================================================"
echo "ECE-AntifrAud Pipeline"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Dataset     : $DATASET"
echo "Feature eng : $FEATURE_ENG_SCRIPT"
echo "Started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Step 1: Feature Engineering ──────────────────────────────────────────────
echo ""
echo "[1/3] Running feature engineering..."
$PYTHON "$FEATURE_ENG_SCRIPT"
echo "[1/3] Feature engineering complete."

# ── Step 2: GTAN Training ────────────────────────────────────────────────────
echo ""
echo "[2/3] Running GTAN algorithm..."
$PYTHON main.py --method gtan
echo "[2/3] GTAN training complete."

# ── Step 3: Archive Experiment ───────────────────────────────────────────────
echo ""
echo "[3/3] Archiving experiment..."
bash scripts/archive_experiment_data.sh \
  --experiment-name "$EXPERIMENT_NAME" \
  --upload-to-s3 \
  --cleanup-sources
echo "[3/3] Archive complete."

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE: $EXPERIMENT_NAME"
echo "Finished    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"