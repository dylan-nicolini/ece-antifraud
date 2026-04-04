#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# archive_experiment_data.sh
#
# Purpose:
#   Create an experiment package under:
#     ~/ece-antifraud/experiments/<experiment-name>
#
#   Preserve source structure inside package:
#     <experiment>/
#       artifacts/   <- from ~/ece-antifraud/methods/gtan/artifacts
#       data/        <- from ~/ece-antifraud/data
#       logs/
#
#   Copy into package:
#     artifacts/: recursively include *.csv, *.npz, *.json, *.txt
#                 while preserving subfolder structure
#     data/:      *.csv, *.log, *.bin
#
#   Validate everything copied correctly.
#   Zip the entire experiment folder.
#   Optionally upload the zip to S3 via AWS CLI.
#   Leave copied files in place.
#   Optionally clean up original source files at the very end.
#
# Usage:
#   Dry run:
#     ./archive_experiment_data.sh --dry-run --experiment-name exp01_gtan
#
#   Package only:
#     ./archive_experiment_data.sh --experiment-name exp01_gtan
#
#   Package + upload to S3:
#     ./archive_experiment_data.sh --experiment-name exp01_gtan --upload-to-s3
#
#   Package + upload + cleanup originals:
#     ./archive_experiment_data.sh --experiment-name exp01_gtan --upload-to-s3 --cleanup-sources
###############################################################################

DRY_RUN=false
CLEANUP_SOURCES=false
UPLOAD_TO_S3=false
EXPERIMENT_NAME=""

EXPERIMENTS_ROOT="${HOME}/ece-antifraud/experiments"
ARTIFACTS_SOURCE_DIR="${HOME}/ece-antifraud/methods/gtan/artifacts"
DATA_SOURCE_DIR="${HOME}/ece-antifraud/data"
S3_BASE_URI="s3://cc-fraud-storage/experiment-data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --cleanup-sources)
      CLEANUP_SOURCES=true
      shift
      ;;
    --upload-to-s3)
      UPLOAD_TO_S3=true
      shift
      ;;
    --experiment-name)
      EXPERIMENT_NAME="${2:-}"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Usage: $0 [--dry-run] [--upload-to-s3] [--cleanup-sources] --experiment-name <experiment_name>"
      exit 1
      ;;
  esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "ERROR: --experiment-name is required"
  echo "Usage: $0 [--dry-run] [--upload-to-s3] [--cleanup-sources] --experiment-name <experiment_name>"
  exit 1
fi

if [[ ! -d "$EXPERIMENTS_ROOT" ]]; then
  echo "ERROR: Experiments root directory does not exist: $EXPERIMENTS_ROOT"
  exit 1
fi

if [[ ! -d "$ARTIFACTS_SOURCE_DIR" ]]; then
  echo "ERROR: Artifacts source directory does not exist: $ARTIFACTS_SOURCE_DIR"
  exit 1
fi

if [[ ! -d "$DATA_SOURCE_DIR" ]]; then
  echo "ERROR: Data source directory does not exist: $DATA_SOURCE_DIR"
  exit 1
fi

if [[ "$UPLOAD_TO_S3" == true ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI is not installed or not in PATH."
    exit 1
  fi
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "ERROR: zip is not installed or not in PATH."
  exit 1
fi

if ! command -v zipinfo >/dev/null 2>&1; then
  echo "ERROR: zipinfo is not installed or not in PATH."
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

EXPERIMENT_DIR="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}"
EXPERIMENT_ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts"
EXPERIMENT_DATA_DIR="${EXPERIMENT_DIR}/data"
EXPERIMENT_LOGS_DIR="${EXPERIMENT_DIR}/logs"

ZIP_PATH="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}.zip"
LOG_PATH="${EXPERIMENT_LOGS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
MANIFEST_PATH="${EXPERIMENT_LOGS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_manifest.txt"
ZIP_LIST_PATH="${EXPERIMENT_LOGS_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_zip_entries.txt"

TMP_ARTIFACTS="$(mktemp)"
TMP_DATA="$(mktemp)"
TMP_MANIFEST="$(mktemp)"
TMP_ZIP_LIST="$(mktemp)"

cleanup_temp() {
  rm -f "$TMP_ARTIFACTS" "$TMP_DATA" "$TMP_MANIFEST" "$TMP_ZIP_LIST"
}
trap cleanup_temp EXIT

log_console() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log() {
  if [[ "$DRY_RUN" == true ]]; then
    log_console "$1"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_PATH"
  fi
}

cleanup_on_error() {
  log "ERROR: Script failed."
  if [[ "$DRY_RUN" == true ]]; then
    log "Dry-run mode enabled. No files were changed."
  else
    log "Any original source cleanup was not performed unless already logged."
    log "Packaged files remain in place unless this failure occurred before copy completed."
  fi
}
trap cleanup_on_error ERR

find "$ARTIFACTS_SOURCE_DIR" -type f \( \
    -iname '*.csv' -o \
    -iname '*.npz' -o \
    -iname '*.json' -o \
    -iname '*.txt' \
  \) | sort > "$TMP_ARTIFACTS"

find "$DATA_SOURCE_DIR" -maxdepth 1 -type f \( \
    -iname '*.csv' -o \
    -iname '*.log' -o \
    -iname '*.bin' \
  \) | sort > "$TMP_DATA"

ARTIFACT_COUNT="$(wc -l < "$TMP_ARTIFACTS" | tr -d ' ')"
DATA_COUNT="$(wc -l < "$TMP_DATA" | tr -d ' ')"
TOTAL_COUNT=$((ARTIFACT_COUNT + DATA_COUNT))

if [[ "$TOTAL_COUNT" -eq 0 ]]; then
  log_console "No matching files found to package."
  exit 0
fi

log_console "============================================================"
log_console "STARTING EXPERIMENT PACKAGING"
log_console "Mode                : $([[ "$DRY_RUN" == true ]] && echo 'DRY-RUN' || echo 'LIVE')"
log_console "Cleanup sources     : $([[ "$CLEANUP_SOURCES" == true ]] && echo 'YES' || echo 'NO')"
log_console "Upload to S3        : $([[ "$UPLOAD_TO_S3" == true ]] && echo 'YES' || echo 'NO')"
log_console "Experiment name     : $EXPERIMENT_NAME"
log_console "Experiments root    : $EXPERIMENTS_ROOT"
log_console "Experiment dir      : $EXPERIMENT_DIR"
log_console "Zip path            : $ZIP_PATH"
log_console "Artifacts source    : $ARTIFACTS_SOURCE_DIR"
log_console "Data source         : $DATA_SOURCE_DIR"
if [[ "$UPLOAD_TO_S3" == true ]]; then
  log_console "S3 destination      : ${S3_BASE_URI}/${EXPERIMENT_NAME}.zip"
fi
log_console "============================================================"

log_console "Artifacts files found: $ARTIFACT_COUNT"
while IFS= read -r file; do
  [[ -n "$file" ]] && log_console "  [ARTIFACT] $file"
done < "$TMP_ARTIFACTS"

log_console "Data files found: $DATA_COUNT"
while IFS= read -r file; do
  [[ -n "$file" ]] && log_console "  [DATA] $file"
done < "$TMP_DATA"

if [[ "$DRY_RUN" == true ]]; then
  log "------------------------------------------------------------"
  log "PLAN"
  log "------------------------------------------------------------"
  log "1. Create experiment folder structure:"
  log "   $EXPERIMENT_DIR"
  log "   $EXPERIMENT_ARTIFACTS_DIR"
  log "   $EXPERIMENT_DATA_DIR"
  log "   $EXPERIMENT_LOGS_DIR"
  log "2. Copy artifacts (*.csv, *.npz, *.json, *.txt) into artifacts/ preserving subfolders"
  log "3. Copy data (*.csv, *.log, *.bin) into data/"
  log "4. Validate every copied file by existence and file size"
  log "5. Create zip: $ZIP_PATH"
  log "6. Validate zip contents"
  if [[ "$UPLOAD_TO_S3" == true ]]; then
    log "7. Upload zip to S3: ${S3_BASE_URI}/${EXPERIMENT_NAME}.zip"
    if [[ "$CLEANUP_SOURCES" == true ]]; then
      log "8. After S3 validation succeeds, delete original source files"
    else
      log "8. Leave original source files untouched"
    fi
  else
    if [[ "$CLEANUP_SOURCES" == true ]]; then
      log "7. After validation succeeds, delete original source files"
    else
      log "7. Leave original source files untouched"
    fi
  fi

  log "------------------------------------------------------------"
  log "DRY-RUN COPY ACTIONS"
  log "------------------------------------------------------------"
  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    rel_path="${src_file#${ARTIFACTS_SOURCE_DIR}/}"
    log "WOULD COPY [ARTIFACT]: $src_file -> ${EXPERIMENT_ARTIFACTS_DIR}/${rel_path}"
  done < "$TMP_ARTIFACTS"

  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    log "WOULD COPY [DATA]: $src_file -> ${EXPERIMENT_DATA_DIR}/$(basename "$src_file")"
  done < "$TMP_DATA"

  log "------------------------------------------------------------"
  log "DRY-RUN VALIDATION ACTIONS"
  log "------------------------------------------------------------"
  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    rel_path="${src_file#${ARTIFACTS_SOURCE_DIR}/}"
    log "WOULD VALIDATE [ARTIFACT]: ${EXPERIMENT_ARTIFACTS_DIR}/${rel_path} (expected size=$(stat -c%s "$src_file") bytes)"
  done < "$TMP_ARTIFACTS"

  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    log "WOULD VALIDATE [DATA]: ${EXPERIMENT_DATA_DIR}/$(basename "$src_file") (expected size=$(stat -c%s "$src_file") bytes)"
  done < "$TMP_DATA"

  log "------------------------------------------------------------"
  log "DRY-RUN ZIP ACTIONS"
  log "------------------------------------------------------------"
  log "WOULD CREATE ZIP: $ZIP_PATH"
  log "WOULD ZIP DIRECTORY: $EXPERIMENT_DIR"

  if [[ "$UPLOAD_TO_S3" == true ]]; then
    log "------------------------------------------------------------"
    log "DRY-RUN S3 ACTIONS"
    log "------------------------------------------------------------"
    log "WOULD UPLOAD: $ZIP_PATH -> ${S3_BASE_URI}/${EXPERIMENT_NAME}.zip"
  fi

  if [[ "$CLEANUP_SOURCES" == true ]]; then
    log "------------------------------------------------------------"
    log "DRY-RUN CLEANUP ACTIONS"
    log "------------------------------------------------------------"
    while IFS= read -r src_file; do
      [[ -z "$src_file" ]] && continue
      log "WOULD DELETE SOURCE [ARTIFACT]: $src_file"
    done < "$TMP_ARTIFACTS"
    while IFS= read -r src_file; do
      [[ -z "$src_file" ]] && continue
      log "WOULD DELETE SOURCE [DATA]: $src_file"
    done < "$TMP_DATA"
  fi

  log "------------------------------------------------------------"
  log "DRY-RUN COMPLETE"
  log "No files were copied, zipped, uploaded, or deleted."
  log_console "============================================================"
  exit 0
fi

if [[ -e "$EXPERIMENT_DIR" ]]; then
  echo "ERROR: Experiment directory already exists: $EXPERIMENT_DIR"
  echo "Choose a different --experiment-name or remove the existing folder first."
  exit 1
fi

if [[ -f "$ZIP_PATH" ]]; then
  echo "ERROR: Zip file already exists: $ZIP_PATH"
  echo "Choose a different --experiment-name or remove the existing zip first."
  exit 1
fi

mkdir -p "$EXPERIMENT_ARTIFACTS_DIR" "$EXPERIMENT_DATA_DIR" "$EXPERIMENT_LOGS_DIR"
: > "$LOG_PATH"

{
  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    rel_path="${src_file#${ARTIFACTS_SOURCE_DIR}/}"
    echo "ARTIFACT|$src_file|${EXPERIMENT_ARTIFACTS_DIR}/${rel_path}"
  done < "$TMP_ARTIFACTS"

  while IFS= read -r src_file; do
    [[ -z "$src_file" ]] && continue
    echo "DATA|$src_file|${EXPERIMENT_DATA_DIR}/$(basename "$src_file")"
  done < "$TMP_DATA"
} > "$TMP_MANIFEST"

cp "$TMP_MANIFEST" "$MANIFEST_PATH"

log "------------------------------------------------------------"
log "COPYING FILES"
log "------------------------------------------------------------"

COPIED_COUNT=0
while IFS='|' read -r file_type src_file dest_file; do
  [[ -z "$src_file" ]] && continue

  dest_dir="$(dirname "$dest_file")"
  mkdir -p "$dest_dir"

  if [[ -e "$dest_file" ]]; then
    log "ERROR: Destination already exists: $dest_file"
    exit 1
  fi

  cp -p "$src_file" "$dest_file"
  log "COPIED [$file_type]: $src_file -> $dest_file"
  COPIED_COUNT=$((COPIED_COUNT + 1))
done < "$MANIFEST_PATH"

log "Total copied files: $COPIED_COUNT"

log "------------------------------------------------------------"
log "VALIDATING COPIED FILES"
log "------------------------------------------------------------"

VALIDATED_COUNT=0
ARTIFACT_VALIDATED=0
DATA_VALIDATED=0

while IFS='|' read -r file_type src_file dest_file; do
  [[ -z "$src_file" ]] && continue

  if [[ ! -f "$dest_file" ]]; then
    log "VALIDATION FAILED [$file_type]: Missing copied file: $dest_file"
    exit 1
  fi

  src_size="$(stat -c%s "$src_file")"
  dest_size="$(stat -c%s "$dest_file")"

  if [[ "$src_size" -ne "$dest_size" ]]; then
    log "VALIDATION FAILED [$file_type]: Size mismatch: src=$src_size dest=$dest_size file=$dest_file"
    exit 1
  fi

  log "VALIDATED [$file_type]: $dest_file (size=$dest_size bytes)"
  VALIDATED_COUNT=$((VALIDATED_COUNT + 1))

  if [[ "$file_type" == "ARTIFACT" ]]; then
    ARTIFACT_VALIDATED=$((ARTIFACT_VALIDATED + 1))
  else
    DATA_VALIDATED=$((DATA_VALIDATED + 1))
  fi
done < "$MANIFEST_PATH"

if [[ "$VALIDATED_COUNT" -ne "$TOTAL_COUNT" ]]; then
  log "VALIDATION FAILED: Expected $TOTAL_COUNT files, validated $VALIDATED_COUNT"
  exit 1
fi

log "Validation passed."
log "Artifact files validated: $ARTIFACT_VALIDATED"
log "Data files validated    : $DATA_VALIDATED"
log "Total files validated   : $VALIDATED_COUNT"

log "------------------------------------------------------------"
log "CREATING ZIP"
log "------------------------------------------------------------"

(
  cd "$EXPERIMENTS_ROOT"
  zip -q -r "$ZIP_PATH" "$EXPERIMENT_NAME"
)

if [[ ! -f "$ZIP_PATH" ]]; then
  log "ZIP VALIDATION FAILED: Zip file was not created: $ZIP_PATH"
  exit 1
fi

ZIP_SIZE="$(stat -c%s "$ZIP_PATH")"
if [[ "$ZIP_SIZE" -le 0 ]]; then
  log "ZIP VALIDATION FAILED: Zip file is empty: $ZIP_PATH"
  exit 1
fi

zipinfo -1 "$ZIP_PATH" | sort > "$TMP_ZIP_LIST"
cp "$TMP_ZIP_LIST" "$ZIP_LIST_PATH"

ZIP_ENTRY_COUNT="$(wc -l < "$TMP_ZIP_LIST" | tr -d ' ')"
MIN_EXPECTED_ENTRIES=$((TOTAL_COUNT + 2))

if [[ "$ZIP_ENTRY_COUNT" -lt "$MIN_EXPECTED_ENTRIES" ]]; then
  log "ZIP VALIDATION FAILED: Zip entries ($ZIP_ENTRY_COUNT) less than minimum expected ($MIN_EXPECTED_ENTRIES)"
  exit 1
fi

log "ZIP CREATED: $ZIP_PATH"
log "ZIP SIZE   : $ZIP_SIZE bytes"
log "ZIP ENTRIES: $ZIP_ENTRY_COUNT"

log "Zip contents:"
while IFS= read -r zip_entry; do
  log "  ZIP ENTRY: $zip_entry"
done < "$ZIP_LIST_PATH"

if [[ "$UPLOAD_TO_S3" == true ]]; then
  log "------------------------------------------------------------"
  log "UPLOADING ZIP TO S3"
  log "------------------------------------------------------------"

  S3_URI="${S3_BASE_URI}/${EXPERIMENT_NAME}.zip"
  log "S3 destination: $S3_URI"

  aws s3 cp "$ZIP_PATH" "$S3_URI"

  log "S3 upload completed. Verifying uploaded object..."

  aws s3 ls "$S3_URI" >/dev/null 2>&1 || {
    log "S3 VALIDATION FAILED: Uploaded object not found: $S3_URI"
    exit 1
  }

  log "S3 upload validated: $S3_URI"
fi

log "------------------------------------------------------------"
log "FINAL PACKAGE SUMMARY"
log "------------------------------------------------------------"
log "Experiment directory created : $EXPERIMENT_DIR"
log "Artifacts package directory  : $EXPERIMENT_ARTIFACTS_DIR"
log "Data package directory       : $EXPERIMENT_DATA_DIR"
log "Logs directory               : $EXPERIMENT_LOGS_DIR"
log "Manifest file                : $MANIFEST_PATH"
log "Zip listing file             : $ZIP_LIST_PATH"
log "Zip file                     : $ZIP_PATH"

if [[ "$CLEANUP_SOURCES" == true ]]; then
  log "------------------------------------------------------------"
  log "CLEANING UP ORIGINAL SOURCE FILES"
  log "------------------------------------------------------------"

  DELETED_COUNT=0
  while IFS='|' read -r file_type src_file dest_file; do
    [[ -z "$src_file" ]] && continue

    if [[ -f "$src_file" ]]; then
      log "DELETING SOURCE [$file_type]: $src_file"
      rm -f "$src_file"
      DELETED_COUNT=$((DELETED_COUNT + 1))
    else
      log "SKIPPED DELETE [$file_type]: Already missing: $src_file"
    fi
  done < "$MANIFEST_PATH"

  log "Deleted source files: $DELETED_COUNT"

  log "------------------------------------------------------------"
  log "POST-CLEANUP VERIFICATION"
  log "------------------------------------------------------------"

  REMAINING_COUNT=0
  while IFS='|' read -r file_type src_file dest_file; do
    [[ -z "$src_file" ]] && continue

    if [[ -f "$src_file" ]]; then
      log "POST-CLEANUP FAILED [$file_type]: Source still exists: $src_file"
      REMAINING_COUNT=$((REMAINING_COUNT + 1))
    fi
  done < "$MANIFEST_PATH"

  if [[ "$REMAINING_COUNT" -gt 0 ]]; then
    log "POST-CLEANUP FAILED: $REMAINING_COUNT source files still remain."
    exit 1
  fi

  log "Post-cleanup verification passed."
else
  log "Original source files were left untouched."
fi

log "============================================================"
log "EXPERIMENT PACKAGE CREATED SUCCESSFULLY"
log "Artifacts files found : $ARTIFACT_COUNT"
log "Data files found      : $DATA_COUNT"
log "Total files packaged  : $TOTAL_COUNT"
if [[ "$UPLOAD_TO_S3" == true ]]; then
  log "S3 upload             : ${S3_BASE_URI}/${EXPERIMENT_NAME}.zip"
fi
log "Zip file              : $ZIP_PATH"
log "============================================================"