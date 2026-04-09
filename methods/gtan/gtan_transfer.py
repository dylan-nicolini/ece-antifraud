#!/usr/bin/env python3
"""
gtan_transfer.py
================
Run a GTAN checkpoint trained on one dataset against a different dataset.

Supports both directions:
  Direction A:  Train on S-FFSD  → validate against IEEE-CIS
  Direction B:  Train on IEEE-CIS → validate against S-FFSD

And any fraud concentration interval (controlled by --fraud-pct label tag;
the actual concentration comes from whatever CSV you point --neofull-csv at).

Two embedding mismatch strategies:

  reinit (default / Option B)
    Rebuild embedding tables sized to the TARGET dataset's vocabulary.
    Load checkpoint strict=False — GNN/transformer/MLP weights transfer,
    embedding lookup tables re-randomized. Tests GNN weight transfer only.

  clamp (Option A)
    Clip target categorical IDs to [0, train_cardinality - 1].
    Load checkpoint strict=True — full weight transfer including embeddings,
    but out-of-vocabulary entities silently collapse to bucket 0.

Threshold: Youden's J statistic (argmax TPR - FPR on the ROC curve).

Checkpoint format (written by gtan_main_patched.py save_checkpoint):
  {checkpoint_dir}/{run_tag}_fold{fold}.pt           state dict (CPU)
  {checkpoint_dir}/{run_tag}_fold{fold}_meta.json    architecture sidecar

Outputs written to --output-dir:
  transfer_{tag}_metrics.txt    confusion matrix + all metrics
  transfer_{tag}_curves.npz     y_true / y_score arrays for plotting
  transfer_{tag}_curves.csv     same in CSV form

-----------------------------------------------------------------------
Usage — Direction A: trained on S-FFSD, validating against IEEE

  python -m methods.gtan.gtan_transfer \
    --checkpoint-dir  experiments/gtan_S-FFSD_baseline/checkpoints \
    --run-tag         gtan_S-FFSD \
    --fold            1 \
    --neofull-csv     data/ieee_pct15/S-FFSDneofull.csv \
    --fraud-pct       15 \
    --train-dataset   S-FFSD \
    --target-dataset  IEEE-CIS \
    --output-dir      experiments/transfer_SFFSD_to_IEEE \
    --device          cuda

Usage — Direction B: trained on IEEE, validating against S-FFSD

  python -m methods.gtan.gtan_transfer \
    --checkpoint-dir  experiments/gtan_IEEE_pct15/checkpoints \
    --run-tag         gtan_IEEE-CIS \
    --fold            1 \
    --neofull-csv     data/S-FFSDneofull.csv \
    --fraud-pct       native \
    --train-dataset   IEEE-CIS \
    --target-dataset  S-FFSD \
    --output-dir      experiments/transfer_IEEE_to_SFFSD \
    --device          cuda

Usage — sweep all folds x concentrations (PowerShell):

  foreach ($fold in 1,2,3,4,5) {
    foreach ($pct in 3,9,15,20,25) {
      python -m methods.gtan.gtan_transfer `
        --checkpoint-dir experiments/gtan_S-FFSD_baseline/checkpoints `
        --run-tag        gtan_S-FFSD `
        --fold           $fold `
        --neofull-csv    data/ieee_pct$pct/S-FFSDneofull.csv `
        --fraud-pct      $pct `
        --train-dataset  S-FFSD `
        --target-dataset IEEE-CIS `
        --output-dir     experiments/transfer_SFFSD_to_IEEE `
        --device         cuda
    }
  }
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

# ── Package import (handles both module and direct script execution) ──────────
try:
    from .gtan_model import GraphAttnModel
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from methods.gtan.gtan_model import GraphAttnModel


# ── Constants ─────────────────────────────────────────────────────────────────

# Categorical columns used by TransEmbedding — must match training config
CAT_COLS = ["Target", "Location", "Type"]

# Columns used for graph edge building — must match data_process_v3
PAIR_COLS = ["Source", "Target", "Location", "Type"]

EDGE_PER_TRANS = 3


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def load_meta(checkpoint_dir: str, run_tag: str, fold: int) -> dict:
    """
    Load the JSON architecture sidecar written by gtan_main save_checkpoint.
    Contains everything needed to reconstruct GraphAttnModel:
      in_feats, hid_dim, n_layers, heads, dropout, gated,
      cat_cols, cat_cardinalities, feature_columns.
    """
    path = os.path.join(checkpoint_dir, f"{run_tag}_fold{fold}_meta.json")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Meta file not found: {path}\n"
                 f"        Did training complete with checkpoint_dir set in args?")
    with open(path) as f:
        meta = json.load(f)
    print(f"[meta]  loaded : {path}")
    print(f"        trained on    : {meta.get('dataset', 'unknown')}")
    print(f"        in_feats      : {meta['in_feats']}")
    print(f"        n_layers      : {meta['n_layers']}")
    print(f"        cat_cols      : {meta['cat_cols']}")
    print(f"        cardinalities : {meta['cat_cardinalities']}")
    return meta


def load_state_dict(checkpoint_dir: str, run_tag: str, fold: int) -> dict:
    """Load the .pt state dict. Always saved on CPU by save_checkpoint."""
    path = os.path.join(checkpoint_dir, f"{run_tag}_fold{fold}.pt")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Checkpoint .pt not found: {path}")
    state = torch.load(path, map_location="cpu")
    print(f"[ckpt]  loaded : {path}  ({len(state)} weight tensors)")
    return state


# ── Graph building ─────────────────────────────────────────────────────────────

def build_graph(data: pd.DataFrame, edge_per_trans: int = EDGE_PER_TRANS) -> dgl.DGLGraph:
    """
    Rebuild the DGL transaction graph from an S-FFSDneofull DataFrame.
    Identical edge-building logic to data_process_v3 and load_gtan_data.
    Works for any dataset once it has been formatted to S-FFSD schema.
    """
    alls: List[int] = []
    allt: List[int] = []

    for column in PAIR_COLS:
        src: List[int] = []
        tgt: List[int] = []
        for _, c_df in data.groupby(column):
            c_df   = c_df.sort_values(by="Time")
            idxs   = c_df.index.to_list()
            df_len = len(idxs)
            for i in range(df_len):
                end = min(df_len, i + edge_per_trans)
                si  = idxs[i]
                for j in range(i, end):
                    src.append(si)
                    tgt.append(idxs[j])
        alls.extend(src)
        allt.extend(tgt)

    g = dgl.graph((np.array(alls, dtype=np.int64), np.array(allt, dtype=np.int64)))
    print(f"[graph] nodes={g.num_nodes():,}  edges={g.num_edges():,}")
    return g


# ── Model reconstruction ───────────────────────────────────────────────────────

def _dummy_ref_df(cat_cardinalities: Dict[str, int]) -> pd.DataFrame:
    """
    Minimal single-row DataFrame whose column values satisfy TransEmbedding's
    embedding size calculation: max(df[col].unique()) + 1 == cardinality.
    """
    return pd.DataFrame({
        col: [card - 1]
        for col, card in cat_cardinalities.items()
    })


def build_model(
    meta: dict,
    target_cat_cardinalities: Dict[str, int],
    state_dict: dict,
    strategy: str,
    device: str,
) -> GraphAttnModel:
    """
    Reconstruct GraphAttnModel and load weights.

    strategy='reinit'  (Option B, default)
      - Model built with TARGET dataset cardinalities
      - Checkpoint loaded strict=False
      - GNN / transformer / MLP / gate weights transfer exactly
      - Embedding lookup tables re-initialized (new vocabulary size)
      - Missing keys = the 3 cat_table embedding weight tensors (expected)
      - Best choice when target dataset has many new entities (e.g. IEEE card1)

    strategy='clamp'  (Option A)
      - Model built with TRAINING dataset cardinalities (from meta)
      - Checkpoint loaded strict=True (full weight transfer)
      - Caller must have already clamped target cat columns to [0, train_card-1]
      - OOV entities silently map to bucket 0
      - Best choice when vocabulary overlap is high
    """
    emb_cardinalities = (
        meta["cat_cardinalities"] if strategy == "clamp"
        else target_cat_cardinalities
    )

    ref_df   = _dummy_ref_df(emb_cardinalities)
    cat_feat = {col: torch.zeros(1, dtype=torch.long) for col in meta["cat_cols"]}

    model = GraphAttnModel(
        in_feats     = meta["in_feats"],
        hidden_dim   = meta["hid_dim"],
        n_classes    = meta["n_classes"],
        heads        = meta["heads"],
        activation   = nn.PReLU(),
        n_layers     = meta["n_layers"],
        drop         = meta["dropout"],
        device       = device,
        gated        = meta["gated"],
        ref_df       = ref_df,
        cat_features = cat_feat,
    )

    if strategy == "clamp":
        model.load_state_dict(state_dict, strict=True)
        print("[model] strict=True  — full weight transfer including embeddings")
    else:
        # Remove embedding weights so size mismatches don't block loading
        filtered = {
            k: v for k, v in state_dict.items()
            if "cat_table" not in k
        }
        result = model.load_state_dict(filtered, strict=False)
        print(f"[model] strict=False — GNN transfer, embeddings re-initialized")
        print(f"        missing keys    : {len(result.missing_keys)} "
              f"(embedding tables for {meta['cat_cols']})")
        print(f"        unexpected keys : {len(result.unexpected_keys)}")


# ── LPA subtensor (mirrors gtan_lpa.py, no package dependency needed) ─────────

def load_lpa_subtensor(
    node_feat: torch.Tensor,
    work_node_feat: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    seeds: torch.Tensor,
    input_nodes: torch.Tensor,
    device: str,
):
    batch_inputs      = node_feat[input_nodes].to(device)
    batch_work_inputs = {
        col: work_node_feat[col][input_nodes].to(device)
        for col in work_node_feat
        if col not in {"Labels"}
    }
    batch_labels     = labels[seeds].to(device)
    propagate_labels = labels[input_nodes].clone()
    # Mask all seeds as label=2 (unknown) — correct for pure inference
    propagate_labels[:seeds.shape[0]] = 2
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model: GraphAttnModel,
    graph: dgl.DGLGraph,
    feat_df: pd.DataFrame,
    cat_feat: Dict[str, torch.Tensor],
    labels_tensor: torch.Tensor,
    all_idx: List[int],
    device: str,
    batch_size: int,
    n_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass over all nodes in the target dataset.
    All nodes are treated as test nodes (no training happening here).

    Returns
    -------
    y_true  : int array (0/1 ground truth)
    y_score : float array (softmax probability of class 1 / fraud)
    """
    graph    = graph.to(device)
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_dev  = {col: t.to(device) for col, t in cat_feat.items()}

    n_nodes    = len(feat_df)
    all_logits = torch.zeros(n_nodes, 2, dtype=torch.float32)

    test_ind   = torch.from_numpy(np.array(all_idx, dtype=np.int64)).long().to(device)
    sampler    = MultiLayerFullNeighborSampler(n_layers)
    dataloader = NodeDataLoader(
        graph, test_ind, sampler,
        use_ddp=False, device=device,
        batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )

    model.eval()
    with torch.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = \
                load_lpa_subtensor(num_feat, cat_dev, labels_tensor,
                                   seeds, input_nodes, device)
            blocks  = [b.to(device) for b in blocks]
            logits  = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
            all_logits[seeds.cpu()] = logits.cpu()
            if step % 20 == 0:
                print(f"  inference batch {step:04d} / "
                      f"{len(dataloader):04d}")

    y_true  = labels_tensor[all_idx].cpu().numpy()
    y_score = torch.softmax(all_logits, dim=1)[:, 1].numpy()

    # Drop any label=2 rows (LPA mask artifact — shouldn't appear in a clean CSV)
    mask    = y_true != 2
    return y_true[mask].astype(np.int64), y_score[mask]


# ── Threshold selection ────────────────────────────────────────────────────────

def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Youden's J optimal threshold: argmax(TPR - FPR) over the ROC curve.
    Finds the cutoff that maximises the sum of sensitivity and specificity,
    which is the correct tradeoff for imbalanced fraud detection.
    Falls back to 0.5 if the ROC curve cannot be computed.
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        best_idx = int(np.argmax(tpr - fpr))
        return float(thresholds[best_idx])
    except Exception as e:
        print(f"[WARNING] Youden threshold failed ({e}); using 0.5 fallback")
        return 0.5


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute the full metric set used across all experiment groups.
    y_pred is derived from y_score using the Youden threshold.
    """
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0

    try:
        f1_macro = f1_score(y_true, y_pred, average="macro")
    except Exception:
        f1_macro = float("nan")

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float("nan")

    return {
        "threshold": threshold,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "recall":    recall,
        "precision": precision,
        "f1_macro":  f1_macro,
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
        "n":         int(len(y_true)),
        "n_fraud":   int(y_true.sum()),
        "fraud_pct": float(y_true.mean() * 100),
        # keep y_pred internally for artifact writing
        "_y_pred":   y_pred,
    }


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(m: dict, tag: str) -> None:
    print()
    print("=" * 60)
    print(f"  Transfer Results")
    print(f"  {tag}")
    print("=" * 60)
    print(f"  n={m['n']:,}   fraud={m['n_fraud']:,} ({m['fraud_pct']:.2f}%)")
    print(f"  Youden threshold : {m['threshold']:.4f}")
    print()
    print(f"  {'':14s}  Pred 0       Pred 1")
    print(f"  {'Actual 0':14s}  TN={m['TN']:>8,}   FP={m['FP']:>8,}")
    print(f"  {'Actual 1':14s}  FN={m['FN']:>8,}   TP={m['TP']:>8,}")
    print()
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  F1 (macro): {m['f1_macro']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"  PR-AUC    : {m['pr_auc']:.4f}")
    print("=" * 60)


# ── Artifact writing ───────────────────────────────────────────────────────────

def write_artifacts(
    output_dir: str,
    file_tag: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Curves — used by compare_experiments.py and any external plotting
    npz_path = os.path.join(output_dir, f"{file_tag}_curves.npz")
    np.savez_compressed(npz_path, y_true=y_true, y_score=y_score)

    csv_path = os.path.join(output_dir, f"{file_tag}_curves.csv")
    pd.DataFrame({"y_true": y_true, "y_prob": y_score}).to_csv(csv_path, index=False)

    # Human-readable metrics — matches format of existing metric_inputs .txt files
    txt_path = os.path.join(output_dir, f"{file_tag}_metrics.txt")
    with open(txt_path, "w") as f:
        for key, val in metrics.items():
            if key == "_y_pred":
                continue   # skip the array
            f.write(f"{key:<14}: {val}\n")

    print(f"\n[artifacts] {npz_path}")
    print(f"[artifacts] {csv_path}")
    print(f"[artifacts] {txt_path}")


# ── Optional Comet logging ─────────────────────────────────────────────────────

def log_to_comet(
    metrics: dict,
    meta: dict,
    args: argparse.Namespace,
    file_tag: str,
) -> None:
    try:
        from comet_ml import Experiment
    except ImportError:
        print("[comet] comet_ml not installed — skipping")
        return

    api_key = os.environ.get("COMET_API_KEY", "")
    if not api_key:
        print("[comet] COMET_API_KEY not set — skipping")
        return

    exp = Experiment(
        api_key=api_key,
        project_name=os.environ.get("COMET_PROJECT", "ece-thesis-fraud"),
        workspace=os.environ.get("COMET_WORKSPACE", ""),
    )
    exp.set_name(file_tag)
    exp.add_tag("transfer")
    exp.add_tag(f"strategy={args.strategy}")
    exp.add_tag(f"fold={args.fold}")
    exp.add_tag(f"train={args.train_dataset}")
    exp.add_tag(f"target={args.target_dataset}")
    if args.fraud_pct is not None:
        exp.add_tag(f"fraud_pct={args.fraud_pct}")

    exp.log_parameters({
        "train_run_tag":   meta["run_tag"],
        "train_dataset":   args.train_dataset,
        "target_dataset":  args.target_dataset,
        "fold":            args.fold,
        "strategy":        args.strategy,
        "fraud_pct":       args.fraud_pct,
        "neofull_csv":     args.neofull_csv,
    })

    scalar_keys = [
        "threshold", "TP", "FP", "TN", "FN",
        "recall", "precision", "f1_macro",
        "roc_auc", "pr_auc",
        "n", "n_fraud", "fraud_pct",
    ]
    exp.log_metrics({k: metrics[k] for k in scalar_keys})
    exp.end()
    print(f"[comet] logged: {file_tag}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transfer a GTAN checkpoint to a different dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Checkpoint
    g = p.add_argument_group("Checkpoint")
    g.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing {run_tag}_fold{fold}.pt + _meta.json")
    g.add_argument("--run-tag", required=True,
                   help="run_tag used when training (e.g. gtan_S-FFSD or gtan_IEEE-CIS)")
    g.add_argument("--fold", type=int, required=True,
                   help="Which fold checkpoint to load (1-based, matches training output)")

    # ── Target dataset
    g = p.add_argument_group("Target Dataset")
    g.add_argument("--neofull-csv", required=True,
                   help="S-FFSDneofull.csv produced by data_process_v3.py for the TARGET dataset")
    g.add_argument("--train-dataset",  default="S-FFSD",
                   help="Name of the dataset used for training (label only, e.g. S-FFSD or IEEE-CIS)")
    g.add_argument("--target-dataset", default="IEEE-CIS",
                   help="Name of the dataset being validated against (e.g. IEEE-CIS or S-FFSD)")
    g.add_argument("--fraud-pct", default=None,
                   help="Fraud concentration label for artifact naming (e.g. 15 for 15%%, or 'native')")

    # ── Transfer strategy
    g = p.add_argument_group("Transfer Strategy")
    g.add_argument("--strategy", choices=["reinit", "clamp"], default="reinit",
                   help="reinit (default): fresh embeddings for target vocab; "
                        "clamp: clip target IDs to training vocab bounds")

    # ── Inference
    g = p.add_argument_group("Inference")
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--device", default="cpu",
                   help="cuda or cpu")
    g.add_argument("--edge-per-trans", type=int, default=EDGE_PER_TRANS,
                   help="Forward edges per transaction group (default=3, must match training)")

    # ── Output
    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", required=True,
                   help="Directory to write metrics txt, curves npz/csv")
    g.add_argument("--comet", action="store_true",
                   help="Log to Comet.ml (requires COMET_API_KEY env var and COMET_PROJECT)")

    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    t0   = time.time()
    args = parse_args()

    pct_label = str(args.fraud_pct) if args.fraud_pct is not None else "native"

    print("=" * 60)
    print("gtan_transfer.py")
    print(f"  direction      : {args.train_dataset} → {args.target_dataset}")
    print(f"  checkpoint-dir : {args.checkpoint_dir}")
    print(f"  run-tag        : {args.run_tag}  fold={args.fold}")
    print(f"  neofull-csv    : {args.neofull_csv}")
    print(f"  fraud-pct      : {pct_label}")
    print(f"  strategy       : {args.strategy}")
    print(f"  device         : {args.device}")
    print("=" * 60)

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    meta       = load_meta(args.checkpoint_dir, args.run_tag, args.fold)
    state_dict = load_state_dict(args.checkpoint_dir, args.run_tag, args.fold)

    # ── 2. Load target dataset ────────────────────────────────────────────────
    print(f"\n[data] loading {args.neofull_csv} ...")
    if not os.path.exists(args.neofull_csv):
        sys.exit(f"[ERROR] File not found: {args.neofull_csv}")

    data = pd.read_csv(args.neofull_csv, low_memory=False)
    data = data.loc[:, ~data.columns.str.contains("Unnamed")]

    required = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]
    missing  = [c for c in required if c not in data.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns in neofull CSV: {missing}")

    # Filter to clean labels (0 = legit, 1 = fraud)
    data = data[data["Labels"] <= 1].reset_index(drop=True)
    print(f"[data] rows={len(data):,}  "
          f"fraud={int(data['Labels'].sum()):,}  "
          f"({data['Labels'].mean()*100:.2f}%)")

    # ── 3. Validate feature dimension ─────────────────────────────────────────
    # in_feats must match — both datasets must have been run through
    # data_process_v3.py with the same time windows.
    feat_cols = [c for c in data.columns if c != "Labels"]
    if len(feat_cols) != meta["in_feats"]:
        print(f"\n[WARNING] Feature dimension mismatch!")
        print(f"          Checkpoint expects : {meta['in_feats']} features")
        print(f"          Target CSV has     : {len(feat_cols)} features")
        print(f"          Both datasets must be processed with identical")
        print(f"          data_process_v3.py settings (time windows, etc.)")
        print(f"          Inference will crash at the first forward pass.\n")

    # ── 4. Encode categoricals ────────────────────────────────────────────────
    # LabelEncoder maps string/integer entity IDs to a contiguous 0-based range.
    # Source is encoded for graph building; Target/Location/Type for embeddings.
    for col in PAIR_COLS:
        le       = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)

    cat_cols = meta["cat_cols"]   # ["Target", "Location", "Type"]

    # ── 5. Handle embedding cardinality mismatch ──────────────────────────────
    target_cardinalities = {col: int(data[col].max()) + 1 for col in cat_cols}

    if args.strategy == "clamp":
        print("\n[clamp] clamping categorical columns to training vocab bounds ...")
        for col in cat_cols:
            train_card = meta["cat_cardinalities"][col]
            n_oov      = int((data[col] >= train_card).sum())
            if n_oov > 0:
                print(f"  {col}: {n_oov:,} OOV IDs → bucket 0  "
                      f"(train_card={train_card}, target_max={data[col].max()})")
            data[col] = data[col].clip(upper=train_card - 1)
        # After clamping, force cardinalities to match training exactly
        target_cardinalities = {col: meta["cat_cardinalities"][col] for col in cat_cols}

    else:  # reinit
        print("\n[reinit] embedding cardinality comparison:")
        for col in cat_cols:
            tc    = target_cardinalities[col]
            mc    = meta["cat_cardinalities"][col]
            delta = tc - mc
            sign  = "+" if delta >= 0 else ""
            print(f"  {col:<12} train={mc:>6,}  target={tc:>6,}  "
                  f"delta={sign}{delta:,}")

    # ── 6. Build DGL graph ────────────────────────────────────────────────────
    print("\n[graph] building transaction graph ...")
    graph = build_graph(data, edge_per_trans=args.edge_per_trans)

    # ── 7. Prepare tensors ────────────────────────────────────────────────────
    feat_df       = data.drop("Labels", axis=1)
    labels_tensor = torch.from_numpy(data["Labels"].to_numpy()).long()

    graph.ndata["label"] = labels_tensor
    graph.ndata["feat"]  = torch.from_numpy(feat_df.to_numpy()).float()

    cat_feat = {
        col: torch.from_numpy(feat_df[col].to_numpy()).long()
        for col in cat_cols
    }

    # ── 8. Build model ────────────────────────────────────────────────────────
    print(f"\n[model] reconstructing GraphAttnModel ...")
    model = build_model(
        meta                    = meta,
        target_cat_cardinalities = target_cardinalities,
        state_dict              = state_dict,
        strategy                = args.strategy,
        device                  = args.device,
    )

    # ── 9. Run inference ──────────────────────────────────────────────────────
    print("\n[inference] running forward pass over all target nodes ...")
    y_true, y_score = run_inference(
        model         = model,
        graph         = graph,
        feat_df       = feat_df,
        cat_feat      = cat_feat,
        labels_tensor = labels_tensor,
        all_idx       = list(range(len(feat_df))),
        device        = args.device,
        batch_size    = args.batch_size,
        n_layers      = meta["n_layers"],
    )

    # ── 10. Threshold + metrics ───────────────────────────────────────────────
    threshold = youden_threshold(y_true, y_score)
    metrics   = compute_metrics(y_true, y_score, threshold)

    # File tag encodes the full experiment context
    file_tag = (
        f"transfer"
        f"_from{args.train_dataset.replace('-','')}"
        f"_to{args.target_dataset.replace('-','')}"
        f"_fold{args.fold}"
        f"_pct{pct_label}"
        f"_{args.strategy}"
    )

    print_report(metrics, file_tag)

    # ── 11. Write artifacts ───────────────────────────────────────────────────
    write_artifacts(
        output_dir = args.output_dir,
        file_tag   = file_tag,
        y_true     = y_true,
        y_score    = y_score,
        metrics    = metrics,
    )

    # ── 12. Comet ─────────────────────────────────────────────────────────────
    if args.comet:
        log_to_comet(metrics, meta, args, file_tag)

    print(f"\n[done] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()