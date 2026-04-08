#!/usr/bin/env python3
"""
run_gtan_with_checkpoints.py

Drop-in wrapper around the original gtan_main() that adds per-fold checkpoint
saving without touching methods/gtan/gtan_main.py.

Usage (EC2):
  python run_gtan_with_checkpoints.py \
      --experiment-id  B3 \
      --input-csv      data/ieee_B3/S-FFSDneofull.csv \
      --checkpoint-dir experiments/gtan_S-FFSD_B3/checkpoints \
      --fraud-pct      15 \
      --device         cuda \
      --epochs         25 \
      --seed           42
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Import originals untouched ────────────────────────────────────────────────
from methods.gtan.gtan_main  import gtan_main, load_gtan_data
from methods.gtan.gtan_model import GraphAttnModel

# Optional Comet
try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None


# ── Checkpoint saving (was in the patch — now lives here instead) ─────────────

def save_checkpoint(model, feat_df, cat_feat_dict, args, fold, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{args['method']}_{args['dataset']}"

    cat_cardinalities = {
        col: int(feat_df[col].max()) + 1
        for col in cat_feat_dict.keys()
    }

    meta = {
        "run_tag":           run_tag,
        "fold":              fold,
        "in_feats":          int(feat_df.shape[1]),
        "hid_dim":           int(args["hid_dim"] // 4),
        "n_layers":          int(args["n_layers"]),
        "n_classes":         2,
        "heads":             [4] * int(args["n_layers"]),
        "dropout":           args["dropout"],
        "gated":             bool(args["gated"]),
        "cat_cols":          list(cat_feat_dict.keys()),
        "cat_cardinalities": cat_cardinalities,
        "dataset":           args["dataset"],
        "method":            args["method"],
        "feature_columns":   list(feat_df.columns),
    }

    meta_path = os.path.join(output_dir, f"{run_tag}_fold{fold}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    pt_path = os.path.join(output_dir, f"{run_tag}_fold{fold}.pt")
    torch.save(model.cpu().state_dict(), pt_path)
    model.to(args["device"])

    print(f"[checkpoint] fold {fold} → {pt_path}")
    print(f"[checkpoint] meta     → {meta_path}")
    return pt_path, meta_path


# ── Monkey-patch: intercept early_stopper to capture best model per fold ──────

def make_checkpointing_gtan_main(checkpoint_dir):
    """
    Returns a version of gtan_main that saves a checkpoint after each fold,
    by wrapping the early_stopper class used inside gtan_main.

    The original gtan_main is called unmodified — we only wrap the
    early_stopper that it imports from its own package __init__.
    """
    import methods.gtan as gtan_pkg

    # Save a reference to the original early_stopper
    OriginalEarlyStopper = gtan_pkg.early_stopper

    # We need to capture feat_df and cat_feat from gtan_main's local scope.
    # The cleanest hook is to wrap gtan_main itself, re-building the cat_feat
    # dict from feat_df (which is passed in as an argument).
    def gtan_main_with_checkpoints(
        feat_df, graph, train_idx, test_idx, labels, args, cat_features, experiment=None
    ):
        # Rebuild cat_feat the same way gtan_main does internally
        cat_feat = {
            col: torch.from_numpy(feat_df[col].values).long()
            for col in cat_features
        }

        fold_counter = [0]  # mutable closure counter

        class CheckpointingEarlyStopper(OriginalEarlyStopper):
            def earlystop(self, loss, model):
                super().earlystop(loss, model)
                # best_model is updated inside super().earlystop when loss improves
                # We save after every call; save_checkpoint overwrites the same
                # fold file, so only the final best for that fold persists.
                if self.best_model is not None:
                    save_checkpoint(
                        model=self.best_model,
                        feat_df=feat_df,
                        cat_feat_dict=cat_feat,
                        args=args,
                        fold=fold_counter[0],
                        output_dir=checkpoint_dir,
                    )

        # Temporarily replace early_stopper in the gtan package namespace
        gtan_pkg.early_stopper = CheckpointingEarlyStopper

        # Patch fold counter: wrap kfold inside gtan_main by incrementing
        # before each fold via a StratifiedKFold wrapper.
        from sklearn.model_selection import StratifiedKFold
        OriginalKFold = StratifiedKFold

        class CountingKFold(OriginalKFold):
            def split(self, X, y=None, groups=None):
                for split in super().split(X, y, groups):
                    fold_counter[0] += 1
                    yield split

        import sklearn.model_selection as skl_model_selection
        skl_model_selection.StratifiedKFold = CountingKFold

        try:
            # Call original gtan_main completely unmodified
            result = gtan_main(
                feat_df, graph, train_idx, test_idx,
                labels, args, cat_features, experiment
            )
        finally:
            # Always restore originals, even if training crashes
            gtan_pkg.early_stopper = OriginalEarlyStopper
            skl_model_selection.StratifiedKFold = OriginalKFold

        return result

    return gtan_main_with_checkpoints


# ── Data loading for S-FFSD (mirrors load_gtan_data without the S-FFSD split) -

def load_sffsd(neofull_csv: str):
    """Load a pre-processed S-FFSDneofull.csv and return what gtan_main needs."""
    cat_features = ["Target", "Location", "Type"]
    pair_cols    = ["Source", "Target", "Location", "Type"]

    df   = pd.read_csv(neofull_csv, low_memory=False)
    df   = df.loc[:, ~df.columns.str.contains("Unnamed")]
    data = df[df["Labels"] <= 2].reset_index(drop=True)

    alls, allt = [], []
    for column in pair_cols:
        src, tgt = [], []
        for _, c_df in data.groupby(column):
            c_df       = c_df.sort_values(by="Time")
            idxs       = c_df.index.to_list()
            df_len     = len(idxs)
            for i in range(df_len):
                end = min(df_len, i + 3)
                for j in range(i, end):
                    src.append(idxs[i])
                    tgt.append(idxs[j])
        alls.extend(src)
        allt.extend(tgt)

    g = dgl.graph((np.array(alls, dtype=np.int64), np.array(allt, dtype=np.int64)))

    for col in pair_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)

    feat_data = data.drop("Labels", axis=1)
    labels    = data["Labels"]

    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).long()
    g.ndata["feat"]  = torch.from_numpy(feat_data.to_numpy()).float()

    index = list(range(len(labels)))
    train_idx, test_idx, _, _ = train_test_split(
        index, labels, stratify=labels,
        test_size=0.1, random_state=2, shuffle=True
    )

    return feat_data, labels, train_idx, test_idx, g, cat_features


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id",  required=True,
                   help="Short label for this run, e.g. B3 or baseline")
    p.add_argument("--input-csv",      required=True,
                   help="Path to S-FFSDneofull.csv (already feature-engineered)")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Where to write fold checkpoints")
    p.add_argument("--fraud-pct",      type=float, default=None,
                   help="Fraud concentration label for logging (e.g. 15)")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--epochs",         type=int, default=25)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--n-folds",        type=int, default=5)
    p.add_argument("--batch-size",     type=int, default=1024)
    p.add_argument("--hid-dim",        type=int, default=256)
    p.add_argument("--n-layers",       type=int, default=3)
    p.add_argument("--comet-key",      default=None,
                   help="Comet API key (or set COMET_API_KEY env var)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[run] experiment   : {args.experiment_id}")
    print(f"[run] input-csv    : {args.input_csv}")
    print(f"[run] checkpoint   : {args.checkpoint_dir}")
    print(f"[run] fraud-pct    : {args.fraud_pct}")
    print(f"[run] device       : {args.device}")
    print(f"[run] epochs       : {args.epochs}")

    # ── Optional Comet experiment ─────────────────────────────────────────────
    experiment = None
    api_key    = args.comet_key or os.environ.get("COMET_API_KEY", "")
    if api_key and Experiment:
        experiment = Experiment(
            api_key=api_key,
            project_name=os.environ.get("COMET_PROJECT", "ece-thesis-fraud"),
        )
        experiment.set_name(f"gtan_S-FFSD_{args.experiment_id}")
        if args.fraud_pct:
            experiment.add_tag(f"fraud_pct={args.fraud_pct}")
        experiment.add_tag(f"exp={args.experiment_id}")

    # ── Load data ─────────────────────────────────────────────────────────────
    feat_df, labels, train_idx, test_idx, graph, cat_features = load_sffsd(args.input_csv)

    # ── Build args dict (same structure gtan_main expects) ────────────────────
    gtan_args = {
        "device":         args.device,
        "n_fold":         args.n_folds,
        "n_layers":       args.n_layers,
        "hid_dim":        args.hid_dim,
        "dropout":        [0.2, 0.2],
        "gated":          True,
        "lr":             0.003,
        "wd":             1e-5,
        "batch_size":     args.batch_size,
        "early_stopping": 10,
        "max_epochs":     args.epochs,
        "seed":           args.seed,
        "method":         "gtan",
        "dataset":        "S-FFSD",
    }

    # ── Run with checkpointing wrapper ────────────────────────────────────────
    wrapped_gtan_main = make_checkpointing_gtan_main(args.checkpoint_dir)

    wrapped_gtan_main(
        feat_df, graph, train_idx, test_idx,
        labels, gtan_args, cat_features, experiment
    )

    if experiment:
        experiment.end()


if __name__ == "__main__":
    main()