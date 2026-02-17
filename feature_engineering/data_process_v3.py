#!/usr/bin/env python3
"""
data_process.py (V3 + Status Logging)

Canonical S-FFSD pipeline (NO IEEE-specific logic) that works with ANY input CSV that is already
in S-FFSD schema:

  Time, Source, Target, Amount, Location, Type, Labels

Key goals:
- Preserve GTAN/RGTAN naming conventions (same filenames as your original pipeline):
    - S-FFSDneofull.csv
    - graph-S-FFSD.bin
    - S-FFSD_neigh_feat.csv
- Add CLI flags:
    --dataset S-FFSD
    --input-file PATH
    --output-dir PATH
- Optimize for 500K+ rows:
    - Temporal feature engineering rewritten from O(N^2) to O(N * W) using prefix sums + sliding windows
    - Neighbor features computed via DGL message passing (scales), NOT per-node khop subgraphs
- Add robust status logging:
    - Console + file logging (data_process.log)
    - Phase timings
    - Row/node/edge counts
    - Optional memory usage (psutil if available)

Recommended for 500K+:
- If you want fastest runtime:
    --skip-temporal-fe --skip-neigh-feat
- If you want temporal features but still scalable:
    (default) leave temporal on, unique-counts OFF
- Avoid --unique-counts on 500K+ unless you accept heavy runtime.

Example:
  python data_process.py --dataset S-FFSD \
    --input-file /path/to/ieee_experiment1.csv \
    --output-dir  /path/to/out/E1

"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import dgl
import dgl.function as fn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional memory logging
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Default data directory (same spirit as your original)
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Required S-FFSD schema
SFFSD_REQUIRED_COLS = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]

# Output naming conventions (MUST MATCH GTAN/RGTAN expectations)
OUT_NEOFULL = "S-FFSDneofull.csv"
OUT_GRAPH = "graph-S-FFSD.bin"
OUT_NEIGH_FEAT = "S-FFSD_neigh_feat.csv"
OUT_LOG = "data_process.log"


# -----------------------------
# Logging helpers
# -----------------------------
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("SFFSD")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on reruns

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, OUT_LOG))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_memory(logger: logging.Logger) -> None:
    if psutil is None:
        return
    try:
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logger.info(f"Memory RSS: {rss:.2f} MB")
    except Exception:
        # Never let memory logging crash the run
        return


class PhaseTimer:
    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"START: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            self.logger.info(f"END  : {self.name} (elapsed: {elapsed:.2f}s)")
        else:
            self.logger.error(f"FAIL : {self.name} (elapsed: {elapsed:.2f}s) -> {exc}")
        # don't suppress exceptions
        return False


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------
# Schema validation
# -----------------------------
def _assert_sffsd_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Case-insensitive schema validation; returns canonical->actual mapping."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in SFFSD_REQUIRED_COLS if c.lower() not in lookup]
    if missing:
        raise ValueError(
            f"Input is not S-FFSD-shaped. Missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return {c: lookup[c.lower()] for c in SFFSD_REQUIRED_COLS}


# -----------------------------
# V3 Temporal Feature Engineering (FAST)
# -----------------------------
def featmap_gen_v3(
    df: pd.DataFrame,
    logger: logging.Logger,
    time_windows: List[int] | None = None,
    compute_unique_counts: bool = False,
) -> pd.DataFrame:
    """
    Fast temporal feature engineering.

    Included (fast, scalable):
      trans_at_avg_*
      trans_at_totl_*
      trans_at_std_*
      trans_at_bias_*
      trans_at_num_*

    Optional (slow, default OFF):
      trans_target_num_*
      trans_location_num_*
      trans_type_num_*
    """
    if time_windows is None:
        # Same spirit as original list; you can tune these.
        time_windows = [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]

    df = df.copy()

    # Ensure numeric
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(np.int64)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).astype(np.float64)

    # Sort by Time for sliding windows
    with PhaseTimer(logger, "Sort by Time (temporal FE)"):
        df = df.sort_values("Time").reset_index(drop=True)

    t = df["Time"].to_numpy(dtype=np.int64)
    amt = df["Amount"].to_numpy(dtype=np.float64)
    n = len(df)

    # Prefix sums
    with PhaseTimer(logger, "Prefix sums (Amount, Amount^2)"):
        pref = np.zeros(n + 1, dtype=np.float64)
        pref2 = np.zeros(n + 1, dtype=np.float64)
        pref[1:] = np.cumsum(amt)
        pref2[1:] = np.cumsum(amt * amt)

    # Pre-create columns
    for L in time_windows:
        name = str(L)
        df[f"trans_at_totl_{name}"] = 0.0
        df[f"trans_at_avg_{name}"] = 0.0
        df[f"trans_at_std_{name}"] = 0.0
        df[f"trans_at_bias_{name}"] = 0.0
        df[f"trans_at_num_{name}"] = 0

        # unique counts: default 0 unless enabled
        df[f"trans_target_num_{name}"] = 0
        df[f"trans_location_num_{name}"] = 0
        df[f"trans_type_num_{name}"] = 0

    # Sliding window pointers per L
    left_ptrs = {L: 0 for L in time_windows}

    with PhaseTimer(logger, f"Temporal FE O(N*W) (rows={n:,}, windows={len(time_windows)})"):
        # Use plain loops (fast enough); avoid df.at inside inner loop by writing to numpy buffers
        # Buffers per window to avoid heavy pandas scalar writes
        buffers = {}
        for L in time_windows:
            buffers[L] = {
                "sum": np.zeros(n, dtype=np.float64),
                "mean": np.zeros(n, dtype=np.float64),
                "std": np.zeros(n, dtype=np.float64),
                "bias": np.zeros(n, dtype=np.float64),
                "cnt": np.zeros(n, dtype=np.int64),
            }

        for i in range(n):
            ti = t[i]
            ai = amt[i]
            for L in time_windows:
                l = left_ptrs[L]
                boundary = ti - L
                while l < i and t[l] < boundary:
                    l += 1
                left_ptrs[L] = l

                cnt = i - l + 1
                s = pref[i + 1] - pref[l]
                s2 = pref2[i + 1] - pref2[l]
                mean = s / cnt
                var = (s2 / cnt) - (mean * mean)
                std = np.sqrt(var) if var > 0 else 0.0
                bias = ai - mean

                buffers[L]["sum"][i] = s
                buffers[L]["mean"][i] = mean
                buffers[L]["std"][i] = std
                buffers[L]["bias"][i] = bias
                buffers[L]["cnt"][i] = cnt

            if (i + 1) % 100000 == 0:
                logger.info(f"Temporal FE progress: {i+1:,}/{n:,} rows")
                log_memory(logger)

        # Write buffers into dataframe columns
        for L in time_windows:
            name = str(L)
            df[f"trans_at_totl_{name}"] = buffers[L]["sum"]
            df[f"trans_at_avg_{name}"] = buffers[L]["mean"]
            df[f"trans_at_std_{name}"] = buffers[L]["std"]
            df[f"trans_at_bias_{name}"] = buffers[L]["bias"]
            df[f"trans_at_num_{name}"] = buffers[L]["cnt"]

    if compute_unique_counts:
        # WARNING: exact uniques are expensive on 500K+. Use only if you accept heavy runtime.
        with PhaseTimer(logger, "Exact unique-count temporal features (SLOW)"):
            tgt = df["Target"].astype("string").fillna("").to_numpy()
            loc = df["Location"].astype("string").fillna("").to_numpy()
            typ = df["Type"].astype("string").fillna("").to_numpy()

            left_ptrs = {L: 0 for L in time_windows}
            for i in range(n):
                ti = t[i]
                for L in time_windows:
                    l = left_ptrs[L]
                    boundary = ti - L
                    while l < i and t[l] < boundary:
                        l += 1
                    left_ptrs[L] = l

                    name = str(L)
                    df.at[i, f"trans_target_num_{name}"] = int(len(set(tgt[l : i + 1])))
                    df.at[i, f"trans_location_num_{name}"] = int(len(set(loc[l : i + 1])))
                    df.at[i, f"trans_type_num_{name}"] = int(len(set(typ[l : i + 1])))

                if (i + 1) % 50000 == 0:
                    logger.info(f"Unique-count FE progress: {i+1:,}/{n:,} rows")
                    log_memory(logger)

    return df


# -----------------------------
# Neighbor features (FAST message passing)
# -----------------------------
def generate_neigh_features_fast(graph: dgl.DGLGraph, logger: logging.Logger) -> pd.DataFrame:
    """
    Scalable neighbor features via message passing.

    Produces columns:
      degree, riskstat, 1hop_degree, 2hop_degree, 1hop_riskstat, 2hop_riskstat

    Notes:
      - degree: in-degree
      - riskstat: sum of fraud labels over OUT neighbors (direct successors)
      - 1hop_degree: sum of out-degree of OUT neighbors
      - 2hop_degree: sum of 1hop_degree of OUT neighbors
      - 1hop_riskstat: sum of riskstat of OUT neighbors
      - 2hop_riskstat: sum of 1hop_riskstat of OUT neighbors
    """
    with PhaseTimer(logger, "Neighbor features (message passing)"):
        # degree features
        deg_in = graph.in_degrees().to(torch.float32)

        # riskstat: sum labels over OUT neighbors
        graph.ndata["label_f"] = graph.ndata["label"].to(torch.float32)
        graph.update_all(fn.copy_u("label_f", "m"), fn.sum("m", "riskstat_out"))
        riskstat = graph.ndata["riskstat_out"].to(torch.float32)

        # 1-hop degree: sum out-degree of neighbors
        graph.ndata["deg_out"] = graph.out_degrees().to(torch.float32)
        graph.update_all(fn.copy_u("deg_out", "m"), fn.sum("m", "deg_1hop"))
        deg_1hop = graph.ndata["deg_1hop"].to(torch.float32)

        # 2-hop degree: aggregate deg_1hop over neighbors
        graph.ndata["deg_1hop"] = deg_1hop
        graph.update_all(fn.copy_u("deg_1hop", "m"), fn.sum("m", "deg_2hop"))
        deg_2hop = graph.ndata["deg_2hop"].to(torch.float32)

        # 1-hop riskstat: aggregate riskstat over neighbors
        graph.ndata["riskstat"] = riskstat
        graph.update_all(fn.copy_u("riskstat", "m"), fn.sum("m", "risk_1hop"))
        risk_1hop = graph.ndata["risk_1hop"].to(torch.float32)

        # 2-hop riskstat: aggregate risk_1hop over neighbors
        graph.ndata["risk_1hop"] = risk_1hop
        graph.update_all(fn.copy_u("risk_1hop", "m"), fn.sum("m", "risk_2hop"))
        risk_2hop = graph.ndata["risk_2hop"].to(torch.float32)

        feats = torch.stack(
            [deg_in, riskstat, deg_1hop, deg_2hop, risk_1hop, risk_2hop], dim=1
        ).cpu().numpy()
        feats[np.isnan(feats)] = 0.0

        cols = ["degree", "riskstat", "1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
        return pd.DataFrame(feats, columns=cols)


# -----------------------------
# CLI / Main pipeline
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical S-FFSD data_process V3 (optimized + status logging)."
    )
    p.add_argument(
        "--dataset",
        default="S-FFSD",
        choices=["S-FFSD"],
        help="Must be S-FFSD (your IEEE experiments must already be S-FFSD-shaped CSVs).",
    )
    p.add_argument(
        "--input-file",
        default=os.path.join(DATADIR, "S-FFSD.csv"),
        help="Path to S-FFSD-shaped CSV.",
    )
    p.add_argument(
        "--output-dir",
        default=DATADIR,
        help="Directory to write outputs (filenames preserved).",
    )
    p.add_argument(
        "--edge-per-trans",
        type=int,
        default=3,
        help="Forward edges per transaction within each entity group (default=3).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--skip-temporal-fe",
        action="store_true",
        help="Skip temporal feature engineering entirely (fastest).",
    )
    p.add_argument(
        "--unique-counts",
        action="store_true",
        help="Compute exact unique-count temporal features (VERY slow at 500K+).",
    )
    p.add_argument(
        "--skip-neigh-feat",
        action="store_true",
        help="Skip writing S-FFSD_neigh_feat.csv.",
    )
    return p.parse_args()


def run_sffsd_pipeline_v3(
    input_file: str,
    output_dir: str,
    edge_per_trans: int,
    skip_temporal_fe: bool,
    unique_counts: bool,
    skip_neigh_feat: bool,
) -> None:
    logger = setup_logger(output_dir)
    t0 = time.time()

    logger.info("============================================")
    logger.info("START: Canonical S-FFSD processing (V3)")
    logger.info(f"input_file      : {input_file}")
    logger.info(f"output_dir      : {output_dir}")
    logger.info(f"edge_per_trans  : {edge_per_trans}")
    logger.info(f"skip_temporal_fe: {skip_temporal_fe}")
    logger.info(f"unique_counts   : {unique_counts}")
    logger.info(f"skip_neigh_feat : {skip_neigh_feat}")
    logger.info("============================================")
    log_memory(logger)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    with PhaseTimer(logger, "Load CSV"):
        data = pd.read_csv(input_file, low_memory=False)
        logger.info(f"Rows loaded: {len(data):,}")
        colmap = _assert_sffsd_schema(data)
        # Normalize column names to canonical (Time/Source/Target/Amount/Location/Type/Labels)
        data = data.rename(columns={v: k for k, v in colmap.items()})

    log_memory(logger)

    # Temporal features (fast)
    if not skip_temporal_fe:
        data = featmap_gen_v3(data.reset_index(drop=True), logger=logger, compute_unique_counts=unique_counts)
        data.replace(np.nan, 0, inplace=True)
    else:
        logger.info("Skipping temporal feature engineering (--skip-temporal-fe set).")

    # Save intermediate (preserve name)
    neo_full_path = os.path.join(output_dir, OUT_NEOFULL)
    with PhaseTimer(logger, f"Write {OUT_NEOFULL}"):
        data.to_csv(neo_full_path, index=False)
        logger.info(f"Wrote: {neo_full_path}")

    # Reload (preserve original behavior)
    with PhaseTimer(logger, f"Reload {OUT_NEOFULL}"):
        data = pd.read_csv(neo_full_path, low_memory=False).reset_index(drop=True)

    # Build temporal edges grouped by entity column
    with PhaseTimer(logger, "Build edges (grouped by entity)"):
        alls: List[int] = []
        allt: List[int] = []
        for column in ["Source", "Target", "Location", "Type"]:
            logger.info(f"Building edges for: {column}")
            src: List[int] = []
            tgt: List[int] = []

            # tqdm provides progress; logger provides phase boundaries
            for _, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                idxs = c_df.index.to_list()
                df_len = len(idxs)

                # forward edges: i -> i+j for j in [0..edge_per_trans-1]
                # NOTE: includes self edge when j=0 (matches your original loop behavior)
                for i in range(df_len):
                    end = min(df_len, i + edge_per_trans)
                    si = idxs[i]
                    for j in range(i, end):
                        src.append(si)
                        tgt.append(idxs[j])

            alls.extend(src)
            allt.extend(tgt)
            logger.info(f"{column}: edges added = {len(src):,}")

        alls_np = np.asarray(alls, dtype=np.int64)
        allt_np = np.asarray(allt, dtype=np.int64)
        logger.info(f"Total edges (directed, incl. self edges): {len(alls_np):,}")
        log_memory(logger)

    # Build DGL graph (harden dtype)
    with PhaseTimer(logger, "Create DGL graph"):
        g = dgl.graph((alls_np, allt_np))
        logger.info(f"Graph: nodes={g.num_nodes():,} edges={g.num_edges():,}")
        log_memory(logger)

    # Label encode categorical entity columns (kept from original)
    with PhaseTimer(logger, "Encode categorical columns + attach node features"):
        for c in ["Source", "Target", "Location", "Type"]:
            le = LabelEncoder()
            data[c] = le.fit_transform(data[c].apply(str).values)

        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    # Save graph (preserve name)
    graph_path = os.path.join(output_dir, OUT_GRAPH)
    with PhaseTimer(logger, f"Save graph ({OUT_GRAPH})"):
        dgl.data.utils.save_graphs(graph_path, [g])
        logger.info(f"Saved: {graph_path}")

    # Neighbor features (fast message passing)
    if skip_neigh_feat:
        logger.info("Skipping neighbor feature CSV generation (--skip-neigh-feat set).")
    else:
        with PhaseTimer(logger, "Generate + save neighbor feature CSV"):
            graph_loaded = dgl.load_graphs(graph_path)[0][0]
            neigh_df = generate_neigh_features_fast(graph_loaded, logger=logger)

            # Standardize (kept from your original pipeline)
            scaler = StandardScaler()
            neigh_df = pd.DataFrame(scaler.fit_transform(neigh_df), columns=neigh_df.columns)

            neigh_path = os.path.join(output_dir, OUT_NEIGH_FEAT)
            neigh_df.to_csv(neigh_path, index=False)
            logger.info(f"Saved: {neigh_path}")

    total = time.time() - t0
    logger.info("============================================")
    logger.info(f"DONE: Total elapsed = {total:.2f}s")
    logger.info(f"Outputs in: {output_dir}")
    logger.info(f"  - {OUT_NEOFULL}")
    logger.info(f"  - {OUT_GRAPH}")
    if not skip_neigh_feat:
        logger.info(f"  - {OUT_NEIGH_FEAT}")
    logger.info(f"  - {OUT_LOG}")
    logger.info("============================================")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    run_sffsd_pipeline_v3(
        input_file=args.input_file,
        output_dir=args.output_dir,
        edge_per_trans=args.edge_per_trans,
        skip_temporal_fe=args.skip_temporal_fe,
        unique_counts=args.unique_counts,
        skip_neigh_feat=args.skip_neigh_feat,
    )

