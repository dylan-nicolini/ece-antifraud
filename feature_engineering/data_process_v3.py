#!/usr/bin/env python3
"""
data_process.py (V3) — Canonical S-FFSD pipeline, optimized for 500K+ rows

This version keeps S-FFSD naming conventions for GTAN/RGTAN, but adds:
  --dataset S-FFSD   (explicit; only allowed value)
  --input-file       (any CSV already in S-FFSD schema)
  --output-dir       (so experiments don't overwrite)

Core improvements for scale:
1) Temporal feature engineering (featmap_gen) rewritten:
   - Old: O(N^2) scans per row per window
   - New: O(N * W) using:
       - sort by Time
       - prefix sums for Amount and Amount^2
       - sliding window left pointers per window

   Included in V3 (fast):
     trans_at_avg_*
     trans_at_totl_*
     trans_at_std_*
     trans_at_bias_*
     trans_at_num_*

   Optional (slow, default OFF):
     trans_target_num_*
     trans_location_num_*
     trans_type_num_*

2) Neighbor "risk-aware" features rewritten:
   - Old: per-node khop subgraphs (will not finish at 500K)
   - New: DGL message passing aggregations (scales)

Outputs (unchanged names):
  - S-FFSDneofull.csv
  - graph-S-FFSD.bin
  - S-FFSD_neigh_feat.csv

Schema required (case-insensitive):
  Time, Source, Target, Amount, Location, Type, Labels
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import os
import random
import pickle

import numpy as np
import pandas as pd
import torch
import dgl
import dgl.function as fn
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Default data directory (same spirit as original)
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")
SFFSD_REQUIRED_COLS = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _assert_sffsd_schema(df: pd.DataFrame) -> dict:
    """Case-insensitive schema validation + returns canonical->actual col mapping."""
    lookup = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in SFFSD_REQUIRED_COLS if c.lower() not in lookup]
    if missing:
        raise ValueError(
            f"Input is not S-FFSD-shaped. Missing columns: {missing}. Found: {list(df.columns)}"
        )
    return {c: lookup[c.lower()] for c in SFFSD_REQUIRED_COLS}


def sparse_to_adjlist(sp_matrix, filename):
    """Transfer sparse matrix to adjacency list (kept for compatibility; not used in V3 core)."""
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, "wb") as file:
        pickle.dump(adj_lists, file)


# -----------------------------
# V3 Temporal Feature Engineering (FAST)
# -----------------------------
def featmap_gen_v3(
    df: pd.DataFrame,
    time_windows: list[int] | None = None,
    compute_unique_counts: bool = False,
) -> pd.DataFrame:
    """
    Fast temporal feature engineering.

    Requirements:
      - df has Time, Amount, Target, Location, Type (S-FFSD schema)
      - Time is numeric (seconds), Amount numeric

    Behavior:
      - Sort by Time ascending
      - For each window length L, compute window stats ending at each row i:
          mean, sum, std, bias, count
      - Optional unique counts per window (very expensive at 500K); default OFF.

    Returns:
      DataFrame with original columns plus engineered columns.
    """
    if time_windows is None:
        # same idea as original, but this list can be tuned
        time_windows = [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]

    # Work on a copy
    df = df.copy()

    # Ensure numeric
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0).astype(np.int64)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).astype(np.float64)

    # Sort by Time (critical for sliding windows)
    df = df.sort_values("Time").reset_index(drop=True)

    t = df["Time"].to_numpy(dtype=np.int64)
    amt = df["Amount"].to_numpy(dtype=np.float64)

    n = len(df)
    # Prefix sums for sum and sumsq
    pref = np.zeros(n + 1, dtype=np.float64)
    pref2 = np.zeros(n + 1, dtype=np.float64)
    pref[1:] = np.cumsum(amt)
    pref2[1:] = np.cumsum(amt * amt)

    # Allocate outputs
    for L in time_windows:
        name = str(L)
        df[f"trans_at_totl_{name}"] = 0.0
        df[f"trans_at_avg_{name}"] = 0.0
        df[f"trans_at_std_{name}"] = 0.0
        df[f"trans_at_bias_{name}"] = 0.0
        df[f"trans_at_num_{name}"] = 0

        # optional heavy features
        df[f"trans_target_num_{name}"] = 0
        df[f"trans_location_num_{name}"] = 0
        df[f"trans_type_num_{name}"] = 0

    # Two-pointer per window (O(N * W))
    left_ptrs = {L: 0 for L in time_windows}

    for i in range(n):
        ti = t[i]
        ai = amt[i]

        for L in time_windows:
            l = left_ptrs[L]
            # advance left pointer until time >= ti - L
            boundary = ti - L
            while l < i and t[l] < boundary:
                l += 1
            left_ptrs[L] = l

            # window is [l..i], inclusive
            count = i - l + 1
            s = pref[i + 1] - pref[l]
            s2 = pref2[i + 1] - pref2[l]
            mean = s / count
            var = (s2 / count) - (mean * mean)
            std = np.sqrt(var) if var > 0 else 0.0
            bias = ai - mean

            name = str(L)
            df.at[i, f"trans_at_totl_{name}"] = float(s)
            df.at[i, f"trans_at_avg_{name}"] = float(mean)
            df.at[i, f"trans_at_std_{name}"] = float(std)
            df.at[i, f"trans_at_bias_{name}"] = float(bias)
            df.at[i, f"trans_at_num_{name}"] = int(count)

    if compute_unique_counts:
        # WARNING: expensive at scale. This is exact and can be slow.
        # We implement per-window with a rolling set on the slice [l..i].
        # For 500K rows and 15 windows, this can be very heavy.
        tgt = df["Target"].astype("string").fillna("").to_numpy()
        loc = df["Location"].astype("string").fillna("").to_numpy()
        typ = df["Type"].astype("string").fillna("").to_numpy()

        # Reuse left_ptrs but reset
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
                # exact uniques in window slice (slow)
                df.at[i, f"trans_target_num_{name}"] = int(len(set(tgt[l:i+1])))
                df.at[i, f"trans_location_num_{name}"] = int(len(set(loc[l:i+1])))
                df.at[i, f"trans_type_num_{name}"] = int(len(set(typ[l:i+1])))

    return df


# -----------------------------
# V3 Neighbor Features (FAST message passing)
# -----------------------------
def generate_neigh_features_fast(graph: dgl.DGLGraph) -> pd.DataFrame:
    """
    Scalable neighbor features using DGL message passing.

    Produces same semantic columns as original output:
      degree, riskstat, 1hop_degree, 2hop_degree, 1hop_riskstat, 2hop_riskstat

    Notes:
      - degree: in-degree (matches your original code's degree_feat)
      - riskstat: number of risk-labeled OUT-neighbors (matches your original count_risk_neighs)
      - 1hop_degree: sum of neighbor degrees over OUT-neighbors
      - 2hop_degree: sum of neighbor 1hop_degree over OUT-neighbors
      - 1hop_riskstat: sum of neighbor riskstat over OUT-neighbors
      - 2hop_riskstat: sum of neighbor 1hop_riskstat over OUT-neighbors
    """
    # degree (in-degree like original)
    deg = graph.in_degrees().to(torch.float32)

    # riskstat: sum labels over OUT neighbors
    graph.ndata["label_f"] = graph.ndata["label"].to(torch.float32)
    graph.update_all(fn.copy_u("label_f", "m"), fn.sum("m", "riskstat_out"))
    riskstat = graph.ndata["riskstat_out"].to(torch.float32)

    # 1-hop degree aggregate over OUT neighbors
    graph.ndata["deg"] = graph.out_degrees().to(torch.float32)
    graph.update_all(fn.copy_u("deg", "m"), fn.sum("m", "deg_1hop"))
    deg_1hop = graph.ndata["deg_1hop"].to(torch.float32)

    # 2-hop degree aggregate over OUT neighbors of deg_1hop
    graph.ndata["deg_1hop"] = deg_1hop
    graph.update_all(fn.copy_u("deg_1hop", "m"), fn.sum("m", "deg_2hop"))
    deg_2hop = graph.ndata["deg_2hop"].to(torch.float32)

    # 1-hop riskstat aggregate
    graph.ndata["riskstat"] = riskstat
    graph.update_all(fn.copy_u("riskstat", "m"), fn.sum("m", "risk_1hop"))
    risk_1hop = graph.ndata["risk_1hop"].to(torch.float32)

    # 2-hop riskstat aggregate
    graph.ndata["risk_1hop"] = risk_1hop
    graph.update_all(fn.copy_u("risk_1hop", "m"), fn.sum("m", "risk_2hop"))
    risk_2hop = graph.ndata["risk_2hop"].to(torch.float32)

    feats = torch.stack([deg, riskstat, deg_1hop, deg_2hop, risk_1hop, risk_2hop], dim=1).cpu().numpy()
    feats[np.isnan(feats)] = 0.0

    columns = ["degree", "riskstat", "1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
    return pd.DataFrame(feats, columns=columns)


# -----------------------------
# CLI / Main pipeline (canonical S-FFSD)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Canonical S-FFSD processor (V3) optimized for 500K+ rows; preserves GTAN/RGTAN naming."
    )
    p.add_argument("--dataset", default="S-FFSD", choices=["S-FFSD"],
                   help="Must be S-FFSD (your IEEE experiments must already be S-FFSD-shaped CSVs).")
    p.add_argument("--input-file", default=os.path.join(DATADIR, "S-FFSD.csv"),
                   help="Path to S-FFSD-shaped CSV.")
    p.add_argument("--output-dir", default=DATADIR,
                   help="Directory to write outputs (filenames preserved).")
    p.add_argument("--edge-per-trans", type=int, default=3,
                   help="Forward edges per transaction within each entity group (default=3).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--skip-temporal-fe", action="store_true",
                   help="Skip temporal feature engineering entirely (fastest).")
    p.add_argument("--unique-counts", action="store_true",
                   help="Compute exact unique counts in temporal windows (VERY slow at 500K+).")
    p.add_argument("--skip-neigh-feat", action="store_true",
                   help="Skip writing S-FFSD_neigh_feat.csv.")
    return p.parse_args()


def run_sffsd_pipeline_v3(
    input_file: str,
    output_dir: str,
    edge_per_trans: int,
    skip_temporal_fe: bool,
    compute_unique_counts: bool,
    skip_neigh_feat: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("processing S-FFSD data (V3)...")
    print(f"  input_file       : {input_file}")
    print(f"  output_dir       : {output_dir}")
    print(f"  edge_per_trans   : {edge_per_trans}")
    print(f"  skip_temporal_fe : {skip_temporal_fe}")
    print(f"  unique_counts    : {compute_unique_counts}")
    print(f"  skip_neigh_feat  : {skip_neigh_feat}")

    data = pd.read_csv(input_file)
    col = _assert_sffsd_schema(data)
    # Normalize canonical column names (keep original names too, but ensure expected labels)
    data = data.rename(columns={v: k for k, v in col.items()})

    # Temporal feature engineering (optimized)
    if not skip_temporal_fe:
        data = featmap_gen_v3(data.reset_index(drop=True), compute_unique_counts=compute_unique_counts)

    data.replace(np.nan, 0, inplace=True)

    # Preserve original intermediate output name
    neo_full_path = os.path.join(output_dir, "S-FFSDneofull.csv")
    data.to_csv(neo_full_path, index=None)

    # Reload to preserve original behavior pattern
    data = pd.read_csv(neo_full_path).reset_index(drop=True)

    # Build edges (same logic, but hardened int64)
    alls: List[int] = []
    allt: List[int] = []
    pair = ["Source", "Target", "Location", "Type"]

    for column in pair:
        src, tgt = [], []
        for _, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index.to_list()
            # forward edges up to edge_per_trans (including i->i)
            for i in range(df_len):
                for j in range(edge_per_trans):
                    if i + j < df_len:
                        src.append(sorted_idxs[i])
                        tgt.append(sorted_idxs[i + j])
        alls.extend(src)
        allt.extend(tgt)

    alls = np.asarray(alls, dtype=np.int64)
    allt = np.asarray(allt, dtype=np.int64)

    g = dgl.graph((alls, allt))

    # Label encode categorical entity columns (same as original)
    for c in ["Source", "Target", "Location", "Type"]:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c].apply(str).values)

    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]

    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    # Preserve original graph filename
    graph_path = os.path.join(output_dir, "graph-S-FFSD.bin")
    dgl.data.utils.save_graphs(graph_path, [g])

    print(f"Saved graph: {graph_path}")
    print(f"Graph info: {g}")

    if skip_neigh_feat:
        print("Skipping neighbor feature CSV generation.")
        return

    # Generate scalable neighbor features (fast message passing)
    print("Generating neighbor features (FAST message passing)...")
    graph = dgl.load_graphs(graph_path)[0][0]

    neigh_df = generate_neigh_features_fast(graph)

    # Standardize (kept from original)
    scaler = StandardScaler()
    neigh_df = pd.DataFrame(scaler.fit_transform(neigh_df), columns=neigh_df.columns)

    output_path = os.path.join(output_dir, "S-FFSD_neigh_feat.csv")
    neigh_df.to_csv(output_path, index=False)
    print(f"Saved neighbor features: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    run_sffsd_pipeline_v3(
        input_file=args.input_file,
        output_dir=args.output_dir,
        edge_per_trans=args.edge_per_trans,
        skip_temporal_fe=args.skip_temporal_fe,
        compute_unique_counts=args.unique_counts,
        skip_neigh_feat=args.skip_neigh_feat,
    )
