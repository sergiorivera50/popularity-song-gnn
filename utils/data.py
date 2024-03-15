import numpy as np
import os
from utils.edges import create_temporal_edges
import pandas as pd
import argparse


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def preprocess(dataset_path):
    src_l, dst_l, ts_l, label_l = [], [], [], []
    feat_l, idx_l = [], []

    with open(dataset_path) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            src_l.append(u)
            dst_l.append(i)
            ts_l.append(ts)
            label_l.append(label)
            idx_l.append(idx)
            feat_l.append(feat)
    df = pd.DataFrame({"u": src_l, "i": dst_l, "ts": ts_l, "label": label_l, "idx": idx_l})
    return df, np.array(feat_l)


def reindex(df):
    df_copy = df.copy()
    df_copy.u += 1
    df_copy.i += 1
    df_copy.idx += 1
    return df_copy


def process(dataset_name):
    dataset_path = os.path.join("data", "temporal_edges", f"{dataset_name}.csv")
    out_path = os.path.join("out", f"{dataset_name}.csv")
    out_edge_features_path = os.path.join("out", f"{dataset_name}_edge.npy")
    out_node_features_path = os.path.join("out", f"{dataset_name}_node.npy")

    df, feat = preprocess(dataset_path)
    df = reindex(df)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(df.u.max(), df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    df.to_csv(out_path)
    np.save(out_edge_features_path, feat)
    np.save(out_node_features_path, rand_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Temporal edges dataset name")
    args = parser.parse_args()
    process(args.data)
