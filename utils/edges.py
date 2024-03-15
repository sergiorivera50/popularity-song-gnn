import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import argparse


def compute_edges(data, ix, t, window=100, threshold=0.7):
    sample = data[ix]
    corr_mx = np.corrcoef(sample[:, t-window:t])
    # Binarize the mean correlation matrix using threshold
    adj_mx = np.where(np.abs(corr_mx) >= threshold, 1, 0)
    np.fill_diagonal(adj_mx, 0)

    # Remove isolated nodes
    """
    temp = nx.from_numpy_array(binary_mx)
    unconnected_nodes = list(nx.isolates(temp))
    temp.remove_nodes_from(unconnected_nodes)
    """

    # Identify and remove unconnected clusters
    """
    clusters = list(nx.connected_components(temp))
    clusters.sort(reverse=True, key=len)
    clusters_list = [list(e) for e in clusters]
    smaller_clusters = clusters_list[1:]
    nodes_to_remove = set([item for sublist in smaller_clusters for item in sublist])
    temp.remove_nodes_from(list(nodes_to_remove))
    adjacency_mx = nx.to_numpy_array(temp)
    """

    return adj_mx, corr_mx


def plot_adjacency_matrix(mx):
    """
    Visualize matrix as adjacency for topology.
    :param mx: adjacency matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(mx, annot=False, fmt=".2f", cmap='Grays',
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of EEG Sensors')
    plt.xlabel('Sensor Index')
    plt.ylabel('Sensor Index')
    plt.show()


def plot_graph(g):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=10)
    nx.draw_networkx_edges(g, pos, width=.5)
    plt.title('Graph Representation of EEG Sensors')
    plt.axis('off')
    plt.show()


def create_temporal_edges_with_ix(data, ix, n_splits, corr_threshold, t_gap=0):
    track_size = data.shape[-1]
    time_sampling = np.linspace(0, track_size, n_splits + 1)
    window = time_sampling[1] - time_sampling[0]
    edges = []
    for t in time_sampling:
        adj_mx, corr_mx = compute_edges(data, ix, t=int(t), window=int(window), threshold=corr_threshold)
        for i in range(len(adj_mx)):
            for j in range(len(adj_mx[0])):
                if adj_mx[i][j]:
                    label = 0
                    features = [corr_mx[i][j]]
                    edges.append([i, j, float(t_gap + t), label, *features])
    return edges


def save_temporal_edges(dataset_name, edges):
    import csv

    header = ["src", "dst", "timestamp", "state_label", "comma_separated_list_of_features"]
    dataset_path = f"data/temporal_edges/{dataset_name}.csv"
    with open(dataset_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([header] + edges)


def create_temporal_edges(parsed_path, dataset_name, corr_threshold=0.7, n_splits=6, sep_time=2000):
    parsed = np.load(parsed_path, allow_pickle=True).item()
    eeg_array = parsed["eeg_array"]  # shape: (samples, channels, eeg values over time)

    edges = []
    track_size = eeg_array.shape[-1]
    time_sampling = np.linspace(0, track_size, n_splits + 1)
    time_sampling = time_sampling[1:]
    window = time_sampling[1] - time_sampling[0]
    print(f"Using window of {window} timesteps.")

    t_total = window
    n_samples = eeg_array.shape[0]
    for track_ix in range(n_samples):
        new_edges = create_temporal_edges_with_ix(eeg_array, track_ix, n_splits=n_splits, corr_threshold=corr_threshold, t_gap=t_total)
        for e in new_edges:
            edges.append(e)
        t_total += track_size + sep_time
    print(f"Created {len(edges) - 1} total edges.")

    save_temporal_edges(dataset_name, edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", type=str, help="Parsed EEG dataset name")
    parser.add_argument("--name", type=str, help="Name for the new temporal edge dataset")
    parser.add_argument("--n_splits", type=int, default=12, help="Number of correlation windows")
    parser.add_argument("--corr", type=float, default=0.9, help="Correlation threshold for adj matrix")
    parser.add_argument("--sep_time", type=int, default=2000, help="Separation time between tracks")
    args = parser.parse_args()
    create_temporal_edges(args.eeg, args.name, corr_threshold=args.corr, n_splits=args.n_splits, sep_time=args.sep_time)
