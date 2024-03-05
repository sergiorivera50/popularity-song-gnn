import numpy as np


class TemporalNeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighbors]))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph.
        The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps
        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], \
            self.node_to_edge_timestamps[src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood
        of each user in the list.

        :param source_nodes: List[int], source nodes for which to find neighbors.
        :param timestamps: List[float], timestamps to cut off the neighbor search.
        :param n_neighbors: int, number of neighbors to find.
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sampled_idx = self.random_state.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # Re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take the most recent interactions
                    actual_n_neighbors = min(len(source_neighbors), n_neighbors)
                    neighbors[i, -actual_n_neighbors:] = source_neighbors[-actual_n_neighbors:]
                    edge_times[i, -actual_n_neighbors:] = source_edge_times[-actual_n_neighbors:]
                    edge_idxs[i, -actual_n_neighbors:] = source_edge_idxs[-actual_n_neighbors:]

        return neighbors, edge_idxs, edge_times
