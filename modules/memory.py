import torch
from torch import nn
from collections import defaultdict
from abc import ABC, abstractmethod


class Memory(nn.Module, ABC):
    """
    Abstract base class representing a memory module for a neural network.
    This class defines a structure for storing and updating memory nodes with messages.
    """

    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
                 device="cpu", combination_method='sum'):
        """
        :param n_nodes: Number of nodes in the memory.
        :param memory_dimension: Dimensionality of each memory node.
        :param input_dimension: Dimensionality of input messages.
        :param message_dimension (optional): Dimensionality of messages; defaults to input_dimension if not provided.
        :param device: The device (cpu or cuda) on which the tensors are stored.
        :param combination_method: Method used to combine messages (e.g., 'sum', 'mean', etc.).
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension if message_dimension is not None else input_dimension
        self.device = device
        self.combination_method = combination_method

        # Initialize memory parameters
        self.__init_memory__()

    def __init_memory__(self):
        """
        :param Initializes the memory and last update time to all zeroes.
        :param This method is intended to be called at the start of each epoch to reset the memory.
        """
        self.memory = nn.Parameter(torch.zeros(size=(self.n_nodes, self.memory_dimension), device=self.device),
                                   requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(size=(self.n_nodes,), device=self.device),
                                        requires_grad=False)
        self.messages = defaultdict(list)  # Initialize a default dictionary for storing messages

    def store_raw_messages(self, nodes, node_id_to_messages):
        """
        :param Stores raw messages for specified nodes.

        :param Args:
        :param    nodes: List of node indices for which messages are to be stored.
        :param    node_id_to_messages: Dictionary mapping node indices to their messages.
        """
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        """
        :param Retrieves the memory values for specified node indices.

        :param Args:
        :param    node_idxs: Indices of nodes whose memory values are to be retrieved.

        :param Returns:
        :param    A tensor containing the memory values of the specified nodes.
        """
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        """
        :param Sets the memory values for specified node indices.

        :param Args:
        :param    node_idxs: Indices of nodes whose memory values are to be set.
        :param   values: New memory values for the nodes.
        """
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        """
        :param Retrieves the last update time for specified node indices.

        :param Args:
        :param    node_idxs: Indices of nodes whose last update times are to be retrieved.

        :param Returns:
        :param   A tensor containing the last update times of the specified nodes.
        """
        return self.last_update[node_idxs]

    def backup_memory(self):
        """
        :param Creates a backup of the current memory and messages.

        :param Returns:
        :param     A tuple containing clones of the memory tensor, last update times, and messages.
        """
        messages_clone = {k: [(msg[0].clone(), msg[1].clone()) for msg in v] for k, v in self.messages.items()}
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        """
        :param Restores memory and messages from a backup.

        :param Args:
        :param     memory_backup: A tuple containing memory tensor, last update times, and messages to be restored.
        """
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(msg[0].clone(), msg[1].clone()) for msg in v]

    def detach_memory(self):
        """
        :param Detaches memory from the current computation graph. Also detaches stored messages.
        :param This is useful to prevent gradients from flowing back through the memory updates.
        """
        self.memory.detach_()
        for k, v in self.messages.items():
            self.messages[k] = [(msg[0].detach(), msg[1]) for msg in v]

    def clear_messages(self, nodes):
        """
        :param Clears messages for specified nodes.

        :param Args:
        :param nodes: List of node indices for which messages are to be cleared.
        """
        for node in nodes:
            self.messages[node] = []

    @abstractmethod
    def update_memory(self, node_idxs, new_values):
        """
        :param Abstract method to update memory for given nodes with new values.
        :param Subclasses must implement this method to define how memory updates are handled.

        :param Args:
        :param     node_idxs (Tensor): Indices of nodes to update.
        :param     new_values (Tensor): New values to update the memory with.
        """
        pass
