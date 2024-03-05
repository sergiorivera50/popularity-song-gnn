from torch import nn
import torch
from abc import ABC, abstractmethod


class MemoryUpdater(nn.Module, ABC):
    """
    Abstract base class for memory updaters. Defines the interface for updating memory.
    This class cannot be instantiated on its own and requires subclassing.
    """

    @abstractmethod
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Abstract method to update the memory based on unique node IDs, messages, and timestamps.

        :param unique_node_ids: Node IDs for which to update memory.
        :param unique_messages: Messages corresponding to each node ID.
        :param timestamps: Timestamps at which each message is received.
        """
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    """
    A memory updater that processes sequences of messages using a neural network layer.
    Implements the MemoryUpdater interface.
    """

    def __init__(self, memory, message_dim, memory_dim, device):
        """
        Initializes the SequenceMemoryUpdater instance.

        :param memory: The memory component to be updated.
        :param message_dim: The dimensionality of incoming messages.
        :param memory_dim: The dimensionality of the memory vector.
        :param device: The computation device ('cpu' or 'cuda').
        """
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=memory_dim)
        self.message_dim = message_dim
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Updates memory for given node IDs with the new messages at the specified timestamps.

        :param unique_node_ids: Node IDs for which to update memory.
        :param unique_messages: Messages corresponding to each node ID.
        :param timestamps: Timestamps at which each message is received.
        """
        raise NotImplementedError("Subclasses should implement this portion")

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Returns the updated memory and last update timestamps, handling edge cases.

        :param unique_node_ids: Node IDs for which to retrieve updated memory.
        :param unique_messages: Messages corresponding to each node ID for updating memory.
        :param timestamps: Timestamps at which each message is received.

        Returns:
            Tuple of updated memory and last update timestamps.
        """
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(node_idxs=unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to a time in the past."

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(input=unique_messages, hx=updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    """
    Implements memory updating using a GRU cell.
    """

    def __init__(self, memory, message_dim, memory_dim, device):
        """
        Initializes the GRUMemoryUpdater instance.

        :param memory: The memory component to be updated.
        :param message_dim: The dimensionality of incoming messages.
        :param memory_dim: The dimensionality of the memory vector.
        :param device: The computation device ('cpu' or 'cuda').
        """
        super().__init__(memory=memory, message_dim=message_dim, memory_dim=memory_dim, device=device)
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Specific implementation of update_memory for the GRU memory updater.
        """
        super().update_memory(unique_node_ids, unique_messages, timestamps)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    """
    Implements memory updating using an RNN cell.
    """

    def __init__(self, memory, message_dim, memory_dim, device):
        """
        Initializes the RNNMemoryUpdater instance.

        :param memory: The memory component to be updated.
        :param message_dim: The dimensionality of incoming messages.
        :param memory_dim: The dimensionality of the memory vector.
        :param device: The computation device ('cpu' or 'cuda').
        """
        super().__init__(memory=memory, message_dim=message_dim, memory_dim=memory_dim, device=device)
        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        """
        Specific implementation of update_memory for the RNN memory updater.
        """
        super().update_memory(unique_node_ids, unique_messages, timestamps)


def get_memory_updater(module_type, memory, message_dim, memory_dim, device):
    """
    Factory function to create a memory updater based on the module type.

    :param module_type: Type of the memory updater ('gru' or 'rnn').
    :param memory: The memory component to be updated.
    :param message_dim: The dimensionality of incoming messages.
    :param memory_dim: The dimensionality of the memory vector.
    :param device: The computation device ('cpu' or 'cuda').

    Returns:
        An instance of a MemoryUpdater.
    """
    match module_type:
        case "gru":
            return GRUMemoryUpdater(memory=memory, message_dim=message_dim, memory_dim=memory_dim, device=device)
        case "rnn":
            return RNNMemoryUpdater(memory=memory, message_dim=message_dim, memory_dim=memory_dim, device=device)
        case _:
            raise NotImplementedError(f"Memory updater '{module_type}' not implemented.")
