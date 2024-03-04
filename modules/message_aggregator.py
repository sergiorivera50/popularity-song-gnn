import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Tuple


def get_last_message(messages: Tuple[any, any]):
    return messages[-1]  # (msg, timestamp)


class MessageAggregator(nn.Module, ABC):
    """
    Abstract module employed to aggregate messages, given a batch of nodes (ids) and corresponding messages.
    """

    def __init__(self, device: torch.cuda.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def aggregate(self, nodes, messages):
        """
        Abstract aggregation method for messages given a list of equal length containing node ids.
        :param nodes: List of node ids with length batch_size.
        :param messages: Tensor of shape (batch_size, message_length).
        :return: Tensor of shape (n_unique_ids, message_length) containing the aggregated messages.
        """
        pass

    @staticmethod
    def group_by_id(nodes, messages, timestamps):
        nodes_to_messages = defaultdict(list)  # create default items with list()
        for i, node in enumerate(nodes):
            paired_message = (messages[i], timestamps[i])
            nodes_to_messages[node].append(paired_message)
        return nodes_to_messages


class LastMessageAggregator(MessageAggregator):
    """
    Non-learnable implementation of keeping only the most recent message for any given node.
    """

    def __init__(self, device: torch.cuda.device) -> None:
        super().__init__(device)

    def aggregate(self, nodes, messages):
        """
        Aggregation method of keeping the last message for each node.
        :param nodes: List of node ids with length batch_size.
        :param messages: Tensor of shape (batch_size, message_length).
        :return: Tensor of shape (n_unique_ids, message_length) containing the aggregated messages.
        """
        unique_messages = []
        unique_timestamps = []
        nodes_to_update = []

        for node in np.unique(nodes):
            if len(messages[node]) == 0:
                continue
            nodes_to_update.append(node)
            msg, timestamp = get_last_message(messages[node])
            unique_messages.append(msg)
            unique_timestamps.append(timestamp)

        if len(nodes_to_update) == 0:
            return [], [], []

        return nodes_to_update, torch.stack(unique_messages), torch.stack(unique_timestamps)


def get_message_aggregator(module_type: str, device: torch.cuda.device) -> MessageAggregator:
    match module_type:
        case "last":
            return LastMessageAggregator(device=device)
        case _:
            raise NotImplemented(f"Message aggregator '{module_type}' not implemented.")
