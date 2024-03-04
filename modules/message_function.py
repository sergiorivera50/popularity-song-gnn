import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class MessageFunction(nn.Module, ABC):
    """
    Abstract module employed to compute a message for any given interaction.
    """

    @abstractmethod
    def compute_message(self, raw_messages):
        pass


class MLPMessageFunction(MessageFunction):
    """
    Learnable message function implementation.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        :param input_dim: Raw message dimensions.
        :param output_dim: Output message dimensions.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )

    def compute_message(self, raw_messages):
        return self.linear(raw_messages)


class IdentityMessageFunction(MessageFunction):
    """
    Simpler non-learnable identity implementation.
    """

    def compute_message(self, raw_messages):
        return raw_messages


def get_message_function(
        module_type: str,
        raw_message_dim: Optional[int],
        message_dim: Optional[int]
) -> MessageFunction:
    match module_type:
        case "mlp":
            return MLPMessageFunction(
                input_dim=raw_message_dim,
                output_dim=message_dim
            )
        case "identity":
            return IdentityMessageFunction()
        case _:
            raise NotImplemented(f"Message function '{module_type}' not implemented.")
