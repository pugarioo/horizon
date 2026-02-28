from enum import Enum
from typing import Dict


class State(Enum):
    """
    Represents the various states the LLM engine can be in.
    """

    IDLE = "IDLE"
    LOADING_MODEL = "LOADING_MODEL"
    SWAPPING_MODEL = "SWAPPING_MODEL"
    LOADED = "LOADED"
    GENERATING = "GENERATING"
    ERROR = "ERROR"


class Roles(Enum):
    """
    Defines the specialized roles for different agent instances.
    """

    GENERATOR = "GENERATOR"
    CRITIC = "CRITIC"
    JUDGE = "JUDGE"


class Path(Enum):
    """
    Defines the specialized paths for different agent instances.
    """

    A = "PATH_A"
    B = "PATH_B"


class LogEntry:
    """
    Represents a single log entry for agent interactions.
    """

    def __init__(
        self, role: Roles, message: str, token_usage: Dict[str, int], duration: float
    ):
        """
        Initializes a LogEntry.

        Args:
            role: The role of the agent that generated the log.
            message: The log message content.
        """
        self.role = role
        self.message = message
        self.token_usage = token_usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.duration = duration
