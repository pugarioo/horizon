from enum import Enum


class State(Enum):
    """
    Represents the various states the LLM engine can be in.
    """

    IDLE: str = "IDLE"
    LOADING_MODEL: str = "LOADING_MODEL"
    SWAPPING_MODEL: str = "SWAPPING_MODEL"
    LOADED: str = "LOADED"
    GENERATING: str = "GENERATING"
    ERROR: str = "ERROR"


class Roles(Enum):
    """
    Defines the specialized roles for different agent instances.
    """

    GENERATOR_A: str = "GENERATOR_A"
    GENERATOR_B: str = "GENERATOR_B"
    CRITIC_A: str = "CRITIC_A"
    CRITIC_B: str = "CRITIC_B"
    JUDGE: str = "JUDGE"


class LogEntry:
    """
    Represents a single log entry for agent interactions.
    """

    def __init__(self, role: Roles, message: str):
        """
        Initializes a LogEntry.

        Args:
            role: The role of the agent that generated the log.
            message: The log message content.
        """
        self.role = role
        self.message = message
