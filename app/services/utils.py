from enum import Enum


class State(Enum):
    IDLE = "IDLE"
    LOADING_MODEL = "LOADING_MODEL"
    SWAPPING_MODEL = "SWAPPING_MODEL"
    GENERATING = "GENERATING"
    ERROR = "ERROR"


class Roles(Enum):
    GENERATOR = "GENERATOR"
    CRITIC = "CRITIC"
    JUDGE = "JUDGE"


class LogEntry:
    def __init__(self, role: Roles, message: str):
        self.role = role
        self.message = message
