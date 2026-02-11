import uuid

import yaml
from fastapi import WebSocket

import src.paths
from src.services.agent_service import AgentService
from src.services.context_manager import ContextManager
from src.services.websocket_manager import WebSocketManager


class Orchestrator:
    def __init__(self):
        self.agent_service: AgentService = AgentService()
        self.context_manager: ContextManager = ContextManager()
        self.websocket_manager: WebSocketManager = WebSocketManager()

        try:
            with open(src.paths.MODELS_CONFIG_PATH, "r") as file:
                self.model_config: dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading models config: {e}")

    def generate_conversation_id(self):
        return str(uuid.uuid4())

    def generate_response(
        self, user_prompt: str, conversation_id: str, websocket: WebSocket
    ):
        pass
