import uuid
from typing import List

import yaml
from fastapi import WebSocket
from llama_cpp import Iterator
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)

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

    def execute_orchestration(
        self, user_prompt: str, conversation_id: str, websocket: WebSocket
    ) -> None:
        pass

    def _generate_critique(
        self, user_prompt: str, refined_answer: str | None = None
    ) -> CreateChatCompletionResponse: ...

    def _evaluate_and_synthesize(
        self, candiate_solution_a: str, candidate_solution_b: str
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...

    def _generate_initial_response(
        self, conversation_id: str, user_prompt: str
    ) -> CreateChatCompletionResponse:

        relevant_context: str = self.context_manager.search_context(
            query=user_prompt, conversation_id=conversation_id
        )

        content: str = f"{relevant_context} {user_prompt}"

        query: List[ChatCompletionRequestMessage] = [
            {
                "role": "system",
                "content": self.model_config["agents"]["generator"]["system_prompt"],
            },
            {"role": "user", "content": content},
        ]

        response = self.agent_service.generate(
            messages=query, temp=self.model_config["agents"]["generator"]["temperature"]
        )

        return response

    def _generate_refined_response(
        self,
        conversation_id: str,
        user_prompt: str | None = None,
        previouse_response: str | None = None,
        refinement_request: str | None = None,
    ) -> Iterator[CreateChatCompletionStreamResponse]:

        content = f"UserPrompt:{user_prompt}\n PreviousResponse:{previouse_response}\n Task:{refinement_request}"

        query: List[ChatCompletionRequestMessage] = [
            {
                "role": "system",
                "content": self.model_config["agents"]["generator"]["system_prompt"],
            },
            {"role": "user", "content": content},
        ]

        response = self.agent_service.generate(
            messages=query,
            temp=self.model_config["agents"]["generator"]["temperature"],
            stream=True,
        )

        return response
