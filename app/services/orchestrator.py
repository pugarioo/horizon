import re
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

from app.paths import MODELS_CONFIG_PATH
from app.services.agent_service import AgentService
from app.services.context_manager import ContextManager
from app.services.websocket_manager import WebSocketManager


class Orchestrator:
    """
    Coordinates the interactions between different LLM agents and services.

    The Orchestrator is responsible for managing the flow of conversation,
    invoking specialized agents (Judge, Prover, Critic), and handling
    the synthesis of responses.
    """
    def __init__(self):
        """
        Initializes the Orchestrator with necessary services and configurations.
        """
        self.agent_service: AgentService = AgentService()
        self.context_manager: ContextManager = ContextManager()
        self.websocket_manager: WebSocketManager = WebSocketManager()

        try:
            with open(MODELS_CONFIG_PATH, "r") as file:
                self.model_config: dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading models config: {e}")

    def execute_orchestration(
        self, user_prompt: str, conversation_id: str, websocket: WebSocket
    ) -> None:
        """
        Main entry point for starting the orchestration process for a user prompt.

        Args:
            user_prompt: The prompt provided by the user.
            conversation_id: The unique identifier for the conversation.
            websocket: The active WebSocket connection for real-time communication.
        """

        # self.agent_service.load(
        #     path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['llama']}"
        # )
        pass

    def _generate_critique(
        self, user_prompt: str, answer: str | None = None
    ) -> CreateChatCompletionResponse:
        """
        Generates a critique for a given answer based on the user prompt.

        Args:
            user_prompt: The original user prompt.
            answer: The candidate answer to be critiqued.

        Returns:
            A CreateChatCompletionResponse containing the critique.
        """
        content: str = f"UserPrompt:{user_prompt}\n ToReview:{answer}"

        response: CreateChatCompletionResponse = self.agent_service.generate(
            messages=[
                {
                    "role": "system",
                    "content": self.model_config["agents"]["critic"]["system_prompt"],
                },
                {"role": "user", "content": content},
            ],
            temp=self.model_config["agents"]["critic"]["temperature"],
            stream=False,
        )

        return response

    def _evaluate_and_synthesize(
        self,
        user_prompt: str,
        candidate_solution_a: str,
        candidate_solution_b: str,
    ) -> Iterator[CreateChatCompletionStreamResponse]:
        """
        Evaluates two candidate solutions and synthesizes a final response.

        Args:
            user_prompt: The original user prompt.
            candidate_solution_a: The first candidate solution.
            candidate_solution_b: The second candidate solution.

        Returns:
            An iterator of CreateChatCompletionStreamResponse for the synthesized response.
        """

        content: str = f"UserPrompt:{user_prompt}\n SolutionA:{candidate_solution_a}\n SolutionB:{candidate_solution_b}"

        response: Iterator[CreateChatCompletionStreamResponse] = (
            self.agent_service.generate(
                messages=[
                    {
                        "role": "system",
                        "content": self.model_config["agents"]["judge"][
                            "system_prompt"
                        ],
                    },
                    {"role": "user", "content": content},
                ],
                temp=self.model_config["agents"]["judge"]["temperature"],
                stream=True,
            )
        )

        return response

    def _generate_initial_response(
        self, conversation_id: str, user_prompt: str
    ) -> CreateChatCompletionResponse:
        """
        Generates an initial response by fetching relevant context from memory.

        Args:
            conversation_id: The ID of the conversation.
            user_prompt: The prompt provided by the user.

        Returns:
            A CreateChatCompletionResponse with the initial generated answer.
        """

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
    ) -> CreateChatCompletionResponse:
        """
        Refines a previous response based on a specific refinement request.

        Args:
            conversation_id: The ID of the conversation.
            user_prompt: The original user prompt.
            previouse_response: The response that needs refinement.
            refinement_request: The specific instructions for refinement.

        Returns:
            A CreateChatCompletionResponse with the refined answer.
        """

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
            stream=False,
        )

        return response

    def generate_conversation_id(self):
        """
        Generates a new unique conversation ID.

        Returns:
            A string representation of a UUID.
        """
        return str(uuid.uuid4())

    def contains_refinement_request(self, text: str) -> bool:
        """
        Checks if the given text contains a refinement request marker [RR].

        Args:
            text: The text to search within.

        Returns:
            True if the marker is found, False otherwise.
        """
        pattern: str = r"\[\s*RR\s*\]"

        if re.search(pattern, text, re.IGNORECASE):
            return True
        return False
