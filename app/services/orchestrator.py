import re
import time
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
from app.services.utils import LogEntry, Roles
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

        self.agent_service.load(
            path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['llama']}"
        )

        start_time: float = time.time()

        init_completion_a: CreateChatCompletionResponse = (
            self._generate_initial_response(
                conversation_id=conversation_id, user_prompt=user_prompt
            )
        )
        candidate_sol_a: CreateChatCompletionResponse = init_completion_a

        duration: float = time.time() - start_time

        self.context_manager.add_agent_log(
            LogEntry(
                role=Roles.GENERATOR_A,
                content=init_completion_a["choices"][0]["message"]["content"],
                token_usage=init_completion_a["usage"],
                duration=duration,
            )
        )

        self.agent_service.swap(
            path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['qwen']}"
        )

        start_time = time.time()

        init_completion_b: CreateChatCompletionResponse = (
            self._generate_initial_response(conversation_id, user_prompt)
        )
        candidate_sol_b: CreateChatCompletionResponse = init_completion_b

        duration = time.time() - start_time

        self.context_manager.add_agent_log(
            LogEntry(
                role=Roles.GENERATOR_B,
                content=init_completion_b["choices"][0]["message"]["content"],
                token_usage=init_completion_b["usage"],
                duration=duration,
            )
        )

        start_time = time.time()

        critique_a: CreateChatCompletionResponse = self._generate_critique(
            user_prompt=user_prompt,
            answer=init_completion_a["choices"][0]["message"]["content"],
        )

        duration = time.time() - start_time

        self.context_manager.add_agent_log(
            LogEntry(
                role=Roles.CRITIC,
                content=critique_a["choices"][0]["message"]["content"],
                token_usage=critique_a["usage"],
                duration=duration,
            )
        )

        self.agent_service.swap(
            path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['llama']}"
        )

        refined_content_a: str | None = None
        if self.contains_refinement_request(
            text=critique_a["choices"][0]["message"]["content"]
        ):
            start_time = time.time()

            refined_completion_a: CreateChatCompletionResponse = (
                self._generate_refined_response(
                    user_prompt=user_prompt,
                    previous_response=init_completion_a["choices"][0]["message"][
                        "content"
                    ],
                    refinement_request=critique_a["choices"][0]["message"]["content"],
                )
            )

            refined_content_a = refined_completion_a["choices"][0]["message"]["content"]

            duration = time.time() - start_time

            self.context_manager.add_agent_log(
                LogEntry(
                    role=Roles.GENERATOR_A,
                    content=refined_content_a,
                    token_usage=refined_completion_a["usage"],
                    duration=duration,
                )
            )

        start_time = time.time()

        critique_b: CreateChatCompletionResponse = self._generate_critique(
            user_prompt=user_prompt,
            answer=init_completion_b["choices"][0]["message"]["content"],
        )

        duration = time.time() - start_time

        self.context_manager.add_agent_log(
            LogEntry(
                role=Roles.CRITIC_B,
                content=critique_b["choices"][0]["message"]["content"],
                token_usage=critique_b["usage"],
                duration=duration,
            )
        )

        self.agent_service.swap(
            path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['qwen']}"
        )

        refined_content_b: str | None = None

        if self.contains_refinement_request(
            text=critique_b["choices"][0]["message"]["content"]
        ):
            start_time = time.time()

            refined_completion_b: CreateChatCompletionResponse = (
                self._generate_refined_response(
                    user_prompt=user_prompt,
                    previous_response=init_completion_b["choices"][0]["message"][
                        "content"
                    ],
                    refinement_request=critique_b["choices"][0]["message"]["content"],
                )
            )

            refined_content_b = refined_completion_b["choices"][0]["message"]["content"]

            duration = time.time() - start_time

            self.context_manager.add_agent_log(
                LogEntry(
                    role=Roles.GENERATOR_B,
                    content=refined_content_b,
                    token_usage=refined_completion_b["usage"],
                    duration=duration,
                )
            )

        if refined_content_a:
            start_time = time.time()

            candidate_sol_a = self._finalize_candidate_solution(
                user_prompt=user_prompt,
                initial_answer=init_completion_a["choices"][0]["message"]["content"],
                refined_answer=refined_content_a,
                refinement_request=critique_a["choices"][0]["message"]["content"],
            )

            duration = time.time() - start_time

            self.context_manager.add_agent_log(
                LogEntry(
                    role=Roles.CRITIC_A,
                    content=candidate_sol_a["choices"][0]["message"]["content"],
                    token_usage=candidate_sol_a["usage"],
                    duration=duration,
                )
            )
        if refined_content_b:
            self.agent_service.swap(
                f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['llama']}"
            )

            start_time = time.time()

            candidate_sol_b = self._finalize_candidate_solution(
                user_prompt=user_prompt,
                initial_answer=init_completion_b["choices"][0]["message"]["content"],
                refined_answer=refined_content_b,
                refinement_request=critique_b["choices"][0]["message"]["content"],
            )

            duration = time.time() - start_time

            self.context_manager.add_agent_log(
                LogEntry(
                    role=Roles.CRITIC_B,
                    content=candidate_sol_b["choices"][0]["message"]["content"],
                    token_usage=candidate_sol_b["usage"],
                    duration=duration,
                )
            )
        self.agent_service.swap(
            path=f"{MODELS_CONFIG_PATH}/{self.model_config['models']['file_name']['deepseek']}"
        )

        final_a: str = candidate_sol_a["choices"][0]["message"]["content"]
        final_b: str = candidate_sol_b["choices"][0]["message"]["content"]

        final_response: Iterator[CreateChatCompletionStreamResponse] = (
            self._evaluate_and_synthesize(
                user_prompt=user_prompt,
                candidate_solution_a=final_a,
                candidate_solution_b=final_b,
            )
        )

    def _generate_critique(
        self, user_prompt: str, answer: str
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

    def _finalize_candidate_solution(
        self,
        user_prompt: str,
        initial_answer: str,
        refined_answer: str,
        refinement_request: str,
    ) -> CreateChatCompletionResponse:
        content: str = f"UserPrompt:{user_prompt}\n InitialAnswer:{initial_answer}\n RefinedAnswer:{refined_answer}\n RefinementRequest:{refinement_request}"

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
        user_prompt: str | None = None,
        previous_response: str | None = None,
        refinement_request: str | None = None,
    ) -> CreateChatCompletionResponse:
        """
        Refines a previous response based on a specific refinement request.

        Args:
            conversation_id: The ID of the conversation.
            user_prompt: The original user prompt.
            previous_response: The response that needs refinement.
            refinement_request: The specific instructions for refinement.

        Returns:
            A CreateChatCompletionResponse with the refined answer.
        """

        content = f"UserPrompt:{user_prompt}\n PreviousResponse:{previous_response}\n Task:{refinement_request}"

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


def _get_text(self, response: CreateChatCompletionResponse) -> str:
    return response["choices"][0]["message"]["content"]

# def _run_step(self, role: Roles, func, *args) -> None:
#     start_time