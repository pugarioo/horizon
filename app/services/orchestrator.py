import asyncio
import re
import time
from typing import Any, List

import yaml
from fastapi import WebSocket
from llama_cpp import (
    ChatCompletionStreamResponseDelta,
    ChatCompletionStreamResponseDeltaEmpty,
    Iterator,
)
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)

from app.paths import MODELS_CONFIG_PATH, MODELS_DIR
from app.services.agent_service import AgentService
from app.services.context_manager import ContextManager
from app.services.utils import LogEntry, Roles, State
from app.services.websocket_manager import WebSocketManager


class Orchestrator:
    """
    Coordinates the interactions between different LLM agents and services.

    The Orchestrator is responsible for managing the flow of conversation,
    invoking specialized agents (Judge, Prover, Critic), and handling
    the synthesis of responses.
    """

    def __init__(
        self,
        agent_service: AgentService,
        context_manager: ContextManager,
        websocket_manager: WebSocketManager,
    ):
        """
        Initializes the Orchestrator with necessary services and configurations.
        """
        self.agent_service: AgentService = agent_service
        self.context_manager: ContextManager = context_manager
        self.websocket_manager: WebSocketManager = websocket_manager

        try:
            with open(MODELS_CONFIG_PATH, "r") as file:
                self.model_config: dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error loading models config: {e}")

    async def execute_orchestration(
        self,
        user_prompt: str,
        conversation_id: str,
        websocket: WebSocket,
    ) -> None:
        """
        Main entry point for starting the orchestration process for a user prompt.

        Args:
            user_prompt: The prompt provided by the user.
            conversation_id: The unique identifier for the conversation.
            websocket: The active WebSocket connection for real-time communication.
        """

        # Store user message
        self.context_manager.store_memory(conversation_id, user_prompt)

        # 1. Initial Generation - Model A (Llama)
        await self.websocket_manager.send_status(
            websocket, State.LOADING_MODEL, Roles.GENERATOR_A
        )
        await self.agent_service.load(
            path=str(MODELS_DIR / self.model_config["models"]["file_name"]["llama"])
        )

        init_completion_a = await self._run_step(
            websocket,
            Roles.GENERATOR_A,
            self._generate_initial_response,
            conversation_id,
            user_prompt,
        )
        candidate_sol_a = init_completion_a

        # 2. Initial Generation - Model B (Qwen)
        await self.websocket_manager.send_status(
            websocket, State.SWAPPING_MODEL, Roles.GENERATOR_B
        )
        await self.agent_service.swap(
            path=str(MODELS_DIR / self.model_config["models"]["file_name"]["qwen"])
        )

        init_completion_b = await self._run_step(
            websocket,
            Roles.GENERATOR_B,
            self._generate_initial_response,
            conversation_id,
            user_prompt,
        )
        candidate_sol_b = init_completion_b

        # 3. Critique Solution A (using Qwen)
        critique_a = await self._run_step(
            websocket,
            Roles.CRITIC_A,
            self._generate_critique,
            user_prompt,
            self._get_text(init_completion_a),
        )

        # 4. Refinement Solution A (Swap back to Llama)
        await self.websocket_manager.send_status(
            websocket, State.SWAPPING_MODEL, Roles.GENERATOR_A
        )
        await self.agent_service.swap(
            path=str(MODELS_DIR / self.model_config["models"]["file_name"]["llama"])
        )

        refined_content_a = None
        if self.contains_refinement_request(text=self._get_text(critique_a)):
            refined_completion_a = await self._run_step(
                websocket,
                Roles.GENERATOR_A,
                self._generate_refined_response,
                user_prompt,
                self._get_text(init_completion_a),
                self._get_text(critique_a),
            )
            refined_content_a = self._get_text(refined_completion_a)

        # 5. Critique Solution B (using Llama)
        critique_b = await self._run_step(
            websocket,
            Roles.CRITIC_B,
            self._generate_critique,
            user_prompt,
            self._get_text(init_completion_b),
        )

        # 6. Refinement Solution B (Swap to Qwen)
        await self.websocket_manager.send_status(
            websocket, State.SWAPPING_MODEL, Roles.GENERATOR_B
        )
        await self.agent_service.swap(
            path=str(MODELS_DIR / self.model_config["models"]["file_name"]["qwen"])
        )

        refined_content_b = None
        if self.contains_refinement_request(text=self._get_text(critique_b)):
            refined_completion_b = await self._run_step(
                websocket,
                Roles.GENERATOR_B,
                self._generate_refined_response,
                user_prompt,
                self._get_text(init_completion_b),
                self._get_text(critique_b),
            )
            refined_content_b = self._get_text(refined_completion_b)

        # 7. Finalize Candidate Solution A (using Qwen)
        if refined_content_a:
            final_sol_a_completion = await self._run_step(
                websocket,
                Roles.CRITIC_A,
                self._finalize_candidate_solution,
                user_prompt,
                self._get_text(init_completion_a),
                refined_content_a,
                self._get_text(critique_a),
            )
            candidate_sol_a = final_sol_a_completion

        # 8. Finalize Candidate Solution B (Swap to Llama)
        if refined_content_b:
            await self.websocket_manager.send_status(
                websocket, State.SWAPPING_MODEL, Roles.CRITIC_B
            )
            await self.agent_service.swap(
                path=str(MODELS_DIR / self.model_config["models"]["file_name"]["llama"])
            )
            final_sol_b_completion = await self._run_step(
                websocket,
                Roles.CRITIC_B,
                self._finalize_candidate_solution,
                user_prompt,
                self._get_text(init_completion_b),
                refined_content_b,
                self._get_text(critique_b),
            )
            candidate_sol_b = final_sol_b_completion

        # 9. Final Synthesis - Judge (DeepSeek)
        await self.websocket_manager.send_status(
            websocket, State.SWAPPING_MODEL, Roles.JUDGE
        )
        await self.agent_service.swap(
            path=str(MODELS_DIR / self.model_config["models"]["file_name"]["deepseek"])
        )

        await self.websocket_manager.send_status(
            websocket, State.GENERATING, Roles.JUDGE
        )

        stream: Iterator[
            CreateChatCompletionStreamResponse
        ] = await self._evaluate_and_synthesize(
            user_prompt=user_prompt,
            candidate_solution_a=self._get_text(candidate_sol_a),
            candidate_solution_b=self._get_text(candidate_sol_b),
        )

        full_content: str = ""
        for chunk in stream:
            delta: (
                ChatCompletionStreamResponseDelta
                | ChatCompletionStreamResponseDeltaEmpty
            ) = chunk["choices"][0]["delta"]

            content: Any | None = delta.get("content")

            if content:
                token: dict = {"type": "token", "content": content}
                full_content += content
                await self.websocket_manager.send_message(
                    websocket=websocket, content=token
                )
                await asyncio.sleep(0)

        # Store final response
        self.context_manager.store_memory(conversation_id, full_content)

        print(full_content)

        await self.agent_service.unload()

        await self.websocket_manager.send_status(websocket, State.IDLE, None)

    async def _generate_critique(
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

        response: CreateChatCompletionResponse = await self.agent_service.generate(
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

    async def _finalize_candidate_solution(
        self,
        user_prompt: str,
        initial_answer: str,
        refined_answer: str,
        refinement_request: str,
    ) -> CreateChatCompletionResponse:
        content: str = f"UserPrompt:{user_prompt}\n InitialAnswer:{initial_answer}\n RefinedAnswer:{refined_answer}\n RefinementRequest:{refinement_request}"

        response: CreateChatCompletionResponse = await self.agent_service.generate(
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

    async def _evaluate_and_synthesize(
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

        response: Iterator[
            CreateChatCompletionStreamResponse
        ] = await self.agent_service.generate(
            messages=[
                {
                    "role": "system",
                    "content": self.model_config["agents"]["judge"]["system_prompt"],
                },
                {"role": "user", "content": content},
            ],
            temp=self.model_config["agents"]["judge"]["temperature"],
            stream=True,
        )

        return response

    async def _generate_initial_response(
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

        response = await self.agent_service.generate(
            messages=query, temp=self.model_config["agents"]["generator"]["temperature"]
        )

        return response

    async def _generate_refined_response(
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

        response = await self.agent_service.generate(
            messages=query,
            temp=self.model_config["agents"]["generator"]["temperature"],
            stream=False,
        )

        return response

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
        return response["choices"][0]["message"]["content"]  # type: ignore

    async def _run_step(
        self, websocket: WebSocket, role: Roles, func, *args
    ) -> CreateChatCompletionResponse:
        await self.websocket_manager.send_status(websocket, State.GENERATING, role)

        await asyncio.sleep(0)

        start_time: float = time.time()
        response = await func(*args)
        duration: float = time.time() - start_time

        self.context_manager.add_agent_log(
            LogEntry(
                role=role,
                message=self._get_text(response),
                token_usage=response["usage"],
                duration=duration,
            )
        )

        print(role)
        print(self._get_text(response=response))

        return response
