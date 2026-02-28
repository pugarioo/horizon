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
from app.services.utils import LogEntry, Path, Roles, State
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
        Iterative orchestration using dynamic hardware constraints from nested YAML.
        """

        await self.context_manager.store_memory(conversation_id, user_prompt)

        await self.websocket_manager.send_current_path(websocket, Path.A)
        current_candidate_a = await self._run_path(
            websocket, conversation_id, user_prompt, Path.A
        )
        await self.websocket_manager.send_current_path(websocket, Path.B)
        current_candidate_b = await self._run_path(
            websocket, conversation_id, user_prompt, Path.B
        )

        # --- FINAL SYNTHESIS: GEMMA 3 JUDGE ---
        await self.websocket_manager.send_status(
            websocket, State.SWAPPING_MODEL, Roles.JUDGE
        )

        judge_model: str = self.model_config["role_map"]["judge"]
        # Pulling judge config (previously 'deepseek' / 'phi', now 'gemma')
        await self.agent_service.swap(
            path=str(
                MODELS_DIR / self.model_config["models"][judge_model]["file_name"]
            ),
            n_gpu_layers=self.model_config["models"][judge_model]["layers"],
            n_ctx=self.model_config["agents"]["judge"]["context_tokens"],
        )

        await self.websocket_manager.send_status(
            websocket, State.GENERATING, Roles.JUDGE
        )

        solution_a_text = (
            self._get_text(current_candidate_a).replace("[RA]", "").strip()
        )
        solution_b_text = (
            self._get_text(current_candidate_b).replace("[RA]", "").strip()
        )

        stream: Iterator[
            CreateChatCompletionStreamResponse
        ] = await self._evaluate_and_synthesize(
            user_prompt=user_prompt,
            candidate_solution_a=solution_a_text,
            candidate_solution_b=solution_b_text,
        )

        full_content = ""
        for chunk in stream:
            delta: (
                ChatCompletionStreamResponseDelta
                | ChatCompletionStreamResponseDeltaEmpty
            ) = chunk["choices"][0]["delta"]

            content: Any | None = delta.get("content")
            if content:
                full_content += content
                await self.websocket_manager.send_message(
                    websocket=websocket, content={"type": "token", "content": content}
                )
                await asyncio.sleep(0)

        print(full_content)
        # Sanitize and store
        clean_final_content = re.sub(
            r"<think>.*?</think>", "", full_content, flags=re.DOTALL
        ).strip()
        await self.context_manager.store_memory(conversation_id, clean_final_content)

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
            max_tokens=self.model_config["agents"]["critic"]["max_tokens"],
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
            max_tokens=self.model_config["agents"]["judge"]["max_tokens"],
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

        relevant_context: str = await self.context_manager.search_context(
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

        print(self.model_config["agents"]["generator"]["max_tokens"])

        response = await self.agent_service.generate(
            messages=query,
            temp=self.model_config["agents"]["generator"]["temperature"],
            max_tokens=self.model_config["agents"]["generator"]["max_tokens"],
            stream=False,
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
                "content": self.model_config["agents"]["refiner"]["system_prompt"],
            },
            {"role": "user", "content": content},
        ]

        response = await self.agent_service.generate(
            messages=query,
            temp=self.model_config["agents"]["refiner"]["temperature"],
            max_tokens=self.model_config["agents"]["refiner"]["max_tokens"],
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

    async def _run_path(
        self, websocket: WebSocket, conversation_id: str, user_prompt: str, path: Path
    ) -> CreateChatCompletionResponse:

        MAX_RETRIES = 2

        generator_model: str = self.model_config["role_map"]["path"][
            "a" if path == Path.A else "b"
        ]["generator"]
        critic_model: str = self.model_config["role_map"]["path"][
            "a" if path == Path.A else "b"
        ]["critic"]

        await self.websocket_manager.send_status(
            websocket, State.LOADING_MODEL, Roles.GENERATOR
        )
        # Pulling nested config: models -> llama -> [file_name, layers]
        await self.agent_service.load(
            path=str(
                MODELS_DIR / self.model_config["models"][generator_model]["file_name"]
            ),
            n_gpu_layers=self.model_config["models"][generator_model]["layers"],
            n_ctx=self.model_config["agents"]["generator"]["context_tokens"],
        )

        if path == Path.A and self._is_new(conversation_id):
            title = await self.generate_title(user_prompt)
            await self.context_manager.store_memory(conversation_id, title)
            await self.websocket_manager.send_title(websocket, title)

        init_completion = await self._run_step(
            websocket,
            Roles.GENERATOR,
            self._generate_initial_response,
            conversation_id,
            user_prompt,
        )
        current_candidate: CreateChatCompletionResponse = init_completion

        attempts_a = 0

        while attempts_a < MAX_RETRIES:
            # Swap to Qwen for Critique
            await self.websocket_manager.send_status(
                websocket, State.SWAPPING_MODEL, Roles.CRITIC
            )
            await self.agent_service.swap(
                path=str(
                    MODELS_DIR / self.model_config["models"][critic_model]["file_name"]
                ),
                n_gpu_layers=self.model_config["models"][critic_model]["layers"],
                n_ctx=self.model_config["agents"]["critic"]["context_tokens"],
            )

            critique: CreateChatCompletionResponse = await self._run_step(
                websocket,
                Roles.CRITIC,
                self._generate_critique,
                user_prompt,
                self._get_text(current_candidate),
            )
            critique_text: str = self._get_text(critique)

            if not self.contains_refinement_request(text=critique_text):
                break

            # Swap back to Llama for Refinement
            await self.websocket_manager.send_status(
                websocket, State.SWAPPING_MODEL, Roles.GENERATOR
            )
            await self.agent_service.swap(
                path=str(
                    MODELS_DIR
                    / self.model_config["models"][generator_model]["file_name"]
                ),
                n_gpu_layers=self.model_config["models"][generator_model]["layers"],
                n_ctx=self.model_config["agents"]["generator"]["context_tokens"],
            )

            refined_response: CreateChatCompletionResponse = await self._run_step(
                websocket,
                Roles.GENERATOR,
                self._generate_refined_response,
                user_prompt,
                current_candidate,
                critique_text,
            )
            current_candidate = refined_response
            attempts_a += 1

        await self.agent_service.unload()

        return current_candidate

    def _is_new(self, conversation_id: str) -> bool:
        messages: List = self.context_manager.get_conversation_messages(
            conversation_id=conversation_id
        )
        return len(messages) == 1

    async def generate_title(self, user_prompt: str) -> str:
        title: CreateChatCompletionResponse = await self.agent_service.generate(
            messages=[
                {
                    "role": "system",
                    "content": self.model_config["agents"]["title-maker"][
                        "system_prompt"
                    ],
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temp=self.model_config["agents"]["title-maker"]["temperature"],
            max_tokens=self.model_config["agents"]["title-maker"]["max_tokens"],
            stream=False,
        )

        return self._get_text(title)
