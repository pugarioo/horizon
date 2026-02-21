import asyncio
import gc
from typing import List, Literal, overload

from llama_cpp import Iterator, Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


class AgentService:
    """
    Handles the lifecycle and operations of the Small Language Models (LLM).

    This service provides methods to load, unload, and swap GGUF models using llama-cpp-python,
    as well as generating chat completions with optional streaming.
    """

    async def load(self, path: str) -> None:
        """
        Loads am SLM model from the given path.

        Args:
            path: The path to the SLM model .gguf file.
        """
        self.model = await asyncio.to_thread(
            Llama, model_path=path, n_gpu_layers=50, n_ctx=4096, verbose=False
        )

        print("Model loaded succesfully")

    async def unload(self) -> None:
        """
        Unloads the currently loaded Llama model to free up memory.
        """

        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None

            # Execute blocking garbage collection on a background thread
            await asyncio.to_thread(gc.collect)

    async def swap(self, path: str) -> None:
        """
        Swaps the currently loaded SLM model with a new one from the given path.

        Args:
            path: The path to the new Llama model.
        """

        await self.unload()

        await self.load(path=path)

    @overload
    async def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        temp: float,
        stream: Literal[False] = False,
    ) -> CreateChatCompletionResponse: ...

    @overload
    async def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        temp: float,
        stream: Literal[True],
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...

    async def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        temp: float,
        stream: bool = False,
    ) -> CreateChatCompletionResponse | Iterator[CreateChatCompletionStreamResponse]:
        """
        Generates a chat completion response based on the given messages and temperature.

        Args:
            messages: A list of chat completion request messages.
            temp: The temperature for the generation.
            stream: Whether to stream the response.

        Returns:
            A chat completion response or an iterator of stream responses.
        """

        if self.model is None:
            raise ValueError("Model not loaded")

        response = await asyncio.to_thread(
            self.model.create_chat_completion,
            messages=messages,
            temperature=temp,
            stream=stream,
        )

        return response
