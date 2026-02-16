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

    def load(self, path: str) -> None:
        """
        Loads am SLM model from the given path.

        Args:
            path: The path to the SLM model .gguf file.
        """
        self.model: Llama = Llama(
            model_path=path, n_gpu_layers=50, n_ctx=4096, verbose=False
        )

        print("Model loaded succesfully")

    def unload(self) -> None:
        """
        Unloads the currently loaded Llama model to free up memory.
        """

        if self.model:
            del self.model

            gc.collect()

    def swap(self, path: str) -> None:
        """
        Swaps the currently loaded SLM model with a new one from the given path.

        Args:
            path: The path to the new Llama model.
        """

        self.unload()

        self.load(path=path)

    @overload
    def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        temp: float,
        stream: Literal[False] = False,
    ) -> CreateChatCompletionResponse: ...

    @overload
    def generate(
        self,
        messages: List[ChatCompletionRequestMessage],
        temp: float,
        stream: Literal[True],
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...

    def generate(
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

        response = self.model.create_chat_completion(
            messages=messages, temperature=temp, stream=stream
        )

        return response
