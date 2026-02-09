import gc
from typing import Dict, List, Optional

import yaml
from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletion,
    ChatCompletionRequestFunctionCall,
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
)

from src.paths import MODELS_CONFIG_PATH


class AgentService:
    def __init__(self) -> None:
        self.model: Optional[Llama] = None
        self.config = self._load_config()

    def _load_config(self) -> dict | None:
        try:
            with open(MODELS_CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)

        except FileNotFoundError:
            print("File not found: Models Config File Does Not Exist")

            return None

    def load(self, path: str):

        self.model = Llama(model_path=path)

        print("Model loaded succesfully")

    def unload(self) -> None:
        """
        This function unloads the current running SLM by deleting the variable using 'del' keyword
        and using garbage collector to clean up the memory. Then recreates the model variable for the next loading.

        Returns: None
        """

        if self.model:
            self.model = None

            gc.collect()

            self.model = None

    def swap(self, path: str) -> None:
        """
        Docstring for swap


        :param self: Description
        :param role: The role of the model being swapped
        :type role: str
        """

        self.unload()

        self.load(path=path)

        def generate(
            self, messages: List[ChatCompletionRequestMessage], temp: float
        ) -> CreateChatCompletionResponse:

            response = self.model.create_chat_completion(
                messages=messages, temperature=temp
            )

            return response["choices"][0]["message"]["content"]


# UNIT TESTING

agent = AgentService()

agent.unload()
