from llama_cpp import Llama
from src.paths import MODELS_CONFIG_PATH, MODELS_DIR
from typing import Any
import yaml
import gc

class AgentService:
    def __init__(self) -> None:
        self.model: Llama | None = None
        self.config = self._load_config()

    def _load_config(self) -> dict[str: Any] | None:
        try:
            with open(MODELS_CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
            
        except FileNotFoundError:
            print('File not found: Models Config File Does Not Exist')

            return None

    def load(self, role: str):
        path = f"{MODELS_DIR}/{self.config['agents'][role]['model_filename']}"

        self.model = Llama(model_path=path)

        print(f'Model {role} loaded succesfully')

    def unload(self) -> None:
        '''
        This function unloads the current running SLM by deleting the variable using 'del' keyword 
        and using garbage collector to clean up the memory. Then recreates the model variable for the next loading.
        
        Returns: None
        '''

        if (self.model):
            del self.model

            gc.collect()
            
            self.model: Llama | None = None
        

    def swap(self, role: str) -> None:
        '''
        Docstring for swap

        
        :param self: Description
        :param role: The role of the model being swapped
        :type role: str
        '''

        self.unload()

        self.load(role=role)

    def generate(self, prompt: str) -> str:
        if not self.model:
            return "Error: No Model Loaded"
        
        


# UNIT TESTING

# agent = AgentService()

# agent.load('critic')