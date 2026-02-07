import chromadb
import time
import uuid
from src.paths import DB_DIR
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=DB_DIR)

        self.collection = self.client.get_or_create_collection('chat_history')

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def store_memory(self, conversation_id: str, content: str | dict) -> None:
        if isinstance(content, str):
            content = [content]

        self.collection.add(
            documents=content,
            ids=[self._generate_id(conversation_id=conversation_id) for _ in range(len(content))]
        )
    
    def search_context(self, conversatio_id: str, query: str | dict):
        pass
        
    def _generate_id(self, conversation_id: str) -> str:
        timestamp = int(time.time())

        suffix = uuid.uuid4().hex[:4]

        return f"{conversation_id}-{timestamp}-{suffix}"