import chromadb
import time
import uuid
from src.paths import DB_DIR
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=DB_DIR)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.collection = self.client.get_or_create_collection(
            name='chat_history', 
            embedding_function=self.embedde
        )

    def store_memory(self, conversation_id: str, content: str) -> None:
        if isinstance(content, str):
            content = [content]

        self.collection.add(
            documents=content,
            ids=[self._generate_id(conversation_id=conversation_id) for _ in range(len(content))],
            metadatas=[{"conversation_id": conversation_id} for _ in range(len(content))]
        )
    
    def search_context(self, conversation_id: str, query: str, top_k: int = 3):
        if isinstance(query, str):
            query = [query]

        results = self.collection.query(
            query_texts=query,
            n_results=top_k,
            where={'conversation_id': conversation_id}
        )

        return results
        
    def _generate_id(self, conversation_id: str) -> str:
        timestamp = int(time.time())

        suffix = uuid.uuid4().hex[:4]

        return f"{conversation_id}-{timestamp}-{suffix}"
        
    

