import sqlite3
import time
import uuid
from typing import Any

import chromadb
from chromadb.api.models.CollectionCommon import QueryResult
from chromadb.utils import embedding_functions

from src.paths import DB_DIR


class ContextManager:
    def __init__(self) -> None:
        """
        Initializes the ContextManager, setting up both a ChromaDB client for vector embeddings
        and an SQLite database for structured conversation and message storage.
        """
        # Initialize the vector database client
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="chat_history",
            embedding_function=self.embedder,  # type: ignore
        )

        # Initialize the SQLite database
        self.db_conn = sqlite3.connect(DB_DIR / "chat_history.db")
        self.db_cursor = self.db_conn.cursor()
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                name TEXT,
                timestamp INTEGER
            )
            """
        )
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                conversation_id int,
                content TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        self.db_conn.commit()

    def store_memory(self, conversation_id: str, content: str | list[str]) -> None:
        """
        Stores chat content in both the ChromaDB vector collection and the SQLite chat_messages table.

        Args:
            conversation_id: The ID of the conversation to associate the content with.
            content: The chat message content, either as a single string or a list of strings.
        """

        formatted_content: list[str] = (
            [content] if isinstance(content, str) else content
        )

        self.collection.add(
            documents=formatted_content,
            ids=[
                self._generate_id(conversation_id=conversation_id)
                for _ in range(len(content))
            ],
            metadatas=[
                {"conversation_id": conversation_id} for _ in range(len(content))
            ],
        )

        self.db_cursor.execute(
            """
            INSERT INTO chat_messages (id, conversation_id, content)
            VALUES (?, ?, ?)
            """,
            (
                self._generate_id(conversation_id=conversation_id),
                conversation_id,
                formatted_content[0],
            ),
        )
        self.db_conn.commit()

    def search_context(
        self, conversation_id: str, query: str | list[str], top_k: int = 3
    ) -> QueryResult:
        """
        Searches the ChromaDB vector collection for context relevant to a given query
        within a specific conversation.

        Args:
            conversation_id: The ID of the conversation to search within.
            query: The query string or list of strings to search for.
            top_k: The number of top results to return.

        Returns:
            A QueryResult object containing the search results.
        """
        formatted_query: list[str] = [query] if isinstance(query, str) else query

        results = self.collection.query(
            query_texts=formatted_query,
            n_results=top_k,
            where={"conversation_id": conversation_id},
        )

        return results

    def get_conversations(self) -> list[Any]:
        """
        Retrieves all conversation metadata from the SQLite 'conversations' table.

        Returns:
            A list of tuples, each representing a conversation's (id, conversation_id, name, timestamp).
        """
        self.db_cursor.execute(
            """
            SELECT id, conversation_id, name, timestamp
            FROM conversations
            """
        )
        return self.db_cursor.fetchall()

    def get_conversation_messages(self, conversation_id: str) -> list[Any]:
        """
        Fetches chat messages for a specific conversation from the SQLite 'chat_messages' table.

        Args:
            conversation_id: The ID of the conversation to retrieve messages for.

        Returns:
            A list of tuples, each representing a message's (id, content, timestamp).
        """
        self.db_cursor.execute(
            """
            SELECT id, content, timestamp
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            """,
            (conversation_id,),
        )
        return self.db_cursor.fetchall()

    def _generate_id(self, conversation_id: str) -> str:
        """
        Generates a unique ID for a chat message or memory entry.

        Args:
            conversation_id: The ID of the conversation to prefix the generated ID with.

        Returns:
            A unique string identifier combining conversation_id, timestamp, and a UUID suffix.
        """
        timestamp = int(time.time())

        suffix = uuid.uuid4().hex[:4]

        return f"{conversation_id}-{timestamp}-{suffix}"


context_manager = ContextManager()
