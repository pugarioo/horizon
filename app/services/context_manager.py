import asyncio
import sqlite3
import time
import uuid
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.CollectionCommon import QueryResult
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)

from app.paths import DB_DIR, MODELS_DIR
from app.services.utils import LogEntry


class ContextManager:
    def __init__(self) -> None:
        """
        Initializes the ContextManager, setting up both a ChromaDB client for vector embeddings
        and an SQLite database for structured conversation and message storage.
        """
        # Initialize Agent Logs
        self._agent_logs: list[LogEntry] = []

        # Initialize the vector database client
        self._client: ClientAPI = chromadb.PersistentClient(path=DB_DIR)
        self._embedder: SentenceTransformerEmbeddingFunction = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=str(MODELS_DIR / "all-MiniLM-L6-v2")
            )
        )
        self._collection = self._client.get_or_create_collection(
            name="chat_history",
            embedding_function=self._embedder,  # type: ignore
        )

        # Initialize the SQLite database
        self._db_conn: sqlite3.Connection = sqlite3.connect(
            DB_DIR / "chat_history.db", check_same_thread=False
        )
        self._db_cursor: sqlite3.Cursor = self._db_conn.cursor()
        self._db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                name TEXT,
                timestamp INTEGER
            )
            """
        )
        self._db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                content TEXT,
                timestamp INTEGER,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        self._db_conn.commit()

    async def store_memory(
        self, conversation_id: str, content: str | list[str]
    ) -> None:
        """
        Stores chat content in both the ChromaDB vector collection and the SQLite chat_messages table.

        Args:
            conversation_id: The ID of the conversation to associate the content with.
            content: The chat message content, either as a single string or a list of strings.
        """

        formatted_content: list[str] = (
            [content] if isinstance(content, str) else content
        )

        await asyncio.to_thread(
            self._collection.add,
            documents=formatted_content,
            ids=[
                self._generate_id(conversation_id=conversation_id)
                for _ in range(len(formatted_content))
            ],
            metadatas=[
                {"conversation_id": conversation_id}
                for _ in range(len(formatted_content))
            ],
        )

        def _db_write():
            self._db_cursor.execute(
                """
                INSERT INTO chat_messages (id, conversation_id, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._generate_id(conversation_id=conversation_id),
                    conversation_id,
                    formatted_content[0],
                    int(time.time()),
                ),
            )
            self._db_conn.commit()

        await asyncio.to_thread(_db_write)

    async def search_context(
        self, conversation_id: str, query: str, top_k: int = 3
    ) -> str:
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
        formatted_query: list[str] = [query]

        results: QueryResult = await asyncio.to_thread(
            self._collection.query,
            query_texts=formatted_query,
            n_results=top_k,
            where={"conversation_id": conversation_id},
        )

        result_dict = cast(dict[str, Any], results)

        raw_docs = result_dict.get("documents", [[]])[0]

        if not raw_docs:
            return ""

        unique_docs: list[str] = list(dict.fromkeys(raw_docs))

        formatted_docs = [f"-{doc}" for doc in unique_docs]

        return "\n".join(formatted_docs)

    def get_conversations(self) -> list[Any]:
        """
        Retrieves all conversation metadata from the SQLite 'conversations' table.

        Returns:
            A list of tuples, each representing a conversation's (id, conversation_id, name, timestamp).
        """
        self._db_cursor.execute(
            """
            SELECT conversation_id, name, timestamp
            FROM conversations
            """
        )
        return self._db_cursor.fetchall()

    def get_conversation_messages(self, conversation_id: str) -> list[Any]:
        """
        Fetches chat messages for a specific conversation from the SQLite 'chat_messages' table.

        Args:
            conversation_id: The ID of the conversation to retrieve messages for.

        Returns:
            A list of tuples, each representing a message's (id, content, timestamp).
        """
        self._db_cursor.execute(
            """
            SELECT id, content, timestamp
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            """,
            (conversation_id,),
        )
        return self._db_cursor.fetchall()

    def _generate_id(self, conversation_id: str) -> str:
        """
        Generates a unique ID for a chat message or memory entry.

        Args:
            conversation_id: The ID of the conversation to prefix the generated ID with.

        Returns:
            A unique string identifier combining conversation_id, timestamp, and a UUID suffix.
        """
        timestamp: int = int(time.time())

        suffix: str = uuid.uuid4().hex[:4]

        return f"{conversation_id}-{timestamp}-{suffix}"

    def add_agent_log(self, log: LogEntry) -> None:
        self._agent_logs.append(log)

    async def add_conversation_id(self) -> str:
        conversation_id = str(uuid.uuid4())

        def insert_conversation():
            self._db_cursor.execute(
                """
                INSERT INTO conversations (conversation_id, name, timestamp)
                VALUES (?, ?, ?)
                """,
                (conversation_id, "New Conversation", int(time.time())),
            )
            self._db_conn.commit()

        await asyncio.to_thread(insert_conversation)
        return conversation_id
