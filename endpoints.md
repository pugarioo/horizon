# Horizon API Endpoints

This document outlines the available endpoints for the Horizon local LLM engine backend built with FastAPI, specifically mapping out the required structure of all messages exchanged for proper frontend integration.

## Base URLs
- **REST API Base URL:** `http://localhost:8000` (Local development)
- **WebSocket Base URL:** `ws://localhost:8000`

---

## 1. REST Endpoints

### 1.1 Get All Conversations
Retrieves a list of all chat conversations, ordered by the most recent timestamp.

- **URL:** `/chats`
- **Method:** `GET`
- **Response:**
  ```json
  {
      "conversations": [
          {
              "conversation_id": "string (UUID)",
              "name": "string",
              "timestamp": "integer (Unix timestamp)"
          }
      ]
  }
  ```

### 1.2 Get Messages for a Conversation
Retrieves the message history for a specific conversation, ordered by the latest message.

- **URL:** `/chats/messages/{conversation_id}`
- **Method:** `GET`
- **Path Parameters:**
  - `conversation_id` (string): The UUID of the conversation.
- **Response:**
  ```json
  {
      "conversation_id": "string (UUID)",
      "messages": [
          {
              "id": "string",
              "sent_by": "string", 
              "content": "string",
              "timestamp": "integer (Unix timestamp)"
          }
      ]
  }
  ```

---

## 2. WebSocket Chat Interface

The WebSocket interface is the main connection for sending queries to the LLM orchestrator and streaming responses back. 

- **URL:** `/ws`
- **Protocol:** `WebSocket`

### 2.1 Sending Messages (Client → Server)
The client should send JSON-formatted strings. There are two primary message types:

#### a) Initialize a New Chat
Used to start a brand new conversation without a pre-existing `conversation_id`.
```json
{
    "type": "chat_init",
    "data": {
        "content": "string (Your initial prompt here)"
    }
}
```

#### b) Continue an Existing Chat
Used to send a message in an ongoing conversation.
```json
{
    "type": "chat_msg",
    "conversation_id": "string (existing-conversation-uuid)",
    "data": {
        "content": "string (Your follow-up prompt here)"
    }
}
```

### 2.2 Receiving Messages (Server → Client)
The server orchestrates the LLM interaction through `orchestrator.py` and streams updates and responses back over the WebSocket connection. The messages follow standard structures based on different workflow steps.

#### a) Token Stream Update
Sends individual token pieces as the model generates the final response. These should be appended together on the frontend to display the streamed answer.
```json
{
    "type": "token",
    "content": "string (the generated token payload)"
}
```

#### b) Orchestration Status Update
Broadcasts the current execution state of the engine, particularly when swapping between models or loading components.
```json
{
    "type": "status",
    "status": "string (IDLE | LOADING_MODEL | SWAPPING_MODEL | LOADED | GENERATING | ERROR)",
    "agent": "string (GENERATOR | CRITIC | JUDGE) or null"
}
```

#### c) Evaluation Path Status
Informs the frontend of the current path being processed (e.g., executing multiple model responses to be subsequently judged).
```json
{
    "type": "path",
    "path": "string ('Path A' or 'Path B')"
}
```

#### d) Conversation Metadata (Title Generation)
Once a new chat initializes, the server generates an auto-title for the conversation. 
```json
{
    "type": "title",
    "title": "string (The auto-generated title for the chat)"
}
```
*(Note: As currently implemented in the backend, the newly generated `conversation_id` is kept internal via `chat_init` and managed server-side. Wait for backend updates or poll REST API if you require immediate ID association on init, or check if the backend adds `conversation_id` to this message payload in future updates.)*
