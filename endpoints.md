# Horizon API Endpoints

This document outlines the available endpoints for the Horizon local LLM engine backend. The backend is built with FastAPI.

## Base URL
Assuming default local development: `http://localhost:8000`  
WebSocket Base URL: `ws://localhost:8000`

---

## 1. Get All Conversations
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

---

## 2. Get Messages for a Conversation
Retrieves the message history for a specific conversation, ordered by the latest message.

- **URL:** `/chats/messages/{conversation_id}`
- **Method:** `GET`
- **Path Parameters:**
  - `conversation_id` (string): The UUID of the conversation.
- **Response:**
  ```json
  {
      "conversation_id": "string",
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

## 3. WebSocket Chat Interface
Main connection for sending queries to the LLM orchestrator.

- **URL:** `/ws`
- **Protocol:** `WebSocket`

### Sending Messages (Client -> Server)
The client should send JSON-formatted strings. There are two primary message types:

#### a) Initialize a New Chat
Used to start a brand new conversation without a pre-existing `conversation_id`.
```json
{
    "type": "chat_init",
    "data": {
        "content": "Your initial prompt here"
    }
}
```

#### b) Continue an Existing Chat
Used to send a message in an ongoing conversation.
```json
{
    "type": "chat_msg",
    "conversation_id": "existing-conversation-uuid",
    "data": {
        "content": "Your follow-up prompt here"
    }
}
```

### Receiving Messages (Server -> Client)
The server orchestrates the LLM interaction through `orchestrator.py` and streams responses back over the WebSocket connection. The orchestrator logic handles the actual structure of outbound messages.
