import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.services.agent_service import AgentService
from app.services.context_manager import ContextManager
from app.services.orchestrator import Orchestrator
from app.services.websocket_manager import WebSocketManager

app: FastAPI = FastAPI()

agent_service: AgentService = AgentService()
context_manager: ContextManager = ContextManager()
websocket_manager: WebSocketManager = WebSocketManager()

orchestrator: Orchestrator = Orchestrator(
    agent_service=agent_service,
    context_manager=context_manager,
    websocket_manager=websocket_manager,
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message: dict = json.loads(data)
            type: str = message.get("type")
            query: str = message.get("data").get("content")
            id: str = ""

            if type == "chat_init":
                id = await context_manager.add_conversation_id()
            elif type == "chat_msg":
                id = message.get("conversation_id")

            await orchestrator.execute_orchestration(
                user_prompt=query,
                conversation_id=id,
                websocket=websocket,
            )
    except WebSocketDisconnect:
        await agent_service.unload()
    except Exception as e:
        print(f"WebSocket error: {e}")
        await agent_service.unload()


@app.get("/chats")
def get_conversations() -> dict:
    conversations = context_manager.get_conversations()
    return {
        "conversations": [
            {"conversation_id": c[0], "name": c[1], "timestamp": c[2]}
            for c in conversations
        ]
    }


@app.get("/chats/messages/{conversation_id}")
def get_messages(conversation_id: str) -> dict:
    messages = context_manager.get_conversation_messages(conversation_id)
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"id": m[0], "content": m[1], "timestamp": m[2]} for m in messages
        ],
    }
