from fastapi import FastAPI, WebSocket

from app.services.agent_service import AgentService
from app.services.context_manager import ContextManager
from app.services.orchestrator import Orchestrator
from app.services.websocket_manager import WebSocketManager

app: FastAPI = FastAPI()

agent_service: AgentService = AgentService()
context_manager: ContextManager = ContextManager()
websocket_manager: WebSocketManager = WebSocketManager()

orchestrator: Orchestrator = Orchestrator(
    agent_service, context_manager, websocket_manager
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    while True:
        await websocket.send_text("connect")

        text = await websocket.receive_text()

        print(text)


@app.get("/chats")
async def get_conversations() -> dict:
    return {"conversations": []}


@app.get("/chats/messages/{conversation_id}")
async def get_messages(conversation_id: str) -> dict:
    return {"id": conversation_id}
