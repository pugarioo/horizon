from click.core import F
from fastapi import FastAPI, WebSocket

from app.services.context_manager import ContextManager
from app.services.orchestrator import Orchestrator

app: FastAPI = FastAPI()
context_manager: ContextManager = ContextManager()
orchestrator: Orchestrator = Orchestrator(context_manager)


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
