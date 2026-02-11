from fastapi import WebSocket


class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, conversation_id: str, websocket: WebSocket):
        self.active_connections[conversation_id] = websocket

        await websocket.accept()

    async def disconnect(self, conversation_id: str):
        await self.active_connections[conversation_id].close()
        del self.active_connections[conversation_id]

    async def send_message(
        self, conversation_id: str, websocket: WebSocket, content: str
    ):
        await websocket.send_text(content)
