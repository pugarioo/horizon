from fastapi import WebSocket


class WebSocketManager:
    """
    Manages active WebSocket connections for real-time communication.
    """
    def __init__(self):
        """
        Initializes the WebSocketManager with an empty dictionary of active connections.
        """
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, conversation_id: str, websocket: WebSocket):
        """
        Accepts a new WebSocket connection and registers it with a conversation ID.

        Args:
            conversation_id: The ID of the conversation associated with this connection.
            websocket: The WebSocket instance to connect.
        """
        self.active_connections[conversation_id] = websocket

        await websocket.accept()

    async def disconnect(self, conversation_id: str):
        """
        Closes and removes a WebSocket connection for a given conversation ID.

        Args:
            conversation_id: The ID of the conversation to disconnect.
        """
        await self.active_connections[conversation_id].close()
        del self.active_connections[conversation_id]

    async def send_message(
        self, conversation_id: str, websocket: WebSocket, content: str
    ):
        """
        Sends a text message through an active WebSocket connection.

        Args:
            conversation_id: The ID of the conversation (for identification).
            websocket: The WebSocket instance to send the message through.
            content: The string content of the message.
        """
        await websocket.send_text(content)
