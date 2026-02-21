from fastapi import WebSocket

from app.services.utils import Roles, State


class WebSocketManager:
    """
    Manages active WebSocket connections for real-time communication.
    """

    def __init__(self) -> None:
        """
        Initializes the WebSocketManager with an empty dictionary of active connections.
        """
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accepts a new WebSocket connection and registers it with a conversation ID.

        Args:
            conversation_id: The ID of the conversation associated with this connection.
            websocket: The WebSocket instance to connect.
        """
        # self.active_connections[conversation_id] = websocket

        await websocket.accept()

    async def disconnect(self, conversation_id: str):
        """
        Closes and removes a WebSocket connection for a given conversation ID.

        Args:
            conversation_id: The ID of the conversation to disconnect.
        """
        await self.active_connections[conversation_id].close()
        del self.active_connections[conversation_id]

    async def send_status(
        self, websocket: WebSocket, status: State, agent: Roles | None
    ) -> None:
        """
        Sends an event through an active WebSocket connection.

        Args:
            conversation_id: The ID of the conversation (for identification).
            websocket: The WebSocket instance to send the event through.
            event: The event data to send.
        """
        message: dict = {
            "type": "status",
            "status": status.value,
            "agent": agent.value if agent else None,
        }

        await self.send_message(websocket=websocket, content=message)

    async def send_message(
        self,
        websocket: WebSocket,
        content: dict,
    ) -> None:
        """
        Sends a text message through an active WebSocket connection.

        Args:
            websocket: The WebSocket instance to send the message through.
            content: The dictionary content of the message.
        """

        await websocket.send_json(content)
