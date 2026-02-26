from fastapi import WebSocket

from app.services.utils import Roles, State


class WebSocketManager:
    """
    Manages active WebSocket connections for real-time communication.
    """

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

    async def send_title(
        self,
        websocket: WebSocket,
        title: str,
    ) -> None:
        """
        Sends a title message through an active WebSocket connection.

        Args:
            websocket: The WebSocket instance to send the message through.
            title: The title to send.
        """

        await websocket.send_json({"type": "title", "title": title})
