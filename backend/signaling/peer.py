"""WebRTC peer-connection lifecycle management."""

from __future__ import annotations

import asyncio
import logging

from aiortc import RTCPeerConnection, RTCSessionDescription

try:
    from .video_track import VideoTrack
except ImportError:  # Support `python backend/signaling/server.py`.
    from video_track import VideoTrack


LOGGER = logging.getLogger(__name__)


class PeerConnectionManager:
    """Create, monitor, and clean up WebRTC peer connections."""

    def __init__(self) -> None:
        self._connections: set[RTCPeerConnection] = set()

    async def create_answer(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        """Create an answer for an incoming offer and attach a video track."""
        connection = RTCPeerConnection()
        self._connections.add(connection)

        @connection.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = connection.connectionState
            LOGGER.info("WebRTC connection state: %s", state)
            if state in {"failed", "closed"}:
                await connection.close()
                self._connections.discard(connection)

        try:
            await connection.setRemoteDescription(offer)
            connection.addTrack(VideoTrack())
            answer = await connection.createAnswer()
            await connection.setLocalDescription(answer)
            return connection.localDescription
        except Exception:
            await connection.close()
            self._connections.discard(connection)
            raise

    async def close_all(self) -> None:
        """Close every managed connection during application shutdown."""
        await asyncio.gather(
            *(connection.close() for connection in self._connections),
            return_exceptions=True,
        )
        self._connections.clear()
