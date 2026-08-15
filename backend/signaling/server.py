"""HTTP signaling endpoints for the VisionEdge WebRTC browser client."""

from __future__ import annotations

import logging

from aiohttp import web

try:
    from aiortc import RTCSessionDescription
    try:
        from .peer import PeerConnectionManager
    except ImportError:  # Support `python backend/signaling/server.py`.
        from peer import PeerConnectionManager
except ImportError as exc:  # Allows /health to remain useful before setup.
    RTCSessionDescription = None
    PeerConnectionManager = None
    WEBRTC_IMPORT_ERROR = exc
else:
    WEBRTC_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
PEER_MANAGER = PeerConnectionManager() if PeerConnectionManager else None


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "http://localhost:5173",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_cors_headers())
    response = await handler(request)
    response.headers.update(_cors_headers())
    return response


async def hello(request: web.Request) -> web.Response:
    return web.Response(text="VisionEdge Backend Running")


async def health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "message": "VisionEdge Backend Running",
            "webrtc_ready": WEBRTC_IMPORT_ERROR is None,
        }
    )


async def offer(request: web.Request) -> web.Response:
    """Accept a browser SDP offer and return its WebRTC answer."""
    if WEBRTC_IMPORT_ERROR is not None:
        return web.json_response(
            {"error": "WebRTC dependencies are unavailable. Install aiortc and numpy."},
            status=503,
        )

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return web.json_response({"error": "Request body must be valid JSON."}, status=400)

    sdp = payload.get("sdp") if isinstance(payload, dict) else None
    offer_type = payload.get("type") if isinstance(payload, dict) else None
    if not isinstance(sdp, str) or offer_type != "offer":
        return web.json_response(
            {"error": "An SDP offer requires string 'sdp' and type 'offer'."}, status=400
        )

    try:
        answer = await PEER_MANAGER.create_answer(RTCSessionDescription(sdp=sdp, type=offer_type))
        return web.json_response({"sdp": answer.sdp, "type": answer.type})
    except Exception:
        LOGGER.exception("Unable to negotiate WebRTC offer")
        return web.json_response({"error": "Unable to negotiate the WebRTC offer."}, status=500)


async def close_peer_connections(app: web.Application) -> None:
    if PEER_MANAGER:
        await PEER_MANAGER.close_all()


app = web.Application(middlewares=[cors_middleware])
app.router.add_get("/", hello)
app.router.add_get("/health", health)
app.router.add_post("/offer", offer)
app.on_shutdown.append(close_peer_connections)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host="127.0.0.1", port=8080)
