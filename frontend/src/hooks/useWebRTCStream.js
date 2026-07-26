import { useEffect, useRef, useState } from "react";

/**
 * useWebRTCStream
 *
 * Handles the aiortc signaling handshake: create an RTCPeerConnection,
 * generate an SDP offer, POST it to the backend's /offer endpoint, and
 * apply the returned answer. Mirrors streaming/webrtc_server.py's offer
 * handler exactly.
 */
export function useWebRTCStream(streamId, { autoConnect = true } = {}) {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | connecting | connected | failed

  const connect = async () => {
    if (pcRef.current) return;
    setStatus("connecting");

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    pcRef.current = pc;

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
      }
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "connected") setStatus("connected");
      if (["failed", "disconnected", "closed"].includes(pc.connectionState))
        setStatus("failed");
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    try {
      const res = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          stream_id: streamId,
        }),
      });
      const answer = await res.json();
      await pc.setRemoteDescription(answer);
    } catch (err) {
      console.error("WebRTC signaling failed:", err);
      setStatus("failed");
    }
  };

  const disconnect = () => {
    pcRef.current?.close();
    pcRef.current = null;
    setStatus("idle");
  };

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamId]);

  return { videoRef, status, connect, disconnect };
}
