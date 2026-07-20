import { useEffect, useRef, useState } from "react";
import "./App.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;
const SIGNALING_URL = BACKEND_URL
  ? `${BACKEND_URL.replace(/\/$/, "")}/offer`
  : null;

function waitForIceGathering(peerConnection) {
  if (peerConnection.iceGatheringState === "complete") return Promise.resolve();

  return new Promise((resolve) => {
    const onGatheringStateChange = () => {
      if (peerConnection.iceGatheringState === "complete") {
        peerConnection.removeEventListener("icegatheringstatechange", onGatheringStateChange);
        resolve();
      }
    };
    peerConnection.addEventListener("icegatheringstatechange", onGatheringStateChange);
  });
}

function describeState(state) {
  return {
    new: "Preparing connection...",
    connecting: "Connecting...",
    connected: "Connected — receiving live video.",
    disconnected: "Connection interrupted.",
    failed: "Connection failed. Retrying is available.",
    closed: "Connection closed.",
  }[state] ?? "Connecting...";
}

function App() {
  const videoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState("Preparing connection...");

  useEffect(() => {
    let disposed = false;
    let receivedStream = null;

    const updateStatus = (status) => {
      if (!disposed) setConnectionStatus(status);
    };

    async function startPlayback() {
      if (!SIGNALING_URL) {
        updateStatus("Video service is not configured. Set VITE_BACKEND_URL and restart the frontend.");
        return;
      }

      const peerConnection = new RTCPeerConnection();
      peerConnectionRef.current = peerConnection;

      peerConnection.onconnectionstatechange = () => {
        updateStatus(describeState(peerConnection.connectionState));
      };

      peerConnection.ontrack = async ({ streams }) => {
        const stream = streams[0];
        if (!stream || !videoRef.current) return;

        receivedStream = stream;
        stream.oninactive = () => updateStatus("Video stream stopped.");
        stream.getVideoTracks().forEach((track) => {
          track.onended = () => updateStatus("Video stream stopped.");
        });
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
          updateStatus("Connected — receiving live video.");
        } catch {
          updateStatus("Video is ready. Press play to start playback.");
        }
      };

      try {
        peerConnection.addTransceiver("video", { direction: "recvonly" });
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        await waitForIceGathering(peerConnection);

        const response = await fetch(SIGNALING_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(peerConnection.localDescription),
        });
        const answer = await response.json();
        if (!response.ok) throw new Error(answer.error || "Signaling request failed.");

        await peerConnection.setRemoteDescription(answer);
      } catch (error) {
        console.error("WebRTC setup failed:", error);
        updateStatus("Unable to connect to the video service. Check the backend.");
        peerConnection.close();
      }
    }

    startPlayback();
    return () => {
      disposed = true;
      receivedStream?.getTracks().forEach((track) => track.stop());
      peerConnectionRef.current?.close();
      peerConnectionRef.current = null;
    };
  }, []);

  return (
    <main className="app">
      <h1>VisionEdge</h1>
      <p className="status" role="status">{connectionStatus}</p>
      <video
        ref={videoRef}
        aria-label="Live VisionEdge video stream"
        autoPlay
        controls
        muted
        playsInline
        width="800"
        onPlaying={() => setConnectionStatus("Connected — receiving live video.")}
        onEnded={() => setConnectionStatus("Video stream stopped.")}
        onError={() => setConnectionStatus("Video playback failed.")}
      />
    </main>
  );
}

export default App;
