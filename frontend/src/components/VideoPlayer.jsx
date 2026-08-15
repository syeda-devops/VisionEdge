import React from "react";
import { useWebRTCStream } from "../hooks/useWebRTCStream.js";

export default function VideoPlayer({ streamId, label, fps }) {
  const { videoRef, status } = useWebRTCStream(streamId);

  return (
    <div style={styles.wrapper} className="hud-frame">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={styles.video}
      />

      <div style={styles.overlayTop}>
        <span style={styles.idBadge} className="mono">{label}</span>
        {status === "connected" && (
          <span style={styles.liveBadge} className="mono">
            <span className="live-dot" /> LIVE
          </span>
        )}
      </div>

      <div style={styles.overlayBottom}>
        <span style={styles.fpsBadge} className="mono">
          {fps != null ? `${fps.toFixed(1)} FPS` : "-- FPS"}
        </span>
        <span style={{ ...styles.statusBadge, ...statusColor(status) }} className="mono">
          {status.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

function statusColor(status) {
  switch (status) {
    case "connected": return { color: "var(--accent-green)" };
    case "connecting": return { color: "var(--accent-amber)" };
    case "failed": return { color: "var(--accent-red)" };
    default: return { color: "var(--text-dim)" };
  }
}

const styles = {
  wrapper: {
    position: "relative",
    background: "#000",
    borderRadius: 2,
    overflow: "hidden",
    aspectRatio: "16 / 9",
    border: "1px solid var(--line)",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
  },
  overlayTop: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  overlayBottom: {
    position: "absolute",
    bottom: 10,
    left: 10,
    right: 10,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  idBadge: {
    fontSize: 11,
    letterSpacing: "0.05em",
    color: "var(--text-primary)",
    background: "rgba(10,13,17,0.7)",
    padding: "3px 7px",
    borderRadius: 2,
  },
  liveBadge: {
    fontSize: 10,
    letterSpacing: "0.08em",
    color: "var(--accent-red)",
    background: "rgba(10,13,17,0.7)",
    padding: "3px 7px",
    borderRadius: 2,
    display: "flex",
    alignItems: "center",
    gap: 5,
  },
  fpsBadge: {
    fontSize: 11,
    color: "var(--accent-cyan)",
    background: "rgba(10,13,17,0.7)",
    padding: "3px 7px",
    borderRadius: 2,
  },
  statusBadge: {
    fontSize: 10,
    letterSpacing: "0.06em",
    background: "rgba(10,13,17,0.7)",
    padding: "3px 7px",
    borderRadius: 2,
  },
};