import React from "react";
import VideoPlayer from "./components/VideoPlayer.jsx";
import TelemetryDashboard from "./components/TelemetryDashboard.jsx";
import ModelUploader from "./components/ModelUploader.jsx";
import { useTelemetry } from "./hooks/useTelemetry.js";
// Matches the use case in the spec: 10 intersections monitored concurrently.
// In Week 1 (file-demo mode) every tile streams the same mock video; once
// the backend runs in --mode gpu with real RTSP sources, swap these for
// your actual intersection IDs.
const STREAM_IDS = [
  "intersection-01",
  "intersection-02",
  "intersection-03",
  "intersection-04",
  "intersection-05",
  "intersection-06",
  "intersection-07",
  "intersection-08",
  "intersection-09",
  "intersection-10",
];
export default function App() {
  const { snapshot } = useTelemetry();
  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <div style={styles.brand}>
          <span style={styles.brandMark} className="mono">
            VE
          </span>
          <div>
            <div style={styles.title}>VisionEdge</div>
            <div style={styles.subtitle} className="mono">
              edge-ai traffic ops · zero-copy pipeline
            </div>
          </div>
        </div>
        <div style={styles.headerStats} className="mono">
          <HeaderStat label="STREAMS" value={STREAM_IDS.length} />
          <HeaderStat
            label="VRAM"
            value={`${Math.round(snapshot.gpu?.vram_percent || 0)}%`}
          />
          <HeaderStat
            label="GPU"
            value={`${Math.round(snapshot.gpu?.gpu_util_percent || 0)}%`}
          />
        </div>
      </header>
      <main style={styles.main}>
        <section style={styles.grid} className="scrollbar-thin">
          {STREAM_IDS.map((id) => (
            <VideoPlayer
              key={id}
              streamId={id}
              label={id}
              fps={snapshot.streams?.[id]?.fps}
            />
          ))}
        </section>
        <aside style={styles.sidebar}>
          <TelemetryDashboard streamIds={STREAM_IDS} />
          <ModelUploader streamIds={STREAM_IDS} />
        </aside>
      </main>
    </div>
  );
}
function HeaderStat({ label, value }) {
  return (
    <div style={styles.headerStat}>
      <span style={styles.headerStatLabel}>{label}</span>
      <span style={styles.headerStatValue}>{value}</span>
    </div>
  );
}
const styles = {
  app: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    padding: 20,
    gap: 16,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    paddingBottom: 16,
    borderBottom: "1px solid var(--line)",
  },
  brand: { display: "flex", alignItems: "center", gap: 12 },
  brandMark: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 34,
    height: 34,
    borderRadius: 3,
    background: "var(--accent-cyan-dim)",
    border: "1px solid var(--accent-cyan)",
    color: "var(--accent-cyan)",
    fontSize: 13,
    fontWeight: 700,
  },
  title: { fontSize: 16, fontWeight: 700, letterSpacing: "-0.01em" },
  subtitle: { fontSize: 11, color: "var(--text-dim)", marginTop: 1 },
  headerStats: { display: "flex", gap: 20 },
  headerStat: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flexend",
  },
  headerStatLabel: {
    fontSize: 9,
    color: "var(--text-dim)",
    letterSpacing: "0.08em",
  },
  headerStatValue: {
    fontSize: 15,
    color: "var(--accent-cyan)",
    fontWeight: 700,
  },
  main: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "1fr 320px",
    gap: 16,
    minHeight: 0,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gridAutoRows: "min-content",
    gap: 12,
    overflowY: "auto",
    paddingRight: 4,
  },
  sidebar: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    overflowY: "auto",
  },
};
