import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useTelemetry } from "../hooks/useTelemetry.js";

const CHART_COLORS = ["#4fd1e8", "#5fe3a1", "#ffb454", "#ff6b6b", "#c792ea"];

export default function TelemetryDashboard({ streamIds }) {
  const { snapshot, history } = useTelemetry();
  const gpu = snapshot.gpu || {};

  return (
    <div style={styles.panel} className="scrollbar-thin">
      <SectionLabel text="GPU_TELEMETRY" />

      <div style={styles.gaugeRow}>
        <Gauge
          label="VRAM"
          value={gpu.vram_percent}
          unit="%"
          sub={`${Math.round(gpu.vram_used_mb || 0)} / ${Math.round(gpu.vram_total_mb || 0)} MB`}
        />
        <Gauge label="GPU UTIL" value={gpu.gpu_util_percent} unit="%" />
        <Gauge label="DECODER" value={gpu.decoder_util_percent} unit="%" fallback="N/A" />
      </div>

      <SectionLabel text="STREAM_FPS" />
      <div style={styles.chartBox}>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={history}>
            <CartesianGrid stroke="#232a35" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="t" stroke="#7c8797" fontSize={10} tickLine={false} axisLine={{ stroke: "#232a35" }} />
            <YAxis stroke="#7c8797" fontSize={10} tickLine={false} axisLine={{ stroke: "#232a35" }} />
            <Tooltip
              contentStyle={{ background: "#171c24", border: "1px solid #34404f", fontSize: 12 }}
              labelStyle={{ color: "#7c8797" }}
            />
            {streamIds.map((id, i) => (
              <Line
                key={id}
                type="monotone"
                dataKey={`fps_${id}`}
                name={id}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <SectionLabel text="ACTIVE_STREAMS" />
      <div style={styles.streamList}>
        {streamIds.map((id, i) => (
          <div key={id} style={styles.streamRow} className="mono">
            <span style={{ ...styles.dot, background: CHART_COLORS[i % CHART_COLORS.length] }} />
            <span style={{ flex: 1 }}>{id}</span>
            <span style={{ color: "var(--accent-cyan)" }}>
              {(snapshot.streams?.[id]?.fps ?? 0).toFixed(1)} fps
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionLabel({ text }) {
  return <div style={styles.sectionLabel} className="mono">{text}</div>;
}

function Gauge({ label, value, unit, sub, fallback = "0" }) {
  const display = value == null ? fallback : `${Math.round(value)}${unit}`;
  return (
    <div style={styles.gauge}>
      <div style={styles.gaugeLabel} className="mono">{label}</div>
      <div style={styles.gaugeValue} className="mono">{display}</div>
      {sub && <div style={styles.gaugeSub} className="mono">{sub}</div>}
    </div>
  );
}

const styles = {
  panel: {
    background: "var(--bg-panel)",
    border: "1px solid var(--line)",
    borderRadius: 4,
    padding: 16,
    height: "100%",
    overflowY: "auto",
  },
  sectionLabel: {
    fontSize: 10,
    letterSpacing: "0.1em",
    color: "var(--text-dim)",
    marginBottom: 10,
    marginTop: 18,
  },
  gaugeRow: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: 8,
  },
  gauge: {
    background: "var(--bg-panel-raised)",
    border: "1px solid var(--line)",
    borderRadius: 3,
    padding: "10px 8px",
    textAlign: "center",
  },
  gaugeLabel: { fontSize: 9, color: "var(--text-dim)", letterSpacing: "0.06em" },
  gaugeValue: { fontSize: 20, color: "var(--accent-cyan)", marginTop: 4, fontWeight: 700 },
  gaugeSub: { fontSize: 9, color: "var(--text-dim)", marginTop: 2 },
  chartBox: {
    background: "var(--bg-panel-raised)",
    border: "1px solid var(--line)",
    borderRadius: 3,
    padding: "8px 4px 0 0",
  },
  streamList: { display: "flex", flexDirection: "column", gap: 6 },
  streamRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 12,
    background: "var(--bg-panel-raised)",
    border: "1px solid var(--line)",
    borderRadius: 3,
    padding: "7px 10px",
  },
  dot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
};