import { useEffect, useRef, useState } from "react";

/**
 * useTelemetry
 *
 * Polls GET /telemetry (backend/main.py -> benchmark/telemetry.py's
 * TelemetryHub.snapshot()) and keeps a rolling history for charting.
 */
export function useTelemetry(intervalMs = 1000, historyLength = 30) {
  const [snapshot, setSnapshot] = useState({ streams: {}, gpu: {} });
  const [history, setHistory] = useState([]);
  const timerRef = useRef(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch("/telemetry");
        const data = await res.json();
        setSnapshot(data);
        setHistory((prev) => {
          const point = {
            t: new Date(data.timestamp * 1000).toLocaleTimeString(),
            vram: data.gpu?.vram_percent ?? 0,
            gpuUtil: data.gpu?.gpu_util_percent ?? 0,
            ...Object.fromEntries(
              Object.entries(data.streams || {}).map(([id, s]) => [`fps_${id}`, s.fps])
            ),
          };
          const next = [...prev, point];
          return next.length > historyLength ? next.slice(next.length - historyLength) : next;
        });
      } catch (err) {
        // Backend not reachable yet (e.g. still starting) — fail quietly,
        // dashboard just shows the last known values / zeros.
      }
    };

    poll();
    timerRef.current = setInterval(poll, intervalMs);
    return () => clearInterval(timerRef.current);
  }, [intervalMs, historyLength]);

  return { snapshot, history };
}