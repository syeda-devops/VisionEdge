import React, { useState } from "react";

/**
 * Week 4: "upload a different TensorRT engine file to swap the AI model
 * on the fly." POSTs to /streams/{stream_id}/swap-engine, which the
 * backend (main.py -> orchestration/stream_manager.py) handles by
 * stopping and restarting that stream's pipeline with the new engine.
 */
export default function ModelUploader({ streamIds }) {
  const [selectedStream, setSelectedStream] = useState(streamIds[0] || "");
  const [file, setFile] = useState(null);
  const [state, setState] = useState("idle"); // idle | uploading | done | error

  const handleUpload = async () => {
    if (!file || !selectedStream) return;
    setState("uploading");

    const form = new FormData();
    form.append("engine", file);

    try {
      const res = await fetch(`/streams/${selectedStream}/swap-engine`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      setState("done");
    } catch (err) {
      console.error(err);
      setState("error");
    }
  };

  return (
    <div style={styles.panel}>
      <div style={styles.label} className="mono">
        SWAP_MODEL
      </div>

      <select
        value={selectedStream}
        onChange={(e) => setSelectedStream(e.target.value)}
        style={styles.select}
        className="mono"
      >
        {streamIds.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>

      <label style={styles.dropZone} className="mono">
        {file ? file.name : "Choose .engine file"}
        <input
          type="file"
          accept=".engine"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          style={{ display: "none" }}
        />
      </label>

      <button
        onClick={handleUpload}
        disabled={!file || state === "uploading"}
        style={{
          ...styles.button,
          opacity: !file || state === "uploading" ? 0.5 : 1,
        }}
        className="mono"
      >
        {state === "uploading" ? "SWAPPING..." : "APPLY TO STREAM"}
      </button>

      {state === "done" && (
        <div style={styles.statusOk} className="mono">
          Engine swapped successfully.
        </div>
      )}
      {state === "error" && (
        <div style={styles.statusErr} className="mono">
          Swap failed — check backend logs.
        </div>
      )}
    </div>
  );
}

const styles = {
  panel: {
    background: "var(--bg-panel)",
    border: "1px solid var(--line)",
    borderRadius: 4,
    padding: 16,
  },
  label: {
    fontSize: 10,
    letterSpacing: "0.1em",
    color: "var(--text-dim)",
    marginBottom: 10,
  },
  select: {
    width: "100%",
    background: "var(--bg-panel-raised)",
    border: "1px solid var(--line)",
    color: "var(--text-primary)",
    borderRadius: 3,
    padding: "8px 10px",
    fontSize: 12,
    marginBottom: 8,
  },
  dropZone: {
    display: "block",
    width: "100%",
    background: "var(--bg-panel-raised)",
    border: "1px dashed var(--line-bright)",
    color: "var(--text-dim)",
    borderRadius: 3,
    padding: "14px 10px",
    fontSize: 11,
    textAlign: "center",
    cursor: "pointer",
    marginBottom: 8,
    boxSizing: "border-box",
  },
  button: {
    width: "100%",
    background: "var(--accent-cyan-dim)",
    border: "1px solid var(--accent-cyan)",
    color: "var(--accent-cyan)",
    borderRadius: 3,
    padding: "9px 10px",
    fontSize: 11,
    letterSpacing: "0.05em",
  },
  statusOk: { color: "var(--accent-green)", fontSize: 11, marginTop: 8 },
  statusErr: { color: "var(--accent-red)", fontSize: 11, marginTop: 8 },
};
