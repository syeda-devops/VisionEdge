import { useEffect, useState } from "react";
import "./App.css";

const BACKEND_URL = "http://localhost:8080/health";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

  useEffect(() => {
    async function checkBackend() {
      try {
        const response = await fetch(BACKEND_URL);
        const data = await response.json();
        setBackendStatus(data.message);
      } catch {
        setBackendStatus("Backend unavailable. Start the Python server on port 8080.");
      }
    }

    checkBackend();
  }, []);

  return (
    <main className="app">
      <h1>VisionEdge</h1>
      <p className="status">{backendStatus}</p>

      <video
        aria-label="Video preview placeholder"
        autoPlay
        controls
        playsInline
        width="800"
      />
    </main>
  );
}

export default App;
