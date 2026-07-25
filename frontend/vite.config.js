import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/offer": "http://localhost:8080",
      "/streams": "http://localhost:8080",
      "/telemetry": "http://localhost:8080",
    },
  },
});
