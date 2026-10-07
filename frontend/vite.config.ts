/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Geolocation requires a secure context: localhost in development, HTTPS otherwise.
  server: { host: "localhost", port: 5173, strictPort: true },
  test: { environment: "node" },
});
