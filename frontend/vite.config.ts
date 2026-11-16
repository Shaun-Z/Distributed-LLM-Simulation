import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/distributed_llm_simulation/web/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: false } },
  },
});
