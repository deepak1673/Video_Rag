import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs on :8000 in development; proxying keeps the browser same-origin
// (no CORS) and lets <video src> / <img src> use relative /api URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
