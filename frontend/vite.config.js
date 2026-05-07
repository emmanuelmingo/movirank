import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/search":        "http://127.0.0.1:8000",
      "/user":          "http://127.0.0.1:8000",
      "/feature-store": "http://127.0.0.1:8000",
      "/ab":            "http://127.0.0.1:8000",
    },
  },
})
