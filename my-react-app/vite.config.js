import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API calls to FastAPI running at 127.0.0.1:8000
      '/face': {
        target: 'http://127.0.0.1:7000',
        changeOrigin: true,
      },
    },
  },
})
