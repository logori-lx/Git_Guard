import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:886', // ← mock 后端端口
        changeOrigin: true,
        secure: false
      }
    }
  }
})
