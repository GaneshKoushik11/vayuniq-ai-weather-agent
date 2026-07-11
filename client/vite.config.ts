import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        tailwindcss(), // Ensure this is loaded first
        react(),
    ],
    server: {
        port: 5173,
        proxy: {
            "/api": {
                target: "http://localhost:8787",
                changeOrigin: true
            }
        }
    }
})
