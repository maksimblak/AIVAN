import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },

  server: {
    port: 3000,
    host: true,
    open: false,
    // Для работы с Telegram Web App через ngrok/cloudflare tunnel
    hmr: {
      overlay: false,
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    // Оптимизация для production
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'icons': ['lucide-react'],
        },
      },
    },
    // Сжатие и минификация
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
  },

  // Оптимизация для Telegram Web App
  define: {
    'process.env': {},
  },
});
