import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Prebundle the detector at startup, not after cameras have been connected.
  optimizeDeps: { include: ['onnxruntime-web/wasm'] },
  build: {
    rollupOptions: {
      output: { manualChunks: { three: ['three'] } },
    },
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: ['.e2b.app'],
  },
  preview: {
    host: '0.0.0.0',
    allowedHosts: ['.e2b.app'],
  },
});
