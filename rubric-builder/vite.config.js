import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative paths, so the build works under /statistical-modelling/rubric-builder/.
  base: './',
  // The example rubric is read from rubric-eval's folder, next to this one.
  server: { fs: { allow: ['..'] } },
});
