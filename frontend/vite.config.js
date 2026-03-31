import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const basePath = (() => {
  const configuredBasePath = process.env.VITE_BASE_PATH || '/'
  const normalizedBasePath = configuredBasePath.startsWith('/')
    ? configuredBasePath
    : `/${configuredBasePath}`

  return normalizedBasePath.endsWith('/')
    ? normalizedBasePath
    : `${normalizedBasePath}/`
})()

// https://vitejs.dev/config/
export default defineConfig({
  base: basePath,
  plugins: [
    vue()
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
})
