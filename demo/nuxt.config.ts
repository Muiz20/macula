import { resolve } from 'path'
import { fileURLToPath } from 'url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))
const parentDir = resolve(__dirname, '..')

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

  css: ['~/assets/css/main.css'],

  vite: {
    plugins: [
      // @ts-expect-error tailwindcss vite plugin
      (await import('@tailwindcss/vite')).default(),
    ],
    server: {
      fs: {
        allow: [parentDir],
      },
    },
  },

  nitro: {
    publicAssets: [
      {
        dir: resolve(parentDir, 'zig-out'),
        baseURL: '/zig-out',
        maxAge: 0,
      },
      {
        dir: resolve(parentDir, 'data'),
        baseURL: '/data',
        maxAge: 0,
      },
    ],
  },

  app: {
    head: {
      title: 'Macula — OCR Error Detector',
      meta: [
        { name: 'description', content: 'Client-side OCR error detection powered by WebAssembly' },
      ],
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' },
      ],
    },
  },
})
