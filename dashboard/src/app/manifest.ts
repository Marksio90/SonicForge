import type { MetadataRoute } from 'next'

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'SonicForge Dashboard - AI Music Radio',
    short_name: 'SonicForge',
    description: 'Professional AI-Powered 24/7 Music Radio Control Center',
    start_url: '/',
    display: 'standalone',
    background_color: '#0a0a0f',
    theme_color: '#6366f1',
    orientation: 'portrait-primary',
    categories: ['music', 'entertainment', 'productivity'],
    icons: [
      {
        src: '/icon-192.png',
        sizes: '192x192',
        type: 'image/png',
        purpose: 'maskable',
      },
      {
        src: '/icon-512.png',
        sizes: '512x512',
        type: 'image/png',
        purpose: 'maskable',
      },
    ],
    shortcuts: [
      {
        name: 'Dashboard',
        url: '/',
        description: 'View main dashboard',
      },
      {
        name: 'Generate Track',
        url: '/generate',
        description: 'Generate new music',
      },
      {
        name: 'Analytics',
        url: '/analytics',
        description: 'View analytics',
      },
    ],
  }
}
