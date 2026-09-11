import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/springs.css'
import './styles/tokens.css'
import './styles/base.css'
import { App } from './App'
import { StoreProvider } from './state/store'

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing from index.html')

createRoot(root).render(
  <StrictMode>
    <StoreProvider>
      <App />
    </StoreProvider>
  </StrictMode>,
)

/*
 * Register the service worker so the app opens offline once it has been visited.
 *
 * Deliberately after first paint and in production only: a worker racing the
 * first render buys nothing, and in development it would serve a stale bundle
 * over the dev server's own hot updates.
 */
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // Blocked by a private window, an unsupported browser, or an insecure
      // origin. The app works fine without it; there is nothing to report.
    })
  })
}
