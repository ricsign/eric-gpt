/*
 * Service worker.
 *
 * Small and hand-written rather than generated: this app is a handful of hashed
 * assets and no API, so a precache manifest would be more moving parts than the
 * problem needs.
 *
 * Two strategies, chosen for what actually goes wrong:
 *  - Navigations are network-first. A cache-first shell is how PWAs end up
 *    serving a version from three weeks ago, and in a finance app a stale
 *    contribution limit is a correctness bug, not a staleness annoyance.
 *  - Hashed build assets are cache-first, because their URL changes whenever
 *    their content does, so a cached copy can never be wrong.
 */

const VERSION = 'v1'
const SHELL = `compound-shell-${VERSION}`
const ASSETS = `compound-assets-${VERSION}`

const SHELL_URLS = ['/', '/index.html', '/manifest.webmanifest', '/icon.svg']

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL)
      // addAll rejects the whole install if any single URL 404s, which would
      // leave the app with no worker at all. Failures here are not fatal.
      .then((cache) => Promise.allSettled(SHELL_URLS.map((u) => cache.add(u))))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== SHELL && k !== ASSETS).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  // Never touch cross-origin requests: caching someone else's response here
  // would be both surprising and a privacy question we do not need to answer.
  if (url.origin !== self.location.origin) return

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone()
          caches.open(SHELL).then((c) => c.put('/index.html', copy))
          return response
        })
        .catch(() => caches.match('/index.html').then((r) => r ?? Response.error())),
    )
    return
  }

  // Hashed assets: content-addressed, so a hit is always correct.
  if (url.pathname.startsWith('/assets/') || /\.(png|svg|woff2?)$/.test(url.pathname)) {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ??
          fetch(request).then((response) => {
            if (response.ok) {
              const copy = response.clone()
              caches.open(ASSETS).then((c) => c.put(request, copy))
            }
            return response
          }),
      ),
    )
  }
})
