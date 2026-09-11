/**
 * Injects the built asset list into the service worker.
 *
 * Without this the worker only caches what it happens to see, and on a first
 * visit it is not yet controlling the page — so the JS and CSS go straight to
 * the network and are never stored. The result is a worker that serves the HTML
 * offline and then a blank screen, which is worse than no worker at all.
 *
 * Runs after `vite build`, when the hashed filenames finally exist.
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs'
import { join, resolve, relative } from 'node:path'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const dist = resolve(__dirname, '../dist')

/** Every file the app needs to boot, as root-relative URLs. */
function collect(dir) {
  const out = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      out.push(...collect(full))
    } else if (!entry.endsWith('.map') && entry !== 'sw.js') {
      out.push('/' + relative(dist, full).split('\\').join('/'))
    }
  }
  return out
}

const assets = collect(dist).sort()
const swPath = join(dist, 'sw.js')
const sw = readFileSync(swPath, 'utf8')

const marker = "const SHELL_URLS = ['/', '/index.html', '/manifest.webmanifest', '/icon.svg']"
if (!sw.includes(marker)) {
  console.error('build-sw: SHELL_URLS marker not found — did public/sw.js change?')
  process.exit(1)
}

// A content hash over the asset list, so a new build always invalidates the old
// caches even if the worker source itself is byte-identical.
const version = assets.join('|').split('').reduce((h, c) => ((h * 31 + c.charCodeAt(0)) | 0), 7)

writeFileSync(
  swPath,
  sw
    .replace(marker, `const SHELL_URLS = ${JSON.stringify(['/', ...assets])}`)
    .replace("const VERSION = 'v1'", `const VERSION = '${(version >>> 0).toString(36)}'`),
)

console.log(`build-sw: precaching ${assets.length} files, version ${(version >>> 0).toString(36)}`)
