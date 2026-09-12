import { chromium } from 'playwright'
import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Finds the Chromium this environment actually has.
 *
 * The image ships a pinned browser build under PLAYWRIGHT_BROWSERS_PATH, but the
 * npm-installed Playwright may expect a different build number and refuse to
 * launch. Rather than downloading a second copy, resolve whatever is on disk.
 */
export function chromiumPath() {
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers'
  if (!existsSync(root)) return undefined

  const candidates = readdirSync(root)
    .filter((d) => d.startsWith('chromium-'))
    .map((d) => join(root, d, 'chrome-linux', 'chrome'))
    .filter(existsSync)

  return candidates[0]
}

/**
 * Proxy settings for the sandboxed environment, if one is configured.
 *
 * Chromium does not read HTTPS_PROXY the way curl and Node do, so without this
 * it simply cannot reach anything off-box — which is exactly what you need when
 * pointing these scripts at a deployed URL rather than a local server. The CA
 * for the proxy is already in the browser's trust store, so nothing about
 * certificate verification is relaxed here.
 */
function proxyConfig() {
  const server = process.env.HTTPS_PROXY || process.env.https_proxy
  if (!server) return undefined

  const bypass = (process.env.NO_PROXY || process.env.no_proxy || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .join(',')

  return bypass ? { server, bypass } : { server }
}

export async function launch(options = {}) {
  return chromium.launch({
    executablePath: chromiumPath(),
    proxy: proxyConfig(),
    ...options,
  })
}
