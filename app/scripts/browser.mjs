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

export async function launch(options = {}) {
  return chromium.launch({ executablePath: chromiumPath(), ...options })
}
