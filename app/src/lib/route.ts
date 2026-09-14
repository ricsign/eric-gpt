/**
 * Deep links.
 *
 * `napkin.example/4` opens question 4 so a recipient can answer it before seeing
 * anyone's answer. That is the whole point of the link in a shared receipt: a
 * link that opened today's call instead would make the share useless the moment
 * the day rolled over, and a link that revealed the sender's answer would remove
 * the reason to play at all.
 *
 * Pure functions over a pathname, so routing is testable without a browser.
 */

/** Parses a call number out of a path. Returns null for anything else. */
export function callNoFromPath(pathname: string): number | null {
  const m = /^\/(\d{1,6})\/?$/.exec(pathname)
  if (!m) return null
  const n = Number.parseInt(m[1], 10)
  return Number.isSafeInteger(n) && n > 0 ? n : null
}

/** The path a given call lives at. */
export function pathForCall(callNo: number): string {
  return `/${callNo}`
}

/** Absolute URL for sharing. */
export function urlForCall(callNo: number, origin?: string): string {
  const base = origin ?? (typeof location === 'undefined' ? '' : location.origin)
  return `${base}${pathForCall(callNo)}`
}

/**
 * Swaps the address bar to a call without a navigation.
 *
 * Uses replaceState rather than pushState: the loop is linear and the browser
 * back button should leave the app, not step backwards through screens the
 * player has already committed to.
 */
export function setPath(path: string): void {
  if (typeof history === 'undefined') return
  if (location.pathname === path) return
  history.replaceState(null, '', path)
}
