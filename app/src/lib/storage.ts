/**
 * Persistence.
 *
 * Everything this app knows lives in localStorage on the user's own device. That is
 * a product decision as much as a technical one: a finance app that asks for an
 * account before it has proved its worth loses most of its funnel, and one that
 * never uploads your salary is one you can trust on first use.
 *
 * Every access is wrapped: Safari private mode, disabled site data and quota
 * exhaustion all throw on plain `localStorage.getItem`, and none of them should be
 * able to white-screen the app.
 */

const PREFIX = 'compound.'

/** Bumped when a stored shape changes incompatibly; older payloads are discarded. */
const SCHEMA_VERSION = 1

interface Envelope<T> {
  v: number
  data: T
}

export function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    if (!raw) return fallback

    const parsed = JSON.parse(raw) as Envelope<T>
    if (!parsed || typeof parsed !== 'object' || parsed.v !== SCHEMA_VERSION) return fallback
    return parsed.data
  } catch {
    return fallback
  }
}

export function write<T>(key: string, data: T): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify({ v: SCHEMA_VERSION, data }))
  } catch {
    // Quota or private mode. The app keeps working from memory for this session.
  }
}

export function remove(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key)
  } catch {
    /* nothing to do */
  }
}

/** Wipes every key this app owns, leaving other origins' data alone. */
export function clearAll(): void {
  try {
    const doomed: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k?.startsWith(PREFIX)) doomed.push(k)
    }
    doomed.forEach((k) => localStorage.removeItem(k))
  } catch {
    /* nothing to do */
  }
}

/** True when writes actually persist — used to warn before a long session is lost. */
export function isPersistent(): boolean {
  try {
    const probe = `${PREFIX}__probe`
    localStorage.setItem(probe, '1')
    localStorage.removeItem(probe)
    return true
  } catch {
    return false
  }
}
