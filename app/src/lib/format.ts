/**
 * Number and date formatting.
 *
 * Money in this app spans five orders of magnitude — a $7 subscription and a
 * $2.4M retirement projection appear on the same screen — so there are two
 * distinct currency formatters rather than one compromise.
 */

const CURRENCY = 'USD'
const LOCALE = 'en-US'

const full = new Intl.NumberFormat(LOCALE, {
  style: 'currency',
  currency: CURRENCY,
  maximumFractionDigits: 0,
})

const exact = new Intl.NumberFormat(LOCALE, {
  style: 'currency',
  currency: CURRENCY,
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

/** `$1,284,300`. For anything the user should read precisely. */
export function money(n: number): string {
  if (!Number.isFinite(n)) return '—'
  return full.format(Math.round(n))
}

/** `$1,284.30`. For small amounts where cents carry meaning. */
export function moneyExact(n: number): string {
  if (!Number.isFinite(n)) return '—'
  return exact.format(n)
}

/**
 * `$1.28M`. For headline figures and chart axes, where the magnitude is the
 * message and the digits are noise.
 *
 * Hand-rolled rather than using Intl's `notation: 'compact'` because that produces
 * "$1.3M" with inconsistent precision across magnitudes; a projection headline needs
 * a stable number of significant digits so it does not visibly change shape as it
 * animates upward.
 */
export function moneyCompact(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return '—'
  const sign = n < 0 ? '-' : ''
  const v = Math.abs(n)

  if (v < 1000) return `${sign}$${Math.round(v)}`
  if (v < 1_000_000) {
    const k = v / 1000
    return `${sign}$${k < 10 ? k.toFixed(digits) : Math.round(k)}K`
  }
  if (v < 1_000_000_000) {
    const m = v / 1_000_000
    return `${sign}$${m < 10 ? m.toFixed(digits + 1) : m.toFixed(digits)}M`
  }
  return `${sign}$${(v / 1_000_000_000).toFixed(digits + 1)}B`
}

/** `7%` / `7.5%` — trailing zeros trimmed, because `7.0%` reads like a spec sheet. */
export function percent(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return '—'
  const v = n * 100
  const s = Math.abs(v % 1) < 0.05 ? v.toFixed(0) : v.toFixed(digits)
  return `${s}%`
}

/** `+12%` / `-3%`. Sign always shown — used for deltas. */
export function signedPercent(n: number, digits = 0): string {
  if (!Number.isFinite(n)) return '—'
  const v = n * 100
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`
}

/** `2 years 4 months`, `8 months`, `3 years`. Never `2.33 years`. */
export function duration(months: number): string {
  if (!Number.isFinite(months)) return 'never'
  const m = Math.max(0, Math.round(months))
  if (m === 0) return 'today'
  if (m < 12) return `${m} month${m === 1 ? '' : 's'}`

  const years = Math.floor(m / 12)
  const rem = m % 12
  const y = `${years} year${years === 1 ? '' : 's'}`
  if (rem === 0) return y
  return `${y} ${rem} month${rem === 1 ? '' : 's'}`
}

/** `34` from 33.748 — used wherever a partial final month still means one more month. */
export function wholeMonths(months: number): number {
  return Number.isFinite(months) ? Math.ceil(months) : Infinity
}

/** `1st`, `2nd`, `23rd`. */
export function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd']
  const v = n % 100
  return n + (s[(v - 20) % 10] ?? s[v] ?? s[0])
}

/** Parses whatever a person types into a money field: `$1,200`, `1.2k`, `2M`. */
export function parseMoney(raw: string): number | null {
  const cleaned = raw.trim().toLowerCase().replace(/[$,\s]/g, '')
  if (!cleaned) return null

  const m = /^(-?\d*\.?\d+)([km])?$/.exec(cleaned)
  if (!m) return null

  const n = Number.parseFloat(m[1])
  if (!Number.isFinite(n)) return null

  const scale = m[2] === 'k' ? 1000 : m[2] === 'm' ? 1_000_000 : 1
  return n * scale
}

/** Clamp, because every slider needs it. */
export function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n))
}

/** Days between two YYYY-MM-DD day keys. */
export function daysBetween(a: string, b: string): number {
  const toUtc = (s: string) => {
    const [y, m, d] = s.split('-').map(Number)
    return Date.UTC(y, m - 1, d)
  }
  return Math.round((toUtc(b) - toUtc(a)) / 86_400_000)
}

/**
 * The app's day boundary, in the user's local timezone.
 *
 * Streaks are compared on this key rather than on timestamps, so a session at
 * 11:58pm and one at 12:02am correctly count as two different days — and crossing a
 * timezone never silently breaks a streak.
 */
export function dayKey(date: Date = new Date()): string {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

/**
 * A date some number of months from today, as a day key.
 * Used to turn "34 months" into an actual date the user can point at.
 */
export function dateInMonths(months: number, from: Date = new Date()): string | null {
  if (!Number.isFinite(months)) return null
  const d = new Date(from.getFullYear(), from.getMonth(), 1)
  d.setMonth(d.getMonth() + Math.max(0, Math.round(months)))
  return dayKey(d)
}

/** `March 2029`, or `Mar 2029` when space is tight. */
export function monthYear(key: string, short = false): string {
  const [y, m] = key.split('-').map(Number)
  return new Date(y, m - 1, 1).toLocaleDateString('en-US', {
    month: short ? 'short' : 'long',
    year: 'numeric',
  })
}

/**
 * How much sooner or later one date is than another, in plain words.
 * Returns null when they are the same month, so the UI can say nothing.
 */
export function dateShift(from: string, to: string): string | null {
  const months = (a: string, b: string) => {
    const [ay, am] = a.split('-').map(Number)
    const [by, bm] = b.split('-').map(Number)
    return (by - ay) * 12 + (bm - am)
  }
  const delta = months(from, to)
  if (delta === 0) return null

  const n = Math.abs(delta)
  const unit = n < 12 ? `${n} month${n === 1 ? '' : 's'}` : duration(n)
  return `${unit} ${delta < 0 ? 'earlier' : 'later'}`
}
