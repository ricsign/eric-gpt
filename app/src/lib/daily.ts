/**
 * The Daily Drill: one question, the same one for everybody, every day.
 *
 * This is the app's growth loop, and its design is constrained by one hard fact
 * about this category: money is the most taboo personal data on the internet. Any
 * share artifact containing a dollar figure of the user's own — net worth, salary,
 * debt, balance — does not get shared, however good it looks in a design review.
 *
 * So the shared artifact contains no personal financial information at all. It is a
 * Wordle-style glyph: how many tries you took, and your streak. Pure performance,
 * zero disclosure. The question is identical worldwide so a group chat can compare,
 * and the result string carries no tracking link, because a link turns a result into
 * an ad and people stop pasting it.
 */

import { dayKey } from './format'

/** Day 1. Fixed forever — the puzzle number is derived from it. */
export const EPOCH = '2026-01-01'

/** Everyone gets the same puzzle on the same calendar day, in their own timezone. */
export function drillNumber(today: string = dayKey()): number {
  const toUtc = (s: string) => {
    const [y, m, d] = s.split('-').map(Number)
    return Date.UTC(y, m - 1, d)
  }
  return Math.floor((toUtc(today) - toUtc(EPOCH)) / 86_400_000) + 1
}

/**
 * Picks the day's question.
 *
 * A plain `n % length` would cycle the bank in a fixed, guessable order and, worse,
 * would show the same question on the same weekday forever. Multiplying by a number
 * coprime to the bank size walks the whole bank in a scrambled order that still
 * visits every question exactly once per cycle.
 */
export function drillIndex(n: number, bankSize: number): number {
  if (bankSize <= 0) return 0
  // 9973 is prime, so it is coprime to any bank size that is not a multiple of it.
  return ((n * 9973) % bankSize + bankSize) % bankSize
}

export const MAX_ATTEMPTS = 3

export interface DrillResult {
  number: number
  /** One entry per attempt taken, in order. */
  attempts: boolean[]
  solved: boolean
  streak: number
}

/**
 * The shareable glyph.
 *
 * Deliberately spoiler-free: the squares say how many tries were taken, never which
 * option was picked, so pasting it into a group chat cannot ruin the puzzle for
 * anyone who has not played. That property is the entire reason Wordle's grid
 * travelled, and it is why this is the one finance share artifact with no taboo
 * attached to it.
 */
export function glyph(result: DrillResult): string {
  return result.attempts.map((ok) => (ok ? '🟩' : '⬛')).join('')
}

export interface ShareOptions {
  /** Bare domain, on its own line. Never a tracking URL — that reads as spam. */
  domain?: string
}

/**
 * Builds the exact string that goes on the clipboard.
 *
 * No URL with query parameters, no UTM tags, no "I scored X, beat me!" copy. A
 * bare domain on its own line is what Wordle settled on after removing its link,
 * and it is what makes the paste feel like a person rather than a referral.
 */
export function shareText(result: DrillResult, { domain = 'compound.money' }: ShareOptions = {}): string {
  const score = result.solved ? `${result.attempts.length}/${MAX_ATTEMPTS}` : `X/${MAX_ATTEMPTS}`
  const streak = result.streak > 1 ? ` · ${result.streak}🔥` : ''
  return `Compound #${result.number} · ${score}${streak}\n${glyph(result)}\n${domain}`
}

/**
 * Copies the result, preferring the native share sheet on a phone.
 *
 * `navigator.share` is what makes this feel like an app rather than a website — it
 * opens iMessage, WhatsApp and the rest directly, which is exactly where money gets
 * discussed. Clipboard is the fallback, and the caller shows a confirmation either way.
 */
export async function shareResult(result: DrillResult): Promise<'shared' | 'copied' | 'failed'> {
  const text = shareText(result)

  if (typeof navigator !== 'undefined' && navigator.share) {
    try {
      await navigator.share({ text })
      return 'shared'
    } catch (err) {
      // AbortError means the user dismissed the sheet: that is not a failure, and
      // silently falling back to the clipboard would be surprising.
      if (err instanceof Error && err.name === 'AbortError') return 'failed'
    }
  }

  try {
    await navigator.clipboard.writeText(text)
    return 'copied'
  } catch {
    return 'failed'
  }
}
