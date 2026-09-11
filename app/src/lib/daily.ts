/**
 * One question a day, the same one for everybody.
 *
 * Deliberately *not* a growth loop. An earlier version shipped a Wordle-style
 * glyph to share and it was cut on an information-theoretic argument: three
 * attempts at a four-option question yields three distinguishable outcomes, so
 * the glyph encodes almost nothing, the modal result is a perfect score, and it
 * reads as a brag only to someone who already knows the format. Wordle's grid
 * travels because it encodes five letters across six rows of three states. This
 * does not, and shipping the shape without the substance would have been
 * cargo-culting.
 *
 * What survives is the useful part: a shared question everyone gets on the same
 * day, so it can be argued about, with the arithmetic revealed either way.
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
