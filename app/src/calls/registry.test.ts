import { describe, expect, it } from 'vitest'
import { CALLS, callById, callCount } from './registry'
import { DEFAULT_PROFILE, judge, resolveOptimal, stepCount, type Profile } from './types'

/**
 * An optimum may depend on the player's profile, so every record-shape
 * assertion has to hold for the whole spread of players, not just the default.
 */
const PROFILES: Profile[] = [
  { salary: 28_000, age: 24 },
  DEFAULT_PROFILE,
  { salary: 145_000, age: 41 },
  { salary: 320_000, age: 55 },
]

/**
 * These tests guard the invariants that fail *silently*.
 *
 * A crowd array one element short does not throw — it renders a histogram with a
 * missing bar and a subtly wrong mode, and nobody notices for a month. A start
 * value that drifts onto the optimum does not throw either; it just quietly turns
 * the day's call into a no-op where the player locks in without moving the control.
 * Everything here exists because it would otherwise ship looking fine.
 */

/** Names in compute.ts. Typos here are only caught at runtime, on the day. */
const COMPUTE_FNS = [
  'employerMatch',
  'debtSplit',
  'emergencyFund',
  'promoDeadline',
  'anchorOffer',
  'withholding',
  'repairOrReplace',
  'feeDragCall',
  'rentVsBuy',
  'timingMarket',
]

/**
 * Imperative openers. Deliberately wider than the ten verbs currently in use so a
 * new call can be written without editing this list, but narrow enough that the
 * failure modes we actually care about — a rule opening on an article ("A 0% offer
 * is..."), a noun ("Highest rate first") or a number ("Three months of costs") —
 * still fail. A rule that is not an instruction is not a rule.
 */
const IMPERATIVE_VERBS = new Set([
  'avoid', 'ask', 'bank', 'buy', 'check', 'choose', 'clear', 'compare', 'count',
  'counter', 'cover', 'cut', 'delay', 'do', 'drop', 'find', 'fix', 'get', 'give',
  'go', 'hold', 'ignore', 'keep', 'kill', 'know', 'leave', 'let', 'lock', 'look',
  'make', 'move', 'name', 'never', 'open', 'pay', 'put', 'read', 'refuse', 'rent',
  'save', 'sell', 'set', 'skip', 'split', 'start', 'stay', 'stop', 'take', 'treat',
  'owe', 'use', 'wait', 'walk', 'want', 'watch',
])

const words = (s: string) => s.trim().split(/\s+/).filter(Boolean)

describe('the set of calls', () => {
  it('is exactly ten, numbered 1..10 with no gaps or duplicates', () => {
    expect(callCount).toBe(10)
    expect(CALLS.map((c) => c.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  })

  it('looks every call up by id, and misses cleanly', () => {
    for (const c of CALLS) expect(callById(c.id)).toBe(c)
    expect(callById(99)).toBeUndefined()
    expect(callById(0)).toBeUndefined()
  })

  it('uses each compute function exactly once', () => {
    const used = CALLS.map((c) => c.compute)
    expect(new Set(used).size).toBe(used.length)
    for (const name of used) expect(COMPUTE_FNS).toContain(name)
  })

  it('chains tomorrow-teasers into a loop, each one quoting the next title', () => {
    // The teaser is the last thing on the receipt, so a stale one advertises a call
    // that no longer exists. Tying it to the next title makes that impossible.
    CALLS.forEach((c, i) => {
      const next = CALLS[(i + 1) % CALLS.length]
      expect(c.tomorrow, `call ${c.id} teaser`).toBe(next.title)
    })
  })
})

describe.each(CALLS.map((c) => [c.id, c] as const))('call %i', (_id, call) => {
  it('has a title that fits three lines and is not a question', () => {
    expect(words(call.title).length).toBeLessThanOrEqual(12)
    expect(call.title.endsWith('?')).toBe(false)
    expect(call.title.trim()).toBe(call.title)
  })

  it('states no figure in its title that its own facts do not back up', () => {
    // Titles are written months before the facts get re-sourced. If a headline says
    // "50c" and the match tile later reads 25%, the app is lying on its loudest
    // screen. Every digit in a title has to be corroborated by a tile or by the
    // range of the control itself.
    const haystack = [
      ...call.fixed.map((f) => `${f.k} ${f.v}`),
      String(call.variable.min),
      String(call.variable.max),
      String(call.variable.step),
    ]
      .join(' ')
      .replace(/,/g, '')

    for (const token of call.title.match(/\d+/g) ?? []) {
      // Digit boundaries, so "5" does not quietly pass by matching inside "500".
      expect(new RegExp(`(?<!\\d)${token}(?!\\d)`).test(haystack), `"${token}" in title`).toBe(true)
    }
  })

  it('has a rule that is short and gives an instruction', () => {
    expect(words(call.rule).length).toBeLessThan(10)
    const first = words(call.rule)[0].toLowerCase().replace(/[^a-z-]/g, '')
    expect(IMPERATIVE_VERBS.has(first), `rule opens on "${first}"`).toBe(true)
  })

  it('has exactly two fact tiles and a stated assumption', () => {
    expect(call.fixed).toHaveLength(2)
    for (const f of call.fixed) {
      expect(f.k.length).toBeGreaterThan(0)
      expect(f.k).toBe(f.k.toUpperCase())
      expect(f.v.length).toBeGreaterThan(0)
    }
    expect(call.assumptions.trim().length).toBeGreaterThan(0)
  })

  it('has a variable whose range divides evenly into steps', () => {
    const { min, max, step } = call.variable
    expect(step).toBeGreaterThan(0)
    expect(max).toBeGreaterThan(min)
    // A range that is not a whole number of steps means the control can never be
    // dragged to its own maximum.
    expect((max - min) % step).toBe(0)
  })

  it('starts the control somewhere reachable that is not the answer', () => {
    const { min, max, step, start } = call.variable
    expect(start).toBeGreaterThanOrEqual(min)
    expect(start).toBeLessThanOrEqual(max)
    expect((start - min) % step).toBe(0)
    // The whole product is the gesture. Starting on the optimum removes it —
    // for every player, not just the median one.
    for (const profile of PROFILES) {
      expect(
        judge(start, call.optimal, profile),
        `start is the optimal answer at salary ${profile.salary}`,
      ).not.toBe('optimal')
    }
  })

  it('puts the optimum inside the range the player can actually reach', () => {
    const { min, max, step } = call.variable
    for (const profile of PROFILES) {
      const optimal = resolveOptimal(call.optimal, profile)
      const bounds = typeof optimal === 'number' ? [optimal] : [optimal.min, optimal.max]
      for (const b of bounds) {
        expect(b, `salary ${profile.salary}`).toBeGreaterThanOrEqual(min)
        expect(b, `salary ${profile.salary}`).toBeLessThanOrEqual(max)
        // An optimum off the step grid is unreachable: the player can drag
        // either side of it and is told they were wrong both times.
        expect((b - min) % step, `salary ${profile.salary}`).toBe(0)
      }
      if (typeof optimal !== 'number') {
        expect(optimal.min, `salary ${profile.salary}`).toBeLessThan(optimal.max)
      }
    }
  })

  it('has one crowd weight per position on the control', () => {
    // The histogram indexes crowd[] by step position. Drift here mislabels every
    // bar rather than throwing, so this is the single most load-bearing assertion
    // in the file.
    expect(call.crowd).toHaveLength(stepCount(call.variable))
  })

  it('has crowd weights that can be normalised', () => {
    for (const n of call.crowd) {
      expect(Number.isFinite(n)).toBe(true)
      expect(n).toBeGreaterThanOrEqual(0)
    }
    expect(call.crowd.reduce((a, b) => a + b, 0)).toBeGreaterThan(0)
  })

  it('has a crowd with real structure rather than a placeholder', () => {
    // Real financial behaviour is spiky — defaults, round numbers, whatever a form
    // suggested. A flat or near-flat array is the signature of a distribution that
    // was stubbed and never written, which renders as a plausible-looking wall of
    // equal bars and misleads every player who reads it.
    const peak = Math.max(...call.crowd)
    const mean = call.crowd.reduce((a, b) => a + b, 0) / call.crowd.length
    expect(peak, 'crowd is nearly uniform').toBeGreaterThanOrEqual(mean * 2)
  })
})
