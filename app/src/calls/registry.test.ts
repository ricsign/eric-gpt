import { describe, expect, it } from 'vitest'
import { COMPUTE, SCENARIO } from './compute'
import { CALLS, callById, callCount } from './registry'
import { DEFAULT_PROFILE, judge, resolveOptimal, stepCount, type Call, type Profile } from './types'

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
 * A rate in the fine print that the maths stopped using does not throw at all: it
 * ships, on the receipt, forever. Everything here exists because it would
 * otherwise look fine.
 */

/** Every reachable position on a control. */
function sweep(v: Call['variable']): number[] {
  const out: number[] = []
  for (let x = v.min; x <= v.max + 1e-9; x += v.step) out.push(Math.round(x / v.step) * v.step)
  return out
}

const at65 = (call: Call, value: number, profile: Profile) =>
  COMPUTE[call.compute](value, profile).at65

/**
 * Imperative openers. Deliberately wider than the verbs currently in use so a
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

/** Every figure written into a string: "$4,200 AT 24.99%" -> [4200, 24.99]. */
const numbers = (text: string): number[] =>
  (text.match(/\d[\d,]*(?:\.\d+)?/g) ?? []).map((n) => Number(n.replace(/,/g, '')))

/** Copy rounds to the dollar and to two places; the model does not. */
const has = (pool: number[], n: number) =>
  pool.some((p) => Math.abs(p - n) <= Math.max(0.51, Math.abs(p) * 0.001))

/**
 * Figures every call may state because the whole app is built on them: the 7%
 * nominal return, the 4% cash yield, retirement at 65, twelve months in a year,
 * and a plain zero.
 */
const SHARED_FIGURES = [7, 4, 65, 12, 0]

/**
 * Per-call figures that compute.ts models but does not list in SCENARIO,
 * because SCENARIO only covers what the fact tiles say. Each one is a named
 * constant in compute.ts, and each is here because the fine print names it.
 */
const EXTRA_FIGURES: Record<string, number[]> = {
  // CARD_APR — the card an uncovered shock ends up on.
  emergencyFund: [24.99],
  // MORT_MONTHS/12, OWN_COST, APPRECIATION (which is also the rent's growth).
  rentVsBuy: [30, 1.5, 3],
}

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
    // Two calls sharing a function would give the same day twice with different
    // copy. (That the name resolves at all is integration.test.ts's job.)
    const used = CALLS.map((c) => c.compute)
    expect(new Set(used).size).toBe(used.length)
  })

  it('never repeats a title or a rule', () => {
    // Rules are filed to the player's Rules screen forever; a duplicate reads as
    // a bug in the app rather than as the same advice twice.
    expect(new Set(CALLS.map((c) => c.title)).size).toBe(CALLS.length)
    expect(new Set(CALLS.map((c) => c.rule)).size).toBe(CALLS.length)
  })

  it('chains tomorrow-teasers into a loop, each one quoting the next title', () => {
    // The teaser is the last thing on the receipt, so a stale one advertises a call
    // that no longer exists. Tying it to the next title makes that impossible.
    CALLS.forEach((c, i) => {
      const next = CALLS[(i + 1) % CALLS.length]
      expect(c.tomorrow, `call ${c.id} teaser`).toBe(next.title)
    })
  })

  it('writes every player-facing string in the alphabet the product uses', () => {
    // No emoji, anywhere, ever — and no smart quotes or stray glyphs either,
    // because every one of these strings is also rendered into the receipt's
    // monospace column and into a share sheet.
    const allowed = /^[ -~·—]*$/
    for (const c of CALLS) {
      const strings = [
        c.title,
        c.rule,
        c.tomorrow,
        c.assumptions,
        c.variable.label,
        c.variable.unit,
        ...c.fixed.flatMap((f) => [f.k, f.v]),
      ]
      for (const s of strings) {
        expect(allowed.test(s), `call ${c.id}: "${s}"`).toBe(true)
      }
    }
  })
})

describe.each(CALLS.map((c) => [c.id, c] as const))('call %i', (_id, call) => {
  it('has a title that fits three lines and is not a question', () => {
    // Twelve words is the contract's ceiling. 50 characters is the empirical
    // one: it is the longest title measured at 402px that still wrapped to
    // three lines of Archivo Black, and every title over it wrapped to four.
    // Neither is sufficient — the same 44 characters wrap to three lines or to
    // four depending on word order, so only a browser can settle it. These
    // catch the gross regression; scripts/shoot.mjs catches the rest.
    expect(words(call.title).length).toBeLessThanOrEqual(12)
    expect(call.title.length).toBeLessThanOrEqual(50)
    expect(call.title.endsWith('?')).toBe(false)
    expect(call.title.trim()).toBe(call.title)
  })

  it('has a rule that is short and gives an instruction', () => {
    expect(words(call.rule).length).toBeLessThan(10)
    const first = words(call.rule)[0].toLowerCase().replace(/[^a-z-]/g, '')
    expect(IMPERATIVE_VERBS.has(first), `rule opens on "${first}"`).toBe(true)
  })

  it('has two fact tiles that fit the tile they are set in', () => {
    // 22px Courier Prime in a half-width tile is eleven characters to the line,
    // and two lines is what makes all ten call screens the same height, so the
    // control sits under the same thumb every day. A third line eats into the
    // 194px of slack the column has at 402x874, and .call clips, never scrolls.
    expect(call.fixed).toHaveLength(2)
    for (const f of call.fixed) {
      expect(f.k.length).toBeGreaterThan(0)
      expect(f.k).toBe(f.k.toUpperCase())
      // The key is one nowrap line, ellipsised past ~22 characters.
      expect(f.k.length, `tile key "${f.k}"`).toBeLessThanOrEqual(18)
      expect(f.v.length).toBeGreaterThan(0)
      expect(f.v.length, `tile value "${f.v}"`).toBeLessThanOrEqual(23)
    }
  })

  it('keeps the fine print to fine print', () => {
    // 11px on the outcome screen, under the receipt button. Past about 200
    // characters it stops being a footnote and starts being the paragraph the
    // product does not have.
    const a = call.assumptions
    expect(a.trim()).toBe(a)
    expect(a.length).toBeGreaterThan(0)
    expect(a.length, 'assumptions have become a paragraph').toBeLessThanOrEqual(200)
    expect(call.variable.label, 'the readout caption is authored lowercase').toBe(
      call.variable.label.toLowerCase(),
    )
  })

  it('states no figure the maths does not actually use', () => {
    // The seam nothing else covers. A card row that contradicts the model is
    // caught by integration.test.ts against SCENARIO; the title, the readout
    // caption and the assumptions are prose, and prose is where a scenario rots
    // silently — an earlier draft of this file told the player the sofa card
    // charged 29.99% while the model charged 26.99, and priced a $441 car
    // payment at $520. Both shipped through every other test in the suite.
    const pool = [
      ...(SCENARIO[call.compute] ?? []),
      ...(EXTRA_FIGURES[call.compute] ?? []),
      ...SHARED_FIGURES,
      call.variable.min,
      call.variable.max,
      call.variable.step,
    ]
    const prose = [call.title, call.variable.label, call.assumptions, ...call.fixed.map((f) => f.v)]

    for (const text of prose) {
      for (const n of numbers(text)) {
        expect(has(pool, n), `call ${call.id} says ${n} in "${text}"`).toBe(true)
      }
    }
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

  it('never calls a play optimal that is worse than where the player started', () => {
    // The bug this exists for: call 7's band once ran from $0, so walking away
    // from a repairable car to start a $441 payment — the single worst position
    // on that dial — was stamped "Optimal play". integration.test.ts could not
    // see it, because it only checks that the *peak* is inside the band.
    for (const profile of PROFILES) {
      const fromStart = at65(call, call.variable.start, profile)
      for (const v of sweep(call.variable)) {
        if (judge(v, call.optimal, profile) !== 'optimal') continue
        expect(
          at65(call, v, profile),
          `call ${call.id}: ${v} is called optimal at salary ${profile.salary} but is worse ` +
            `than the start position ${call.variable.start}`,
        ).toBeGreaterThan(fromStart)
      }
    }
  })

  it('never calls a play optimal that is worse than a play it calls wrong', () => {
    for (const profile of PROFILES) {
      const inside: number[] = []
      const outside: number[] = []
      for (const v of sweep(call.variable)) {
        ;(judge(v, call.optimal, profile) === 'optimal' ? inside : outside).push(
          at65(call, v, profile),
        )
      }
      if (!outside.length) continue
      expect(
        Math.min(...inside),
        `call ${call.id} at salary ${profile.salary}: the worst optimal play loses to the ` +
          'worst wrong one',
      ).toBeGreaterThan(Math.min(...outside))
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
