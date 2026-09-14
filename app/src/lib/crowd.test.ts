import { describe, expect, it } from 'vitest'
import { SEED_WEIGHT, blendCrowd, modeOf, percentileOf } from './crowd'
import { tabSummary } from '../screens/Tab'
import { CALLS } from '../calls/registry'
import {
  DEFAULT_PROFILE,
  resolveOptimal,
  stepCount,
  type CallResult,
  type Verdict,
} from '../calls/types'

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)

describe('blendCrowd', () => {
  it('returns nothing to draw when there is nothing to draw', () => {
    expect(blendCrowd([], [])).toEqual([])
    // A real call with no answers yet: zeros, not NaN. A NaN here would render
    // as a chart of invisible bars rather than an empty one.
    expect(blendCrowd([0, 0, 0], [0, 0, 0])).toEqual([0, 0, 0])
  })

  it('normalises to a 100-unit crowd', () => {
    expect(sum(blendCrowd([1, 2, 1], [0, 0, 0]))).toBeCloseTo(100, 10)
    expect(sum(blendCrowd([1, 2, 1], [5, 0, 0]))).toBeCloseTo(100, 10)
    expect(sum(blendCrowd([], [3, 1]))).toBeCloseTo(100, 10)
  })

  it('reads the seed as a shape, not a scale', () => {
    // Seeds are authored as relative weights; doubling every entry is the same
    // distribution and must not change how much the seed is worth.
    expect(blendCrowd([1, 3], [10, 0])).toEqual(blendCrowd([100, 300], [10, 0]))
  })

  it('decays: the seed holds SEED_WEIGHT/(SEED_WEIGHT+n) of the chart', () => {
    const uniform = [1, 1]

    // No observations: the seed is the whole picture.
    expect(blendCrowd(uniform, [0, 0])).toEqual([50, 50])

    // Exactly SEED_WEIGHT observations, all in bucket 0: the seed is worth half.
    // 60 + 120 against 60 => 75/25.
    const half = blendCrowd(uniform, [SEED_WEIGHT, 0])
    expect(half[0]).toBeCloseTo(75, 10)
    expect(half[1]).toBeCloseTo(25, 10)

    // Ten times the seed's weight: it is down to a ninth of bucket 1.
    const late = blendCrowd(uniform, [SEED_WEIGHT * 10, 0])
    expect(late[1]).toBeCloseTo((60 / (60 + 60 + 1200)) * 100, 10)

    // Monotone: more observations in a bucket never lowers that bucket's share.
    let previous = 0
    for (const n of [0, 1, 10, 100, 1000, 10_000]) {
      const share = blendCrowd(uniform, [n, 0])[0]
      expect(share).toBeGreaterThanOrEqual(previous)
      previous = share
    }
    expect(previous).toBeGreaterThan(99)
  })

  it('falls back to observations when a call has no seed', () => {
    expect(blendCrowd([0, 0], [1, 3])).toEqual([25, 75])
  })

  it('survives a stale local histogram from an older version of the call', () => {
    // The seed defines the variable, so its length wins: extra buckets are
    // dropped and missing ones read as zero rather than shifting the chart.
    expect(blendCrowd([1, 1], [0, 0, 999])).toEqual([50, 50])
    expect(blendCrowd([1, 1], [])).toEqual([50, 50])
  })

  it('ignores impossible weights instead of propagating them', () => {
    const blended = blendCrowd([1, 1], [Number.NaN, -5])
    expect(blended).toEqual([50, 50])
    expect(blended.every(Number.isFinite)).toBe(true)
  })

  it('never invents or loses a bucket', () => {
    for (const call of CALLS) {
      const steps = stepCount(call.variable)
      const blended = blendCrowd(call.crowd, new Array(steps).fill(0))
      expect(blended).toHaveLength(steps)
      expect(blended.every(Number.isFinite)).toBe(true)
      expect(sum(blended)).toBeCloseTo(100, 8)
    }
  })
})

describe('percentileOf', () => {
  it('is the no-information answer when there is no crowd', () => {
    expect(percentileOf(3, [], 0, 1, 6)).toBe(50)
    expect(percentileOf(3, [0, 0, 0, 0], 0, 1, 6)).toBe(50)
  })

  it('splits ties down the middle', () => {
    // Everyone, including the player, answered exactly as well as each other.
    expect(percentileOf(6, [0, 0, 0, 0, 0, 0, 100], 0, 1, 6)).toBeCloseTo(50, 6)

    // Half the crowd tied with the player, half did worse: 50 worse plus half of
    // the 51-strong tie group (50 of them plus the player), over 101.
    const d = [0, 50, 0, 50]
    expect(percentileOf(1, d, 0, 1, 1)).toBeCloseTo(((50 + 51 / 2) / 101) * 100, 6)
  })

  it('scores distance from optimal, not direction', () => {
    // A distribution that is symmetric about the optimum must score a player who
    // fell two short exactly as one who went two over. Anything else is scoring
    // a value as "safer", which is the instinct the whole product is arguing with.
    const d = [10, 20, 30, 20, 10]
    expect(percentileOf(0, d, 0, 1, 2)).toBeCloseTo(percentileOf(4, d, 0, 1, 2), 10)
    expect(percentileOf(1, d, 0, 1, 2)).toBeGreaterThan(percentileOf(0, d, 0, 1, 2))
  })

  it('handles an optimum that sits between buckets', () => {
    // A range collapsed to its midpoint (3-6 months -> 4.5) leaves buckets 4 and
    // 5 equally good; they must score identically.
    const d = [20, 20, 20, 10, 10, 10, 10]
    expect(percentileOf(4, d, 0, 1, 4.5)).toBeCloseTo(percentileOf(5, d, 0, 1, 4.5), 10)
  })

  it('scores every answer inside a band as equally right', () => {
    // Three months of expenses and six months of expenses are both correct. Given
    // the band rather than its midpoint, the whole band ties — scoring 3 below
    // 4.5 would tell someone who answered correctly that they were nearly wrong.
    const d = [10, 10, 10, 10, 10, 10, 10]
    const band = { min: 3, max: 6 }
    const inside = [3, 4, 5, 6].map((v) => percentileOf(v, d, 0, 1, band))
    for (const p of inside) expect(p).toBeCloseTo(inside[0], 10)

    // And an answer outside it is measured from the nearest edge, not the middle:
    // one short of three months and one over six months are the same near-miss.
    expect(percentileOf(2, d, 0, 1, band)).toBeLessThan(inside[0])
    expect(percentileOf(2, d, 0, 1, band)).toBeCloseTo(percentileOf(7, d, 0, 1, band), 10)
    expect(percentileOf(1, d, 0, 1, band)).toBeLessThan(percentileOf(2, d, 0, 1, band))
  })

  it('handles a band with no upper bound', () => {
    // Contributing more than the match cap is never a mistake, so everything from
    // the cap upward ties, and only falling short is scored as a miss.
    const d = [10, 10, 10, 10]
    const open = { min: 2, max: Number.POSITIVE_INFINITY }
    expect(percentileOf(2, d, 0, 1, open)).toBeCloseTo(percentileOf(3, d, 0, 1, open), 10)
    expect(percentileOf(1, d, 0, 1, open)).toBeLessThan(percentileOf(2, d, 0, 1, open))
    expect(percentileOf(0, d, 0, 1, open)).toBeGreaterThanOrEqual(0)
  })

  it('never lets anyone beat themselves', () => {
    // Uniquely nearest the optimum, with every other bucket strictly worse: still
    // not 100, because the player is in the crowd being measured.
    const best = percentileOf(0, [0, 10, 10, 10], 0, 1, 0)
    expect(best).toBeLessThan(100)
    expect(Number(best.toFixed(0))).toBeLessThan(100)

    // And the worst possible answer is not better than nobody.
    const worst = percentileOf(3, [10, 10, 10, 0], 0, 1, 0)
    expect(worst).toBeGreaterThan(0)
    expect(Number(worst.toFixed(0))).toBeGreaterThan(0)
  })

  it('stays inside [0,100] for every answer to every shipped call', () => {
    for (const call of CALLS) {
      const { min, step } = call.variable
      const steps = stepCount(call.variable)
      const crowd = blendCrowd(call.crowd, new Array(steps).fill(0))
      const best = resolveOptimal(call.optimal, DEFAULT_PROFILE)

      for (let i = 0; i < steps; i++) {
        const p = percentileOf(min + i * step, crowd, min, step, best)
        expect(p).toBeGreaterThanOrEqual(0)
        expect(p).toBeLessThanOrEqual(100)
        expect(Number.isFinite(p)).toBe(true)
      }
    }
  })

  it('stays inside [0,100] for values off the ends of the chart', () => {
    const d = [10, 10, 10]
    for (const v of [-99, 99, Number.NaN, Number.POSITIVE_INFINITY]) {
      const p = percentileOf(v, d, 0, 1, 1)
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(100)
    }
  })

  it('rewards the better call on a real distribution', () => {
    // Call 1: the crowd piles up on the 3% auto-enrolment default, the match caps
    // at 6%. Answering 6 must beat answering 3 even though 3 is the popular call
    // — that gap is the entire point of showing the crowd.
    const call = CALLS[0]
    const crowd = blendCrowd(call.crowd, new Array(stepCount(call.variable)).fill(0))
    const atDefault = percentileOf(3, crowd, 0, 1, 6)
    const atMatch = percentileOf(6, crowd, 0, 1, 6)
    expect(atMatch).toBeGreaterThan(atDefault)
  })
})

describe('modeOf', () => {
  it('has no mode for an empty or empty-handed distribution', () => {
    expect(modeOf([], 0, 1)).toBeNaN()
    expect(modeOf([0, 0, 0], 0, 1)).toBeNaN()
    expect(modeOf([Number.NaN, -1], 0, 1)).toBeNaN()
  })

  it('maps the heaviest bucket back to its value', () => {
    expect(modeOf([1], 5, 25)).toBe(5)
    expect(modeOf([1, 9, 2], 0, 1)).toBe(1)
    expect(modeOf([1, 2, 40, 2], 0, 25)).toBe(50)
    expect(modeOf([1, 2, 3], 3, 0.5)).toBe(4)
  })

  it('breaks ties toward the lower value', () => {
    // Stability matters more than fairness here: a mode that flips between two
    // equal peaks as observations arrive makes the chart look broken.
    expect(modeOf([5, 5, 5], 0, 1)).toBe(0)
    expect(modeOf([1, 7, 3, 7], 0, 1)).toBe(1)
  })

  it('finds the mode at either end', () => {
    expect(modeOf([9, 1, 1], 0, 1)).toBe(0)
    expect(modeOf([1, 1, 9], 0, 1)).toBe(2)
  })
})

/* ---- The Tab ---------------------------------------------------------------- */

let nextId = 1

function result(patch: Partial<CallResult> & { verdict: Verdict }): CallResult {
  return {
    callId: nextId++,
    value: 0,
    delta: 0,
    at65: 0,
    day: '2026-09-01',
    practice: false,
    ...patch,
  }
}

describe('tabSummary', () => {
  it('is zero, not NaN, before the first call', () => {
    expect(tabSummary([])).toEqual({
      total: 0,
      played: 0,
      optimal: 0,
      ratio: 0,
      streak: 0,
      missed: 0,
      capture: 0,
    })
  })

  it('only moves on a good decision', () => {
    const s = tabSummary([
      result({ verdict: 'optimal', at65: 400_000, delta: 0, day: '2026-09-01' }),
      result({ verdict: 'short', at65: 210_000, delta: -190_000, day: '2026-09-02' }),
    ])
    // The miss contributes nothing to the total and does not claw anything back.
    expect(s.total).toBe(400_000)
    expect(s.optimal).toBe(1)
    expect(s.played).toBe(2)
    expect(s.ratio).toBe(0.5)
    expect(s.missed).toBe(190_000)
  })

  it('never falls as more calls are played', () => {
    const history: CallResult[] = []
    let previous = 0
    for (const verdict of ['short', 'optimal', 'over', 'short', 'optimal'] as Verdict[]) {
      history.push(result({ verdict, at65: 100_000, delta: verdict === 'optimal' ? 0 : -40_000 }))
      const total = tabSummary(history).total
      expect(total).toBeGreaterThanOrEqual(previous)
      previous = total
    }
  })

  it('counts an overshoot as a miss without counting it as a cost', () => {
    // Over-contributing can project to more money at 65 and still be the wrong
    // call. It must not inflate "left on the table", and it must not read as
    // having captured more than the best play was worth.
    const s = tabSummary([result({ verdict: 'over', at65: 500_000, delta: 60_000 })])
    expect(s.missed).toBe(0)
    expect(s.optimal).toBe(0)
    expect(s.capture).toBe(1)
  })

  it('captures the share of the best plays actually taken', () => {
    const s = tabSummary([
      result({ verdict: 'optimal', at65: 100_000, delta: 0 }),
      result({ verdict: 'short', at65: 50_000, delta: -50_000 }),
    ])
    // Best plays were worth 100k + 100k; the player's calls were worth 150k.
    expect(s.capture).toBeCloseTo(0.75, 10)
  })

  describe('streak', () => {
    const on = (day: string, verdict: Verdict = 'optimal') => result({ verdict, day })

    it('counts consecutive days back from the most recent', () => {
      expect(tabSummary([on('2026-09-01')]).streak).toBe(1)
      expect(tabSummary([on('2026-09-01'), on('2026-09-02'), on('2026-09-03')]).streak).toBe(3)
    })

    it('does not care whether the calls were good', () => {
      // A streak is showing up, not being right. Scoring it would punish the days
      // the product is most useful.
      expect(tabSummary([on('2026-09-01', 'short'), on('2026-09-02', 'over')]).streak).toBe(2)
    })

    it('ends at the first missed day', () => {
      const s = tabSummary([on('2026-09-01'), on('2026-09-02'), on('2026-09-05'), on('2026-09-06')])
      expect(s.streak).toBe(2)
      expect(s.played).toBe(4)
    })

    it('counts a day once, however many results it holds', () => {
      // Two results on one day is not two days. It should not be possible with a
      // call closing on answer, but a deep-linked replay is one bug away from it.
      expect(tabSummary([on('2026-09-01'), on('2026-09-01'), on('2026-09-02')]).streak).toBe(2)
    })

    it('does not depend on the order results arrive in', () => {
      const days = ['2026-09-03', '2026-09-01', '2026-09-02']
      expect(tabSummary(days.map((d) => on(d))).streak).toBe(3)
    })

    it('crosses a month and a year boundary', () => {
      expect(tabSummary([on('2026-09-30'), on('2026-10-01')]).streak).toBe(2)
      expect(tabSummary([on('2026-12-31'), on('2027-01-01')]).streak).toBe(2)
    })
  })
})
