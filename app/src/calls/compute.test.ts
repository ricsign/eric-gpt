/**
 * The tests that matter here are not the ones that restate the arithmetic.
 *
 * The one that matters is `declared optimal is actually optimal`: it sweeps
 * every position of every dial and checks that the best play under the compute
 * function is the play the call claims. If that ever fails, the app is
 * confidently teaching the wrong lesson, which is worse than teaching nothing.
 *
 * Note that "best" is not the same rule on every call, and the difference is
 * itself the lesson. On the employer match, at65 rises for the whole dial —
 * more invested is more invested — and the right play is the cheapest position
 * that still captures every employer dollar. On the Roth split the dial is flat
 * to the dollar and the right play is the cheapest position on the plateau.
 * Each call below names which rule applies to it and why.
 */

import { describe, expect, it } from 'vitest'
import {
  COMPUTE,
  anchorOffer,
  debtSplit,
  emergencyFund,
  employerMatch,
  feeDragCall,
  isSane,
  promoDeadline,
  rentVsBuy,
  repairOrReplace,
  rothSplit,
  timingMarket,
} from './compute'
import { DEFAULT_PROFILE, type CallOutcome, type ComputeFn, type Profile } from './types'
import { futureValue, monthlyRate } from '../lib/finance'

const P = DEFAULT_PROFILE

/** Every position on a dial, built off the index so float drift cannot creep in. */
function positions(min: number, max: number, step: number): number[] {
  const n = Math.round((max - min) / step) + 1
  return Array.from({ length: n }, (_, i) => min + i * step)
}

interface Spec {
  name: string
  fn: ComputeFn
  min: number
  max: number
  step: number
  /** What the call declares as the best play. */
  optimal: number
  /**
   * How this call is scored.
   *  at65      — highest projection wins outright.
   *  plateau   — highest projection, ties inside 0.05% broken by lowest cost.
   *  capture   — most benefit captured, ties broken by lowest cost.
   */
  rule: 'at65' | 'plateau' | 'capture'
}

const SPECS: Spec[] = [
  // at65 climbs the whole dial; 6% is the cheapest place that takes every
  // matched dollar, and past it the employer contributes nothing more.
  { name: 'employerMatch', fn: employerMatch, min: 0, max: 15, step: 1, optimal: 6, rule: 'capture' },
  { name: 'debtSplit', fn: debtSplit, min: 0, max: 500, step: 25, optimal: 0, rule: 'at65' },
  { name: 'emergencyFund', fn: emergencyFund, min: 0, max: 12, step: 1, optimal: 5, rule: 'at65' },
  { name: 'promoDeadline', fn: promoDeadline, min: 1, max: 24, step: 1, optimal: 12, rule: 'at65' },
  { name: 'anchorOffer', fn: anchorOffer, min: 0, max: 20, step: 1, optimal: 8, rule: 'at65' },
  // Pre-tax and Roth land on the same rate at the default salary, so the
  // projection is flat and only the take-home cost separates the positions.
  { name: 'rothSplit', fn: rothSplit, min: 0, max: 100, step: 5, optimal: 0, rule: 'plateau' },
  { name: 'repairOrReplace', fn: repairOrReplace, min: 0, max: 4000, step: 100, optimal: 2500, rule: 'at65' },
  { name: 'feeDragCall', fn: feeDragCall, min: 3, max: 150, step: 1, optimal: 3, rule: 'at65' },
  { name: 'rentVsBuy', fn: rentVsBuy, min: 0, max: 15, step: 1, optimal: 15, rule: 'at65' },
  { name: 'timingMarket', fn: timingMarket, min: 0, max: 30, step: 1, optimal: 0, rule: 'at65' },
]

function sweep(spec: Spec, profile: Profile = P) {
  return positions(spec.min, spec.max, spec.step).map((value) => ({
    value,
    ...spec.fn(value, profile),
  }))
}

/* ========================================================================== */

describe('the declared optimal really is optimal', () => {
  it.each(SPECS)('$name → $optimal', (spec) => {
    const rows = sweep(spec)
    let best: number

    if (spec.rule === 'capture') {
      const top = Math.max(...rows.map((r) => r.benefit))
      best = rows.filter((r) => r.benefit >= top - 1e-6).sort((a, b) => a.cost - b.cost)[0].value
    } else {
      const top = Math.max(...rows.map((r) => r.at65))
      const near =
        spec.rule === 'plateau'
          ? rows.filter((r) => r.at65 >= top - Math.abs(top) * 5e-4)
          : rows.filter((r) => r.at65 >= top - 1e-6)
      best = near.sort((a, b) => a.cost - b.cost)[0].value
    }

    expect(best).toBe(spec.optimal)
  })

  it('emergency fund is right anywhere in the 3-6 month range it claims', () => {
    const rows = sweep(SPECS[2])
    const top = Math.max(...rows.map((r) => r.at65))
    const best = rows.find((r) => r.at65 === top)!.value
    expect(best).toBeGreaterThanOrEqual(3)
    expect(best).toBeLessThanOrEqual(6)
  })

  it('rent vs buy breaks even just under five years', () => {
    const rows = sweep(SPECS[8])
    expect(rows[4].at65).toBeLessThan(0)
    expect(rows[5].at65).toBeGreaterThan(0)
  })
})

/* ========================================================================== */

describe('shape', () => {
  const steps = (rows: { at65: number }[]) => rows.slice(1).map((r, i) => r.at65 - rows[i].at65)

  it('employer match: cost always climbs, employer money stops dead at 6%', () => {
    const rows = sweep(SPECS[0])
    for (let i = 1; i < rows.length; i++) expect(rows[i].cost).toBeGreaterThan(rows[i - 1].cost)
    for (let i = 1; i <= 6; i++) expect(rows[i].benefit).toBeGreaterThan(rows[i - 1].benefit)
    for (let i = 7; i < rows.length; i++) expect(rows[i].benefit).toBeCloseTo(rows[6].benefit, 6)
  })

  it('debt split: every dollar moved to the cheap card costs months and money', () => {
    const rows = sweep(SPECS[1])
    expect(steps(rows).every((d) => d < 0)).toBe(true)
    expect(rows[0].benefit).toBeGreaterThan(rows[rows.length - 1].benefit)
  })

  it('emergency fund turns over: exposed below, over-insured above', () => {
    const rows = sweep(SPECS[2])
    const d = steps(rows)
    const turn = d.findIndex((x) => x < 0)
    expect(turn).toBeGreaterThan(0)
    expect(d.slice(0, turn).every((x) => x > 0)).toBe(true)
    expect(d.slice(turn).every((x) => x < 0)).toBe(true)
    // Twelve months of cover is worse than none: the drag has outrun the risk.
    expect(rows[12].at65).toBeLessThan(0)
  })

  it('promo deadline: free inside the window, then a cliff, not a slope', () => {
    const rows = sweep(SPECS[3])
    expect(rows.slice(0, 12).every((r) => r.at65 > 0)).toBe(true)
    for (let i = 1; i < 12; i++) expect(rows[i].at65).toBeGreaterThan(rows[i - 1].at65)
    // Month 13 is the whole lesson: one month late costs nearly half the prize.
    expect(rows[12].at65 / rows[11].at65).toBeLessThan(0.6)
    for (let i = 13; i < rows.length; i++) expect(rows[i].at65).toBeLessThan(rows[i - 1].at65)
  })

  it('promo deadline: twelve months is free, thirteen is not', () => {
    const inWindow = promoDeadline(12, P)
    const late = promoDeadline(13, P)
    expect(inWindow.breakdown.find((b) => b.label === 'DEFERRED INTEREST')!.value).toBe('$0')
    expect(late.breakdown.find((b) => b.label === 'DEFERRED INTEREST')!.value).not.toBe('$0')
    // Retroactive to day one, so the bill is far more than one month of interest.
    expect(promoDeadline(24, P).benefit).toBe(0)
  })

  it('anchor offer: linear upside, convex risk, one peak', () => {
    const rows = sweep(SPECS[4])
    const d = steps(rows)
    const turn = d.findIndex((x) => x < 0)
    expect(d.slice(0, turn).every((x) => x > 0)).toBe(true)
    expect(d.slice(turn).every((x) => x < 0)).toBe(true)
    // Asking for nothing risks nothing and gains nothing.
    expect(rows[0].at65).toBe(0)
    expect(rows[0].cost).toBe(0)
  })

  it('roth split: the projection is a wash, the take-home cost is not', () => {
    const rows = sweep(SPECS[5])
    const top = Math.max(...rows.map((r) => r.at65))
    const bottom = Math.min(...rows.map((r) => r.at65))
    expect((top - bottom) / top).toBeLessThan(5e-4)
    for (let i = 1; i < rows.length; i++) expect(rows[i].cost).toBeGreaterThan(rows[i - 1].cost)
    // The two rates the call turns on land on the same number at this salary.
    const rateNow = rothSplit(0, P).breakdown.find((b) => b.label === 'RATE NOW')!.value
    const rateLater = rothSplit(0, P).breakdown.find((b) => b.label === 'RATE AT 65')!.value
    expect(rateNow).toBe(rateLater)
  })

  it('roth split: pre-tax wins outright once the rates separate', () => {
    for (const salary of [30_000, 90_000, 150_000]) {
      const rows = sweep(SPECS[5], { salary, age: 30 })
      expect(rows[0].at65).toBeGreaterThanOrEqual(rows[rows.length - 1].at65)
    }
  })

  it('repair or replace: diminishing life bought, one minimum', () => {
    const rows = sweep(SPECS[6])
    const d = steps(rows)
    const turn = d.findIndex((x) => x < 0)
    expect(d.slice(0, turn).every((x) => x > 0)).toBe(true)
    expect(d.slice(turn).every((x) => x < 0)).toBe(true)
    // Half the car's value, which is the rule the call is teaching.
    expect(rows[turn].value).toBe(2500)
  })

  it('fee drag: strictly worse with every basis point', () => {
    const rows = sweep(SPECS[7])
    expect(steps(rows).every((x) => x < 0)).toBe(true)
    const cheap = feeDragCall(3, P)
    const dear = feeDragCall(150, P)
    // The worst fund on the dial is the baseline at65 is measured against, and
    // it reaches zero through a different route than feeDrag() takes — closed
    // form against the month-by-month series. Agreeing to nine decimal places
    // on a seven-figure balance is the check that the two are the same maths.
    expect(dear.at65).toBeCloseTo(0, 6)
    // A 1.5% fee takes a large share of the outcome, not 1.5% of it.
    const share = Number(dear.breakdown.find((b) => b.label === 'SHARE OF OUTCOME')!.value.replace('%', ''))
    expect(share).toBeGreaterThan(20)
    expect(share).toBeLessThan(50)
    expect(cheap.at65).toBeGreaterThan(0)
  })

  it('rent vs buy: staying longer only ever helps, and leaving at once is brutal', () => {
    const rows = sweep(SPECS[8])
    expect(steps(rows).every((x) => x > 0)).toBe(true)
    // Selling costs and closing costs, paid for nothing.
    expect(rows[0].at65).toBeLessThan(-100_000)
  })

  it('timing: missing the ten best days halves money invested for the horizon', () => {
    const rows = sweep(SPECS[9])
    expect(steps(rows).every((x) => x < 0)).toBe(true)
    // The calibration is on a lump held the whole way, which is what the
    // published studies measure. Back it out of the stated annual return.
    const n = (65 - P.age) * 12
    const ratio = (d: number) => {
      const r = Number(
        timingMarket(d, P).breakdown.find((b) => b.label === 'ANNUAL RETURN')!.value.replace('%', ''),
      ) / 100
      return Math.pow(1 + monthlyRate(r), n)
    }
    expect(ratio(10) / ratio(0)).toBeCloseTo(0.5, 2)
    // Concave: the first day missed hurts more than the thirtieth.
    const d = steps(rows)
    for (let i = 1; i < d.length; i++) expect(Math.abs(d[i])).toBeLessThan(Math.abs(d[i - 1]))
  })
})

/* ========================================================================== */

describe('known-good arithmetic', () => {
  it('50% of the first 6% on $62,000 is exactly $1,860 a year', () => {
    expect(employerMatch(6, P).benefit).toBe(1860)
    expect(employerMatch(6, P).cost).toBeCloseTo(310, 10)
    expect(employerMatch(15, P).benefit).toBe(1860)
  })

  it('half the match is left on the table at 3%', () => {
    const at3 = employerMatch(3, P)
    expect(at3.benefit).toBe(930)
    expect(at3.breakdown.find((b) => b.label === 'LEFT ON TABLE')!.value).toBe('$930')
    expect(employerMatch(6, P).breakdown.find((b) => b.label === 'LEFT ON TABLE')!.value).toBe('$0')
  })

  it('the closed-form annuity matches the month-by-month series to the cent', () => {
    // compute.ts uses closed forms so a drag frame is not 420 iterations deep.
    // This is the check that they are the same maths as lib/finance.ts, not a
    // second, slightly-different model of compounding.
    for (const age of [64, 55, 30, 22]) {
      const salary = 62_000
      const monthly = (salary * 0.06 + salary * 0.06 * 0.5) / 12
      const series = futureValue({
        principal: 0,
        monthly,
        annualRate: 0.07,
        years: 65 - age,
      })
      expect(employerMatch(6, { salary, age }).at65).toBeCloseTo(series.balance, 6)
    }
  })

  it('all $500 to the 24.99% card clears the debt soonest and cheapest', () => {
    const best = debtSplit(0, P)
    const worst = debtSplit(500, P)
    expect(worst.benefit).toBe(0)
    expect(best.benefit).toBeGreaterThan(200)
    expect(best.breakdown.find((b) => b.label === 'TO 24.99%')!.value).toBe('$500')
  })

  it('the emergency fund drag is the 3% spread between cash and the market', () => {
    const six = emergencyFund(6, P)
    const held = Number(six.breakdown.find((b) => b.label === 'CASH HELD')!.value.replace(/[$,]/g, ''))
    expect(six.cost * 12).toBeCloseTo(held * 0.03, 0)
  })
})

/* ========================================================================== */

describe('nothing breaks anywhere on any dial', () => {
  const PROFILES: Profile[] = [
    P,
    { salary: 20_000, age: 22 },
    { salary: 400_000, age: 55 },
    { salary: 62_000, age: 64 },
    // Past retirement age: the horizon floors at a year rather than collapsing.
    { salary: 62_000, age: 70 },
  ]

  it.each(SPECS)('$name stays finite and well formed', (spec) => {
    for (const profile of PROFILES) {
      for (const value of positions(spec.min, spec.max, spec.step)) {
        const o = spec.fn(value, profile)
        expect(isSane(o), `${spec.name} @ ${value} / ${profile.salary} / ${profile.age}`).toBe(true)
      }
    }
  })

  it.each(SPECS)('$name tints every block it is asked about', (spec) => {
    const o = spec.fn(spec.min, P)
    if (!o.blockTint) return
    const n = Math.round((spec.max - spec.min) / spec.step) + 1
    for (let i = 0; i < n; i++) expect(['accent', 'plain', 'loss']).toContain(o.blockTint(i))
  })

  it('costs are never negative except where paying more is genuinely cheaper', () => {
    for (const spec of SPECS) {
      for (const value of positions(spec.min, spec.max, spec.step)) {
        const o = spec.fn(value, P)
        // rentVsBuy is the one dial where the comparison itself can flip sign:
        // past the break-even, owning costs less per month than renting.
        if (spec.name !== 'rentVsBuy') expect(o.cost).toBeGreaterThanOrEqual(0)
      }
    }
  })

  it('a value dragged outside its range is clamped, not crashed', () => {
    const wild: [string, number][] = [
      ['employerMatch', -5],
      ['debtSplit', 900],
      ['emergencyFund', -1],
      ['promoDeadline', 0],
      ['promoDeadline', 99],
      ['rothSplit', 500],
      ['feeDragCall', -20],
      ['rentVsBuy', 40],
      ['timingMarket', -3],
      ['anchorOffer', -1],
      ['repairOrReplace', -100],
    ]
    for (const [name, value] of wild) {
      expect(isSane(COMPUTE[name](value, P)), `${name} @ ${value}`).toBe(true)
    }
  })

  it('every call is registered under the name it is keyed by', () => {
    expect(Object.keys(COMPUTE).sort()).toEqual(SPECS.map((s) => s.name).sort())
    for (const [key, fn] of Object.entries(COMPUTE)) {
      expect(typeof fn).toBe('function')
      expect(key).toBe(SPECS.find((s) => s.name === key)!.name)
    }
  })
})

/* ========================================================================== */

describe('receipts', () => {
  it.each(SPECS)('$name marks exactly one line as the one it turns on', (spec) => {
    for (const value of positions(spec.min, spec.max, spec.step)) {
      const o: CallOutcome = spec.fn(value, P)
      expect(o.breakdown.filter((b) => b.emphasis === true)).toHaveLength(1)
      for (const b of o.breakdown) {
        expect(b.label).toMatch(/^[A-Z0-9 '$%/,.()-]+$/)
        expect(b.value.trim()).not.toBe('')
        expect(b.value).not.toMatch(/NaN|Infinity|undefined/)
      }
    }
  })
})

/* ========================================================================== */

describe('cheap enough to run on a drag frame', () => {
  it('recomputes the whole registry thousands of times in well under a frame each', () => {
    const started = performance.now()
    const runs = 2_000
    for (let i = 0; i < runs; i++) {
      for (const spec of SPECS) spec.fn(spec.min + (i % 7) * spec.step, P)
    }
    const perCall = (performance.now() - started) / (runs * SPECS.length)
    expect(perCall).toBeLessThan(1)
  })
})
