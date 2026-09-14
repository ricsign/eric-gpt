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
 * that still captures every employer dollar. Every other call is a plain argmax
 * on at65. Each spec below names which rule applies to it.
 *
 * The second job of this file is the seam with registry.ts. Nothing in the type
 * system ties a compute function to the dial it is dragged on, and `blockTint`
 * is handed a step POSITION — so a call whose control steps in 3s while the
 * tint assumes steps of 1 paints the wrong blocks, forever, silently. That is
 * not a hypothetical: it shipped. `the specs describe the dials the player
 * actually drags` and `tints the dial in the player's own units` catch it.
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
  withholding,
  timingMarket,
} from './compute'
import {
  DEFAULT_PROFILE,
  judge,
  resolveOptimal,
  stepCount,
  type CallOutcome,
  type ComputeFn,
  type Profile,
} from './types'
import { futureValue, monthlyRate, payOffDebt } from '../lib/finance'
import { CALLS } from './registry'

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
   *  capture   — most benefit captured, ties broken by lowest cost.
   */
  rule: 'at65' | 'capture'
  /**
   * What the control must look like at a given position on the dial, written in
   * the units on the readout rather than in block indices. Stating it this way
   * round is the whole point: an index-space expectation would have agreed with
   * an index-space bug.
   */
  tint?: [value: number, expected: 'accent' | 'plain' | 'loss'][]
}

const SPECS: Spec[] = [
  // at65 climbs the whole dial; 6% is the cheapest place that takes every
  // matched dollar, and past it the employer contributes nothing more.
  {
    name: 'employerMatch', fn: employerMatch, min: 0, max: 15, step: 1, optimal: 6, rule: 'capture',
    // The employer's money, and only the employer's money, is lit.
    tint: [[0, 'plain'], [1, 'accent'], [6, 'accent'], [7, 'plain'], [15, 'plain']],
  },
  { name: 'debtSplit', fn: debtSplit, min: 0, max: 500, step: 25, optimal: 0, rule: 'at65' },
  {
    name: 'emergencyFund', fn: emergencyFund, min: 0, max: 12, step: 1, optimal: 5, rule: 'at65',
    tint: [[0, 'loss'], [2, 'loss'], [3, 'accent'], [6, 'accent'], [7, 'plain'], [12, 'plain']],
  },
  {
    name: 'promoDeadline', fn: promoDeadline, min: 1, max: 24, step: 1, optimal: 12, rule: 'at65',
    // The promo window is free and the block after it detonates.
    tint: [[1, 'accent'], [12, 'accent'], [13, 'loss'], [24, 'loss']],
  },
  { name: 'anchorOffer', fn: anchorOffer, min: 0, max: 20, step: 1, optimal: 8, rule: 'at65' },
  // The safe-harbour edge: the last point where no penalty is due and the
  // player is still holding their own money.
  {
    name: 'withholding', fn: withholding, min: 60, max: 180, step: 5, optimal: 90, rule: 'at65',
    tint: [[60, 'loss'], [85, 'loss'], [90, 'accent'], [100, 'accent'], [105, 'plain'], [180, 'plain']],
  },
  { name: 'repairOrReplace', fn: repairOrReplace, min: 0, max: 4000, step: 100, optimal: 2500, rule: 'at65' },
  {
    // The dial steps in 3bp, not 1bp. Getting this wrong here is what let the
    // tint bug through the last time: the sweep tested 148 positions the player
    // cannot reach and never tested the 50 they can.
    name: 'feeDragCall', fn: feeDragCall, min: 3, max: 150, step: 3, optimal: 3, rule: 'at65',
    tint: [[3, 'accent'], [18, 'accent'], [21, 'plain'], [72, 'plain'], [75, 'loss'], [150, 'loss']],
  },
  {
    name: 'rentVsBuy', fn: rentVsBuy, min: 0, max: 15, step: 1, optimal: 15, rule: 'at65',
    tint: [[0, 'loss'], [4, 'loss'], [5, 'accent'], [15, 'accent']],
  },
  {
    name: 'timingMarket', fn: timingMarket, min: 0, max: 30, step: 1, optimal: 0, rule: 'at65',
    tint: [[0, 'accent'], [1, 'loss'], [10, 'loss'], [11, 'plain'], [30, 'plain']],
  },
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
      best = rows.filter((r) => r.at65 >= top - 1e-6).sort((a, b) => a.cost - b.cost)[0].value
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

  it('withholding: a cliff below the safe harbour, a slope above par', () => {
    const rows = sweep(SPECS[5])
    const at = (pct: number) => rows.find((r) => r.value === pct)!
    // Every step up to the safe harbour is an improvement, because each one
    // shrinks a shortfall that is accruing interest at the IRS rate.
    for (const pct of [65, 70, 75, 80, 85, 90]) {
      expect(at(pct).at65, `${pct}%`).toBeGreaterThan(at(pct - 5).at65)
    }
    // Past it, every extra dollar is lent to the Treasury for free.
    for (const pct of [95, 100, 120, 150, 180]) {
      expect(at(pct).at65, `${pct}%`).toBeLessThan(at(90).at65)
    }
    for (let i = rows.findIndex((r) => r.value === 100) + 1; i < rows.length; i++) {
      expect(rows[i].at65).toBeLessThan(rows[i - 1].at65)
    }
  })

  it('withholding: crossing the safe harbour is a step, not a nudge', () => {
    const rows = sweep(SPECS[5])
    const at = (pct: number) => rows.find((r) => r.value === pct)!.at65
    // 85 -> 90 clears a penalty as well as closing a gap; 90 -> 95 only gives
    // money away. The first move has to be worth much more than the second.
    const clearing = at(90) - at(85)
    const giving = at(90) - at(95)
    expect(clearing).toBeGreaterThan(giving * 3)
  })

  it('withholding: the answer is 90% for everyone, because it is a statute', () => {
    // The one call in the set whose optimum cannot move with the player. What
    // moves is the cost of being wrong, which is why the dial is still worth
    // dragging on any salary.
    const peak = (profile: Profile) => {
      const rows = sweep(SPECS[5], profile)
      const top = Math.max(...rows.map((r) => r.at65))
      return rows.find((r) => r.at65 === top)!.value
    }
    for (const profile of [
      { salary: 20_000, age: 22 },
      P,
      { salary: 150_000, age: 30 },
      { salary: 420_000, age: 58 },
    ]) {
      expect(peak(profile), `salary ${profile.salary}`).toBe(90)
    }
    // Being wrong at $150k costs more than being wrong at $20k.
    const missBy = (profile: Profile) => {
      const rows = sweep(SPECS[5], profile)
      const at = (pct: number) => rows.find((r) => r.value === pct)!.at65
      return at(90) - at(180)
    }
    expect(missBy({ salary: 150_000, age: 30 })).toBeGreaterThan(missBy({ salary: 20_000, age: 30 }))
  })

  it('withholding: the tax owed never moves, only who is holding it', () => {
    const owed = (v: number) =>
      withholding(v, P).breakdown.find((b) => b.label === 'TAX OWED')!.value
    // The dial is a W-4, not a tax cut. If this ever stops being true the call
    // is teaching the opposite of its own rule.
    expect(new Set([60, 90, 100, 140, 180].map(owed)).size).toBe(1)
    expect(withholding(90, P).breakdown.some((b) => b.label === 'IRS INTEREST')).toBe(false)
    expect(withholding(60, P).breakdown.some((b) => b.label === 'IRS INTEREST')).toBe(true)
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

  it.each(SPECS)('$name tints the dial in the player\'s own units', (spec) => {
    const o = spec.fn(spec.min, P)
    if (!spec.tint) {
      expect(o.blockTint, `${spec.name} grew a tint with no expectation to check it`).toBeUndefined()
      return
    }
    expect(o.blockTint).toBeTypeOf('function')

    const n = Math.round((spec.max - spec.min) / spec.step) + 1
    for (let i = 0; i < n; i++) expect(['accent', 'plain', 'loss']).toContain(o.blockTint!(i))

    // The assertions are written against the number on the readout, and the
    // position is derived from it here. Asserting in position space is what made
    // the old version of this test agree with a bug that was in position space.
    for (const [value, expected] of spec.tint) {
      const index = Math.round((value - spec.min) / spec.step)
      expect(o.blockTint!(index), `${spec.name} @ ${value}`).toBe(expected)
    }

    // A tint that never uses more than one colour is decoration, not teaching.
    const used = new Set(Array.from({ length: n }, (_, i) => o.blockTint!(i)))
    expect(used.size, `${spec.name} paints the whole dial one colour`).toBeGreaterThan(1)
  })

  it('cost is money leaving the player, so it is never negative', () => {
    for (const spec of SPECS) {
      for (const profile of PROFILES) {
        for (const value of positions(spec.min, spec.max, spec.step)) {
          expect(spec.fn(value, profile).cost, `${spec.name} @ ${value}`).toBeGreaterThanOrEqual(0)
        }
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
      ['withholding', 500],
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
    // The set equality is the assertion; iterating the map to check each key
    // equals itself, which an earlier version did, asserts nothing at all.
    expect(Object.keys(COMPUTE).sort()).toEqual(SPECS.map((s) => s.name).sort())
    for (const fn of Object.values(COMPUTE)) expect(fn).toBeTypeOf('function')
  })
})

/* ========================================================================== */

describe('the specs describe the dials the player actually drags', () => {
  it('every spec matches the control on its call record', () => {
    // `blockTint` is handed a block index, so every compute function encodes the
    // geometry of its own control. Nothing in the type system enforces that, and
    // the sweeps in this file are the only thing standing between a control that
    // steps in 3s and a tint written for steps of 1. When registry.ts moves a
    // dial, this fails first and the tint gets fixed with it.
    for (const spec of SPECS) {
      const call = CALLS.find((c) => c.compute === spec.name)
      expect(call, `no call record uses ${spec.name}`).toBeDefined()
      const v = call!.variable
      expect(
        { min: v.min, max: v.max, step: v.step },
        `${spec.name}: the control moved and compute.ts has not been told`,
      ).toEqual({ min: spec.min, max: spec.max, step: spec.step })
      expect(positions(spec.min, spec.max, spec.step)).toHaveLength(stepCount(v))
    }
  })

  it('the best play under the maths is a play the record calls correct', () => {
    // Age is fixed at 30 by onboarding; salary is a $15k-$400k log slider. That
    // rectangle is every player the app can actually produce, so the optimum has
    // to hold across all of it and not merely at the four profiles that are
    // convenient to type out.
    for (const spec of SPECS) {
      const call = CALLS.find((c) => c.compute === spec.name)!
      for (let salary = 15_000; salary <= 400_000; salary += 5_000) {
        const profile: Profile = { salary, age: DEFAULT_PROFILE.age }
        const rows = sweep(spec, profile)
        const top = Math.max(...rows.map((r) => r.at65))

        if (top - Math.min(...rows.map((r) => r.at65)) <= 1e-6) {
          // A flat dial has no best play to check, so rather than skip it
          // quietly, assert that it is the one call that is allowed to go flat
          // and only where it is allowed to. See the standard-deduction test.
          expect(`${spec.name} below $${salary}`).toBe(`withholding below $${salary}`)
          expect(salary).toBeLessThan(17_000)
          continue
        }

        const best = rows.filter((r) => r.at65 >= top - 1e-6).sort((a, b) => a.cost - b.cost)[0]
        expect(
          judge(best.value, call.optimal, profile),
          `${spec.name} at $${salary}: the maths peaks at ${best.value}, the record claims ` +
            JSON.stringify(resolveOptimal(call.optimal, profile)),
        ).toBe('optimal')
      }
    }
  })
})

/* ========================================================================== */

describe('the closed forms agree with the slow maths they replaced', () => {
  it('debt: the closed-form amortisation matches payOffDebt to the dollar', () => {
    // monthsToClear/balanceAfter are the continuous version of the iterative
    // schedule in lib/finance.ts, which is separately tested. If they ever drift
    // apart, one of the two is wrong and the drag is the one nobody checks.
    for (const [balance, apr, payment] of [
      [4_200, 0.2499, 500],
      [2_800, 0.1199, 500],
      [4_200, 0.2499, 125],
    ] as const) {
      const iterative = payOffDebt({ balance, apr, monthlyPayment: payment })
      // All $500 at one card is the single-card case, so debtSplit's first leg
      // has to reproduce it: total paid is payment x months, whole or partial.
      const months = -Math.log(1 - ((apr / 12) * balance) / payment) / Math.log(1 + apr / 12)
      expect(Math.ceil(months)).toBe(iterative.months)

      // The two can never agree exactly and should not be forced to: the
      // schedule charges a whole final month of interest, the closed form stops
      // part-way through it. The whole gap must fit inside that last month's
      // interest, which is the tightest bound that is actually true.
      const continuous = payment * months - balance
      const lastMonth = payment * (apr / 12)
      expect(continuous).toBeLessThan(iterative.totalInterest)
      expect(iterative.totalInterest - continuous).toBeLessThan(lastMonth)
    }
  })

  it('debt: sending everything to the 24.99% card reproduces its own schedule', () => {
    const interest = Number(
      debtSplit(0, P).breakdown.find((b) => b.label === 'INTEREST PAID')!.value.replace(/[$,]/g, ''),
    )
    // Phase one: $500 clears the expensive card while the cheap one accrues
    // untouched. Phase two: the whole $500 falls on what is left of the cheap
    // one. Walked month by month here, closed-form there.
    const iH = 0.2499 / 12
    const iL = 0.1199 / 12
    const nH = -Math.log(1 - (iH * 4_200) / 500) / Math.log(1 + iH)
    const grown = 2_800 * Math.pow(1 + iL, nH)
    const nL = -Math.log(1 - (iL * grown) / 500) / Math.log(1 + iL)
    const expected = (500 * nH - 4_200) + (grown - 2_800) + (500 * nL - grown)
    expect(interest).toBeCloseTo(expected, 0)
  })

  it('promo: the retroactive bill matches a month-by-month deferred-interest run', () => {
    // The clause is the entire call. If the closed form over-or-understates it,
    // the cliff moves and the lesson moves with it.
    for (const m of [13, 15, 18, 24]) {
      const pay = 3_000 / m
      const i = 0.2699 / 12
      let balance = 3_000
      let accrued = 0
      for (let k = 0; k < 12; k++) {
        accrued += balance * i
        balance -= pay
      }
      let carried = balance + accrued
      let after = 0
      for (let k = 0; k < 600 && carried > 0.005; k++) {
        const int = carried * i
        after += int
        carried = carried + int - Math.min(pay, carried + int)
      }
      const shown = Number(
        promoDeadline(m, P)
          .breakdown.find((b) => b.label === 'DEFERRED INTEREST')!
          .value.replace(/[$,]/g, ''),
      )
      expect(shown, `${m} months`).toBeCloseTo(accrued + after, -1)
    }
  })

  it('housing: the two geometric sums match a month-by-month simulation', () => {
    // rentVsBuy is the most intricate closed form in the file — a mortgage
    // balance, two geometric series and a mid-year settlement convention. This
    // rebuilds it the dumb way and requires agreement inside 0.2%.
    const HOME = 360_000
    const i = 0.065 / 12
    const g = Math.pow(1 + i, 360)
    const payment = (HOME * 0.8 * i * g) / (g - 1)
    const rm = monthlyRate(0.07)

    for (const years of [1, 3, 5, 10, 15]) {
      let owed = HOME * 0.8
      let invested = HOME * 0.23
      let rent = 2_160
      let running = (0.015 * HOME) / 12
      for (let m = 1; m <= years * 12; m++) {
        owed = owed + owed * i - payment
        invested = invested * (1 + rm) + (payment + running - rent)
        if (m % 12 === 0) {
          rent *= 1.03
          running *= 1.03
        }
      }
      const simulated = 0.94 * HOME * Math.pow(1.03, years) - owed - invested
      // The minus sign is kept: whether buying is ahead is the answer, not a
      // formatting detail, so the simulation has to agree on the sign too.
      const shown = Number(
        rentVsBuy(years, P)
          .breakdown.find((b) => b.label === 'BUY MINUS RENT')!
          .value.replace(/[$,]/g, ''),
      )
      expect(Math.sign(shown), `${years} years`).toBe(Math.sign(simulated))
      // Relative, because the figure runs from four to six digits across the
      // dial and a fixed dollar bound would be slack at one end and impossible
      // at the other. The residual is the mid-year settlement convention.
      expect(Math.abs(simulated / shown - 1), `${years} years`).toBeLessThan(0.002)
    }
  })
})

/* ========================================================================== */

describe('the dial always has something in it', () => {
  const spread = (spec: Spec, profile: Profile) => {
    const rows = sweep(spec, profile).map((r) => r.at65)
    return Math.max(...rows) - Math.min(...rows)
  }

  it.each(SPECS)('$name moves across the whole salary track', (spec) => {
    // A call whose projection does not move as you drag is an article wearing a
    // control. Checked at both ends of the slider, not only in the middle.
    for (const salary of [20_000, 62_000, 150_000, 400_000]) {
      expect(spread(spec, { salary, age: DEFAULT_PROFILE.age }), `$${salary}`).toBeGreaterThan(1)
    }
  })

  it('withholding goes dead below the standard deduction, and that is not fixable here', () => {
    // A single filer under the 2026 standard deduction owes no federal income
    // tax, so there is genuinely no W-4 decision to make and every position on
    // the dial is identical. The salary slider starts at $15,000, which is under
    // that line — so the bottom $1,100 of the track produces a call with nothing
    // in it. Inventing a liability to fill it would be a fabricated number on a
    // receipt, which is worse; the fix is a salary floor or a fixed scenario pay
    // on the card, and neither of those lives in this file. Pinned here so it is
    // visible rather than discovered by a player.
    const dead = { salary: 15_000, age: 30 }
    expect(spread(SPECS[5], dead)).toBe(0)
    expect(withholding(90, dead).breakdown.find((b) => b.label === 'TAX OWED')!.value).toBe('$0')
    // The first salary that owes tax brings the dial back to life.
    expect(spread(SPECS[5], { salary: 17_000, age: 30 })).toBeGreaterThan(0)
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
