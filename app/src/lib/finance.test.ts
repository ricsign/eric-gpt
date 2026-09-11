import { describe, expect, it } from 'vitest'
import {
  aprToApy,
  coastNumber,
  costOfWaiting,
  doublingTime,
  employerMatch,
  feeDrag,
  fireNumber,
  futureValue,
  growthSeries,
  monthlyRate,
  payOffDebt,
  payoffVsInvest,
  recurringCostAsPortfolio,
  ruleOf72,
  yearsToTarget,
} from './finance'

/** Closeness helper for money: within a cent is exact enough for a projection. */
const near = (a: number, b: number, tol = 0.01) => expect(Math.abs(a - b)).toBeLessThan(tol)

describe('rate conversions', () => {
  it('monthlyRate compounds back to the annual rate', () => {
    const r = monthlyRate(0.07)
    near(Math.pow(1 + r, 12) - 1, 0.07, 1e-12)
  })

  it('monthlyRate is below the naive apr/12 shortcut', () => {
    // The shortcut overstates the monthly rate, which is why it overstates growth.
    expect(monthlyRate(0.07)).toBeLessThan(0.07 / 12)
  })

  it('aprToApy matches the textbook monthly case', () => {
    // 12% APR compounded monthly is the canonical 12.6825% APY.
    near(aprToApy(0.12, 12), 0.126825, 1e-6)
  })

  it('aprToApy handles continuous compounding', () => {
    near(aprToApy(0.12, Infinity), Math.exp(0.12) - 1, 1e-12)
  })

  it('a zero rate yields a zero APY', () => {
    expect(aprToApy(0)).toBe(0)
  })
})

describe('growthSeries', () => {
  it('with no return, the balance is exactly what was put in', () => {
    const end = futureValue({ principal: 1000, monthly: 100, annualRate: 0, years: 10 })
    near(end.balance, 1000 + 100 * 120)
    near(end.growth, 0)
  })

  it('a lump sum with no contributions compounds to the closed-form value', () => {
    const end = futureValue({ principal: 10_000, monthly: 0, annualRate: 0.07, years: 30 })
    near(end.balance, 10_000 * Math.pow(1.07, 30), 0.5)
  })

  it('contributions match the ordinary-annuity formula', () => {
    // FV of an end-of-period annuity: PMT * ((1+i)^n - 1) / i
    const i = monthlyRate(0.06)
    const n = 12 * 20
    const expected = 500 * ((Math.pow(1 + i, n) - 1) / i)
    const end = futureValue({ principal: 0, monthly: 500, annualRate: 0.06, years: 20 })
    near(end.balance, expected, 0.5)
  })

  it('emits one point per month plus the starting point', () => {
    expect(growthSeries({ principal: 0, monthly: 1, annualRate: 0.05, years: 3 })).toHaveLength(37)
  })

  it('balance is always contributed plus growth', () => {
    for (const p of growthSeries({ principal: 500, monthly: 250, annualRate: 0.08, years: 12 })) {
      near(p.balance, p.contributed + p.growth, 1e-6)
    }
  })

  it('inflation reduces the real balance but never the nominal one', () => {
    const end = futureValue({
      principal: 10_000,
      monthly: 0,
      annualRate: 0.07,
      years: 30,
      inflation: 0.03,
    })
    expect(end.realBalance).toBeLessThan(end.balance)
    // 7% nominal against 3% inflation is a ~3.88% real rate.
    near(end.realBalance, 10_000 * Math.pow(1.07 / 1.03, 30), 1)
  })

  it('contribution growth raises the amount added each year', () => {
    const flat = futureValue({ principal: 0, monthly: 100, annualRate: 0.07, years: 10 })
    const rising = futureValue({
      principal: 0,
      monthly: 100,
      annualRate: 0.07,
      years: 10,
      contributionGrowth: 0.03,
    })
    expect(rising.contributed).toBeGreaterThan(flat.contributed)
    expect(rising.balance).toBeGreaterThan(flat.balance)
  })

  it('a zero-year horizon returns only the principal', () => {
    const end = futureValue({ principal: 250, monthly: 999, annualRate: 0.9, years: 0 })
    near(end.balance, 250)
  })
})

describe('costOfWaiting', () => {
  const base = { principal: 0, monthly: 500, annualRate: 0.07, years: 40 }

  it('forfeited growth dwarfs the contributions skipped', () => {
    // This is the whole thesis of the app; if it ever stops holding, the maths broke.
    const r = costOfWaiting(base, 10)
    expect(r.growthForfeited).toBeGreaterThan(r.contributionsSkipped)
    expect(r.multiple).toBeGreaterThan(2)
  })

  it('waiting always costs something', () => {
    expect(costOfWaiting(base, 5).gap).toBeGreaterThan(0)
    expect(costOfWaiting(base, 1).gap).toBeGreaterThan(0)
  })

  it('a longer delay costs strictly more', () => {
    expect(costOfWaiting(base, 10).gap).toBeGreaterThan(costOfWaiting(base, 5).gap)
  })

  it('no delay costs nothing', () => {
    near(costOfWaiting(base, 0).gap, 0)
  })

  it('a delay longer than the horizon leaves nothing invested', () => {
    near(costOfWaiting(base, 50).startingLater, 0)
  })
})

describe('doubling', () => {
  it('the exact doubling time really doubles the money', () => {
    near(Math.pow(1.07, doublingTime(0.07)), 2, 1e-9)
  })

  it('the rule of 72 lands within a few months of exact in its useful band', () => {
    for (const rate of [0.06, 0.07, 0.08, 0.09, 0.1]) {
      expect(Math.abs(ruleOf72(rate) - doublingTime(rate))).toBeLessThan(0.4)
    }
  })

  it('the rule of 72 is near-exact at 8% and drifts either side of it', () => {
    // The shortcut is calibrated around 8%: it slightly overestimates below that
    // and underestimates above. Worth knowing before quoting it in a lesson.
    expect(Math.abs(ruleOf72(0.08) - doublingTime(0.08))).toBeLessThan(0.02)
    expect(ruleOf72(0.06)).toBeGreaterThan(doublingTime(0.06))
    expect(ruleOf72(0.3)).toBeLessThan(doublingTime(0.3))
  })

  it('money never doubles at a zero or negative rate', () => {
    expect(doublingTime(0)).toBe(Infinity)
    expect(ruleOf72(-0.02)).toBe(Infinity)
  })
})

describe('feeDrag', () => {
  const base = { principal: 0, monthly: 500, annualRate: 0.07, years: 40 }

  it('a 1% fee costs far more than 1% of the outcome', () => {
    const { shareOfOutcome } = feeDrag(base, 0.01)
    expect(shareOfOutcome).toBeGreaterThan(0.15)
    expect(shareOfOutcome).toBeLessThan(0.35)
  })

  it('an index-fund-scale fee is close to free', () => {
    expect(feeDrag(base, 0.0003).shareOfOutcome).toBeLessThan(0.01)
  })

  it('no fee costs nothing', () => {
    near(feeDrag(base, 0).lost, 0)
  })

  it('a bigger fee always costs more', () => {
    expect(feeDrag(base, 0.02).lost).toBeGreaterThan(feeDrag(base, 0.005).lost)
  })
})

describe('payOffDebt', () => {
  it('matches a known credit-card amortisation', () => {
    // $5,000 at 22% APR paying $200/mo. The closed-form amortisation term is
    // -ln(1 - rP/PMT) / ln(1+r) = 33.75 months, so the last payment is a partial 34th.
    const r = payOffDebt({ balance: 5000, apr: 0.22, monthlyPayment: 200 })
    expect(r.months).toBe(34)
    near(r.totalInterest, 1749.88, 0.5)
  })

  it('flags a payment that never clears the interest', () => {
    const r = payOffDebt({ balance: 10_000, apr: 0.24, monthlyPayment: 150 })
    expect(r.neverPaysOff).toBe(true)
    expect(r.months).toBe(Infinity)
  })

  it('a zero-interest debt costs exactly the balance', () => {
    const r = payOffDebt({ balance: 1200, apr: 0, monthlyPayment: 100 })
    expect(r.months).toBe(12)
    near(r.totalInterest, 0)
  })

  it('the final payment never overshoots the balance', () => {
    const r = payOffDebt({ balance: 5000, apr: 0.22, monthlyPayment: 200 })
    near(r.totalPaid, 5000 + r.totalInterest, 0.02)
    expect(r.schedule[r.schedule.length - 1]).toBeLessThan(0.01)
  })

  it('a bigger payment always costs less interest', () => {
    const small = payOffDebt({ balance: 8000, apr: 0.199, monthlyPayment: 250 })
    const big = payOffDebt({ balance: 8000, apr: 0.199, monthlyPayment: 500 })
    expect(big.totalInterest).toBeLessThan(small.totalInterest)
    expect(big.months).toBeLessThan(small.months)
  })
})

describe('payoffVsInvest', () => {
  const debt = { balance: 8000, apr: 0.22, monthlyPayment: 200 }

  it('clearing 22% card debt beats a 7% expected return', () => {
    const r = payoffVsInvest({ debt, extraMonthly: 400, investReturn: 0.07, years: 15 })
    expect(r.winner).toBe('payoff')
    expect(r.payoffFirst.interestPaid).toBeLessThan(r.investInstead.interestPaid)
  })

  it('investing wins against very cheap debt', () => {
    const r = payoffVsInvest({
      debt: { balance: 20_000, apr: 0.02, monthlyPayment: 300 },
      extraMonthly: 400,
      investReturn: 0.08,
      years: 20,
    })
    expect(r.winner).toBe('invest')
  })

  it('paying the debt down first clears it sooner', () => {
    const r = payoffVsInvest({ debt, extraMonthly: 400, investReturn: 0.07, years: 15 })
    expect(r.payoffFirst.debtClearedMonth).toBeLessThan(r.investInstead.debtClearedMonth)
  })

  it('with no spare money the two strategies are identical', () => {
    const r = payoffVsInvest({ debt, extraMonthly: 0, investReturn: 0.07, years: 10 })
    expect(r.winner).toBe('tie')
  })
})

describe('employerMatch', () => {
  it('a 50%-of-6% match is an instant 50% return when fully used', () => {
    const r = employerMatch({ salary: 80_000, employeeRate: 0.06, matchRate: 0.5, matchLimit: 0.06 })
    near(r.employeeContribution, 4800)
    near(r.employerContribution, 2400)
    near(r.instantReturn, 0.5)
    near(r.unclaimed, 0)
  })

  it('contributing under the limit leaves employer money unclaimed', () => {
    const r = employerMatch({ salary: 80_000, employeeRate: 0.03, matchRate: 0.5, matchLimit: 0.06 })
    near(r.employerContribution, 1200)
    near(r.unclaimed, 1200)
  })

  it('contributing over the limit adds no further match', () => {
    const at = employerMatch({ salary: 80_000, employeeRate: 0.06, matchRate: 0.5, matchLimit: 0.06 })
    const over = employerMatch({ salary: 80_000, employeeRate: 0.15, matchRate: 0.5, matchLimit: 0.06 })
    near(over.employerContribution, at.employerContribution)
    expect(over.instantReturn).toBeLessThan(at.instantReturn)
  })

  it('contributing nothing claims nothing', () => {
    const r = employerMatch({ salary: 80_000, employeeRate: 0, matchRate: 1, matchLimit: 0.04 })
    near(r.employerContribution, 0)
    expect(r.instantReturn).toBe(0)
    near(r.unclaimed, 3200)
  })
})

describe('targets', () => {
  it('fireNumber is spend divided by the withdrawal rate', () => {
    near(fireNumber(40_000, 0.04), 1_000_000)
    near(fireNumber(40_000, 0.035), 40_000 / 0.035)
  })

  it('coastNumber grows into the target with no further contributions', () => {
    const coast = coastNumber({ target: 1_000_000, yearsToRetirement: 30, realReturn: 0.05 })
    near(coast * Math.pow(1.05, 30), 1_000_000, 1)
    expect(coast).toBeLessThan(1_000_000)
  })

  it('yearsToTarget finds a reachable target', () => {
    const y = yearsToTarget({ principal: 0, monthly: 1000, annualRate: 0.07 }, 100_000)
    expect(y).toBeGreaterThan(6)
    expect(y).toBeLessThan(8)
  })

  it('yearsToTarget is zero when the target is already met', () => {
    expect(yearsToTarget({ principal: 50_000, monthly: 0, annualRate: 0.07 }, 10_000)).toBe(0)
  })

  it('yearsToTarget reports an unreachable target as Infinity', () => {
    expect(yearsToTarget({ principal: 0, monthly: 0, annualRate: 0 }, 1_000_000)).toBe(Infinity)
  })
})

describe('recurringCostAsPortfolio', () => {
  it('separates what was spent from what it could have become', () => {
    const r = recurringCostAsPortfolio({ monthlyCost: 400, years: 30, annualRate: 0.07 })
    near(r.invested, 400 * 360)
    expect(r.endBalance).toBeGreaterThan(r.invested * 2)
    near(r.growth, r.endBalance - r.invested, 0.01)
  })
})
