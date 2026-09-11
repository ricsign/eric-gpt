import { describe, expect, it } from 'vitest'
import {
  debtFreeDate,
  federalTax,
  independence,
  investmentHurdle,
  matchGap,
  rentVsBuy,
  runway,
  unusedTaxSpace,
  type Debt,
} from './calculators'
import { FACTS } from '../data/facts'

const STD = FACTS.standardDeductionSingle.value // $16,100 for 2026

describe('federalTax', () => {
  it('someone below the standard deduction owes nothing', () => {
    const t = federalTax(12_000, STD)
    expect(t.taxableIncome).toBe(0)
    expect(t.totalTax).toBe(0)
    expect(t.takeHome).toBe(12_000)
  })

  it('fills brackets from the bottom up', () => {
    // $80k gross, $16,100 deduction -> $63,900 taxable.
    // 10% on the first $12,400, 12% to $50,400, 22% on the rest.
    const t = federalTax(80_000, STD)
    expect(t.taxableIncome).toBe(63_900)

    const expected = 12_400 * 0.1 + (50_400 - 12_400) * 0.12 + (63_900 - 50_400) * 0.22
    expect(t.totalTax).toBeCloseTo(expected, 2)
    expect(t.marginalRate).toBe(0.22)
  })

  it('the effective rate is always below the marginal rate', () => {
    // This is the whole point of the widget — the misconception it exists to kill.
    for (const income of [30_000, 80_000, 150_000, 400_000, 900_000]) {
      const t = federalTax(income, STD)
      expect(t.effectiveRate, `${income}`).toBeLessThan(t.marginalRate)
    }
  })

  it('a raise into a new bracket never reduces take-home pay', () => {
    // The misconception that makes people turn down raises.
    const before = federalTax(50_400 + STD, STD)
    const after = federalTax(50_400 + STD + 1_000, STD)
    expect(after.takeHome).toBeGreaterThan(before.takeHome)
    expect(after.marginalRate).toBeGreaterThan(before.marginalRate)
    // Only the $1,000 is taxed at the higher rate.
    expect(after.totalTax - before.totalTax).toBeCloseTo(1_000 * 0.22, 2)
  })

  it('the per-bracket fill reconstructs the total exactly', () => {
    const t = federalTax(250_000, STD)
    expect(t.fill.reduce((s, f) => s + f.taxFromBracket, 0)).toBeCloseTo(t.totalTax, 6)
    expect(t.fill.reduce((s, f) => s + f.amountInBracket, 0)).toBeCloseTo(t.taxableIncome, 6)
  })

  it('reaches the top bracket on a very large income', () => {
    expect(federalTax(2_000_000, STD).marginalRate).toBe(0.37)
  })
})

describe('debtFreeDate', () => {
  const debts: Debt[] = [
    { id: 'a', name: 'Card', balance: 4_000, apr: 0.24, monthlyPayment: 100 },
    { id: 'b', name: 'Car', balance: 12_000, apr: 0.07, monthlyPayment: 320 },
    { id: 'c', name: 'Store card', balance: 800, apr: 0.29, monthlyPayment: 30 },
  ]

  it('no debts means no wait', () => {
    const r = debtFreeDate([], 200)
    expect(r.months).toBe(0)
    expect(r.hurdleRate).toBe(0)
  })

  it('clears the highest rate first', () => {
    const r = debtFreeDate(debts, 300)
    const store = r.order.find((o) => o.id === 'c')!
    const car = r.order.find((o) => o.id === 'b')!
    expect(store.clearedMonth).toBeLessThan(car.clearedMonth)
  })

  it('extra money always clears the debt sooner and cheaper', () => {
    const slow = debtFreeDate(debts, 0)
    const fast = debtFreeDate(debts, 500)
    expect(fast.months).toBeLessThan(slow.months)
    expect(fast.totalInterest).toBeLessThan(slow.totalInterest)
  })

  it('the hurdle rate is balance-weighted, not a plain average', () => {
    const r = debtFreeDate(debts, 0)
    const plainAverage = (0.24 + 0.07 + 0.29) / 3
    const weighted = (4000 * 0.24 + 12000 * 0.07 + 800 * 0.29) / 16800
    expect(r.hurdleRate).toBeCloseTo(weighted, 6)
    expect(r.hurdleRate).not.toBeCloseTo(plainAverage, 3)
  })

  it('frees a cleared debt\'s minimum onto the next one', () => {
    // The avalanche only works if freed minimums roll forward. Without that, the
    // total months would match paying each debt in isolation.
    const isolated = Math.max(
      ...debts.map((d) => {
        let b = d.balance
        let m = 0
        while (b > 0.005 && m < 1200) {
          b = b + b * (d.apr / 12) - Math.min(d.monthlyPayment, b + b * (d.apr / 12))
          m++
        }
        return m
      }),
    )
    expect(debtFreeDate(debts, 0).months).toBeLessThan(isolated)
  })

  it('flags a payment plan that never clears', () => {
    const r = debtFreeDate([{ id: 'x', name: 'Card', balance: 10_000, apr: 0.26, monthlyPayment: 100 }], 0)
    expect(r.neverClears).toBe(true)
    expect(r.months).toBe(Infinity)
  })
})

describe('investmentHurdle', () => {
  it('returns the debt\'s effective rate as the crossover', () => {
    const h = investmentHurdle({ balance: 5000, apr: 0.24, monthlyPayment: 200 })
    expect(h.hurdleRate).toBeCloseTo(Math.pow(1 + 0.24 / 12, 12) - 1, 10)
    expect(h.clearedByTypicalEquityAssumption).toBe(false)
  })

  it('cheap debt is cleared by a conventional equity assumption', () => {
    expect(
      investmentHurdle({ balance: 20_000, apr: 0.03, monthlyPayment: 300 })
        .clearedByTypicalEquityAssumption,
    ).toBe(true)
  })

  it('never returns a verdict — only a threshold', () => {
    const h = investmentHurdle({ balance: 5000, apr: 0.18, monthlyPayment: 200 })
    expect(Object.keys(h)).toEqual(['hurdleRate', 'clearedByTypicalEquityAssumption'])
  })
})

describe('independence', () => {
  const base = {
    age: 30,
    annualSpend: 40_000,
    invested: 50_000,
    monthly: 1_500,
    takeHomeAnnual: 70_000,
    realReturn: 0.05,
  }

  it('the target is spend divided by the withdrawal rate', () => {
    expect(independence(base).target).toBeCloseTo(1_000_000, 6)
    expect(independence({ ...base, withdrawalRate: 0.035 }).target).toBeCloseTo(
      40_000 / 0.035,
      6,
    )
  })

  it('coasting is always a nearer milestone than the full target', () => {
    // This is the motivational point of Coast FIRE: reachable in years, not decades.
    const r = independence(base)
    expect(r.coastTarget).toBeLessThan(r.target)
    expect(r.yearsToCoast).toBeLessThan(r.yearsToTarget)
  })

  it('recognises someone who has already coasted', () => {
    const r = independence({ ...base, invested: 400_000 })
    expect(r.alreadyCoasting).toBe(true)
    expect(r.yearsToCoast).toBe(0)
  })

  it('reports the savings rate against take-home, not gross', () => {
    expect(independence(base).savingsRate).toBeCloseTo((1500 * 12) / 70_000, 10)
  })

  it('saving more always arrives sooner', () => {
    expect(independence({ ...base, monthly: 3000 }).yearsToTarget).toBeLessThan(
      independence(base).yearsToTarget,
    )
  })

  it('reports an unreachable target honestly rather than guessing', () => {
    const r = independence({ ...base, invested: 0, monthly: 0, realReturn: 0 })
    expect(r.yearsToTarget).toBe(Infinity)
    expect(r.ageAtTarget).toBeNull()
  })
})

describe('runway', () => {
  it('converts savings into months of cover', () => {
    const r = runway({ liquidSavings: 9_000, monthlyEssentials: 3_000, targetMonths: 6 })
    expect(r.months).toBe(3)
    expect(r.gap).toBe(9_000)
  })

  it('reports no gap once the target is met', () => {
    expect(runway({ liquidSavings: 20_000, monthlyEssentials: 3_000, targetMonths: 6 }).gap).toBe(0)
  })

  it('says how long closing the gap takes', () => {
    const r = runway({ liquidSavings: 0, monthlyEssentials: 2_000, targetMonths: 3 })
    expect(r.monthsToClose(1_000)).toBe(6)
    expect(r.monthsToClose(0)).toBe(Infinity)
  })

  it('handles zero spending without dividing by zero', () => {
    expect(runway({ liquidSavings: 5_000, monthlyEssentials: 0, targetMonths: 6 }).months).toBe(
      Infinity,
    )
  })
})

describe('unusedTaxSpace', () => {
  it('values unused room at the marginal rate only', () => {
    const r = unusedTaxSpace({ contributed: 10_000, limit: 24_500, marginalRate: 0.22 })
    expect(r.unused).toBe(14_500)
    expect(r.deductionValueThisYear).toBeCloseTo(14_500 * 0.22, 6)
  })

  it('never reports negative room for an over-contributor', () => {
    const r = unusedTaxSpace({ contributed: 30_000, limit: 24_500, marginalRate: 0.24 })
    expect(r.unused).toBe(0)
    expect(r.percentUsed).toBe(1)
  })
})

describe('rentVsBuy', () => {
  const base = {
    homePrice: 450_000,
    downPayment: 90_000,
    mortgageRate: 0.065,
    termYears: 30,
    monthlyRent: 2_200,
    rentGrowth: 0.03,
    homeAppreciation: 0.03,
    ownershipCostRate: 0.02,
    investmentReturn: 0.07,
    sellingCostRate: 0.06,
  }

  it('returns a break-even year, not a recommendation', () => {
    const r = rentVsBuy(base)
    expect(r).toHaveProperty('breakEvenYear')
    expect(r).not.toHaveProperty('recommendation')
    expect(r.series).toHaveLength(30)
  })

  it('cheap rent pushes break-even later or out of reach', () => {
    const cheapRent = rentVsBuy({ ...base, monthlyRent: 900 })
    const dearRent = rentVsBuy({ ...base, monthlyRent: 3_800 })
    const later = cheapRent.breakEvenYear ?? Infinity
    const sooner = dearRent.breakEvenYear ?? Infinity
    expect(sooner).toBeLessThan(later)
  })

  it('counts the down payment\'s opportunity cost on the renting side', () => {
    // The most commonly omitted term. A higher assumed market return must make
    // renting look better, or the term is not being applied.
    const lowReturn = rentVsBuy({ ...base, investmentReturn: 0.02 }).breakEvenYear ?? Infinity
    const highReturn = rentVsBuy({ ...base, investmentReturn: 0.12 }).breakEvenYear ?? Infinity
    expect(highReturn).toBeGreaterThan(lowReturn)
  })

  it('selling costs delay break-even', () => {
    const low = rentVsBuy({ ...base, sellingCostRate: 0 }).breakEvenYear ?? Infinity
    const high = rentVsBuy({ ...base, sellingCostRate: 0.1 }).breakEvenYear ?? Infinity
    expect(high).toBeGreaterThanOrEqual(low)
  })

  it('reports null rather than inventing a year when owning never wins', () => {
    const r = rentVsBuy({ ...base, monthlyRent: 300, homeAppreciation: -0.02 })
    expect(r.breakEvenYear).toBeNull()
  })
})

describe('matchGap', () => {
  const base = {
    salary: 90_000,
    currentRate: 0.02,
    matchLimit: 0.06,
    matchRate: 0.5,
    age: 30,
    retireAge: 65,
    annualReturn: 0.07,
  }

  it('prices the unclaimed match annually and at retirement', () => {
    const r = matchGap(base)
    // 4 points of salary short, matched at 50 cents.
    expect(r.annualUnclaimed).toBeCloseTo(90_000 * 0.04 * 0.5, 6)
    expect(r.byRetirement).toBeGreaterThan(r.annualUnclaimed * 35)
  })

  it('reports nothing unclaimed once the limit is reached', () => {
    const r = matchGap({ ...base, currentRate: 0.06 })
    expect(r.annualUnclaimed).toBe(0)
    expect(r.byRetirement).toBe(0)
  })

  it('contributing beyond the limit does not create a negative gap', () => {
    expect(matchGap({ ...base, currentRate: 0.15 }).annualUnclaimed).toBe(0)
  })

  it('says what monthly contribution would close the gap', () => {
    expect(matchGap(base).extraMonthlyNeeded).toBeCloseTo((90_000 * 0.04) / 12, 6)
  })
})
