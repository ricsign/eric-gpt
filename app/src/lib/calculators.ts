import {
  coastNumber,
  fireNumber,
  futureValue,
  monthlyRate,
  payOffDebt,
  payoffVsInvest,
  yearsToTarget,
  type DebtInput,
} from './finance'
import { BRACKETS_2026_SINGLE } from '../data/facts'

/**
 * The utility spine.
 *
 * These are the reason to open the app in month six, when the novelty of the daily
 * puzzle has worn off. Three rules govern every function here, and they are product
 * decisions as much as engineering ones:
 *
 *  1. **Return a date, a dollar amount, a threshold, or a percentile — never a
 *     verdict.** "Break-even in year 7" is a computation. "You should buy" is
 *     advice, and advice tailored to an individual is a regulated activity no
 *     disclaimer cures. The output type is the compliance boundary, enforced here
 *     rather than in copy.
 *  2. **Deterministic.** Same inputs, same answer, forever. Nothing here calls a
 *     model, and nothing here is stochastic — the number must be reproducible and
 *     auditable, or it cannot be trusted for a real decision.
 *  3. **Every assumption is an input.** There are no hidden constants. If a figure
 *     matters to the answer, the user can see and change it.
 */

/* ------------------------------------------------------------------------- *
 * Tax
 * ------------------------------------------------------------------------- */

export interface TaxResult {
  taxableIncome: number
  totalTax: number
  /** The rate on the next dollar earned. */
  marginalRate: number
  /** Total tax divided by gross income — always lower than the marginal rate. */
  effectiveRate: number
  /** Per-bracket fill, for the stacked-bar widget. */
  fill: { rate: number; amountInBracket: number; taxFromBracket: number }[]
  takeHome: number
}

/**
 * Federal income tax only, single filer.
 *
 * State tax is deliberately excluded. Including it would multiply the maintenance
 * surface by fifty and, more importantly, would push the output from "here is how
 * brackets work on a number like yours" towards "here is your tax bill", which is a
 * different product with different obligations.
 */
export function federalTax(grossIncome: number, standardDeduction: number): TaxResult {
  const taxableIncome = Math.max(0, grossIncome - standardDeduction)

  let remaining = taxableIncome
  let previousCap = 0
  let totalTax = 0
  let marginalRate: number = BRACKETS_2026_SINGLE[0].rate
  const fill: TaxResult['fill'] = []

  for (const bracket of BRACKETS_2026_SINGLE) {
    if (remaining <= 0) break
    const width = bracket.upTo - previousCap
    const amountInBracket = Math.min(remaining, width)
    const taxFromBracket = amountInBracket * bracket.rate

    fill.push({ rate: bracket.rate, amountInBracket, taxFromBracket })
    totalTax += taxFromBracket
    marginalRate = bracket.rate

    remaining -= amountInBracket
    previousCap = bracket.upTo
  }

  return {
    taxableIncome,
    totalTax,
    marginalRate,
    effectiveRate: grossIncome > 0 ? totalTax / grossIncome : 0,
    fill,
    takeHome: grossIncome - totalTax,
  }
}

/* ------------------------------------------------------------------------- *
 * Debt
 * ------------------------------------------------------------------------- */

export interface Debt extends DebtInput {
  id: string
  name: string
}

export interface DebtFreeResult {
  /** Months until every debt is cleared, paying minimums plus the extra. */
  months: number
  totalInterest: number
  /** The order the avalanche method clears them in. */
  order: { id: string; name: string; clearedMonth: number; apr: number }[]
  /** Balance-weighted average rate — the hurdle every other use of a dollar must beat. */
  hurdleRate: number
  neverClears: boolean
}

/**
 * Debt-free date under the avalanche method.
 *
 * Returns the date and the order, not an instruction. The `hurdleRate` is the most
 * useful single number here: it converts a pile of debts into one threshold the
 * user can compare any other decision against.
 */
export function debtFreeDate(debts: Debt[], extraMonthly: number): DebtFreeResult {
  if (debts.length === 0) {
    return { months: 0, totalInterest: 0, order: [], hurdleRate: 0, neverClears: false }
  }

  const totalBalance = debts.reduce((s, d) => s + d.balance, 0)
  const hurdleRate =
    totalBalance > 0 ? debts.reduce((s, d) => s + d.apr * d.balance, 0) / totalBalance : 0

  // Highest rate first. This always minimises total interest; the snowball ordering
  // is offered in the UI as an explicit alternative, never silently substituted.
  const queue = [...debts].sort((a, b) => b.apr - a.apr)
  const state = queue.map((d) => ({ ...d, remaining: d.balance, clearedMonth: Infinity }))

  let month = 0
  let totalInterest = 0

  while (state.some((d) => d.remaining > 0.005) && month < 1200) {
    month++
    let spare = extraMonthly

    // Minimums first, on everything still outstanding.
    for (const d of state) {
      if (d.remaining <= 0.005) {
        // A cleared debt frees its minimum for the next target — the avalanche.
        spare += d.monthlyPayment
        continue
      }
      const interest = d.remaining * (d.apr / 12)
      totalInterest += interest
      const payment = Math.min(d.monthlyPayment, d.remaining + interest)
      d.remaining = d.remaining + interest - payment
      if (d.remaining <= 0.005 && d.clearedMonth === Infinity) d.clearedMonth = month
    }

    // Then everything spare onto the highest-rate debt still standing.
    for (const d of state) {
      if (spare <= 0) break
      if (d.remaining <= 0.005) continue
      const payment = Math.min(spare, d.remaining)
      d.remaining -= payment
      spare -= payment
      if (d.remaining <= 0.005 && d.clearedMonth === Infinity) d.clearedMonth = month
    }
  }

  const neverClears = state.some((d) => d.remaining > 0.005)

  return {
    months: neverClears ? Infinity : month,
    totalInterest,
    order: state.map((d) => ({
      id: d.id,
      name: d.name,
      clearedMonth: d.clearedMonth,
      apr: d.apr,
    })),
    hurdleRate,
    neverClears,
  }
}

/**
 * Payoff versus invest, expressed as the threshold rather than as an answer.
 *
 * The honest output is "above this expected return, investing wins" — because the
 * expected return is the user's assumption, not a fact, and handing them the
 * crossover point lets them decide with their own belief rather than ours.
 */
export function investmentHurdle(debt: DebtInput): {
  /** Expected annual return above which investing beats paying this debt down. */
  hurdleRate: number
  /** Whether a conventional long-run equity assumption clears it. */
  clearedByTypicalEquityAssumption: boolean
} {
  // Paying down a debt is a risk-free, tax-free return equal to its own rate, so the
  // crossover is simply the debt's effective annual rate.
  const effective = Math.pow(1 + debt.apr / 12, 12) - 1
  return {
    hurdleRate: effective,
    clearedByTypicalEquityAssumption: effective < 0.07,
  }
}

/* ------------------------------------------------------------------------- *
 * Independence
 * ------------------------------------------------------------------------- */

export interface IndependenceResult {
  /** The portfolio that sustains the spend at the chosen withdrawal rate. */
  target: number
  /** Years until the target is reached on current contributions. */
  yearsToTarget: number
  /** Age at which that happens. */
  ageAtTarget: number | null
  /** Balance that would reach the target with no further contributions. */
  coastTarget: number
  /** Years until coasting alone would get there. Usually far sooner, and the point. */
  yearsToCoast: number
  alreadyCoasting: boolean
  /** Share of take-home currently being saved. */
  savingsRate: number
}

export function independence(args: {
  age: number
  annualSpend: number
  invested: number
  monthly: number
  takeHomeAnnual: number
  realReturn: number
  withdrawalRate?: number
  retireAge?: number
}): IndependenceResult {
  const {
    age,
    annualSpend,
    invested,
    monthly,
    takeHomeAnnual,
    realReturn,
    withdrawalRate = 0.04,
    retireAge = 65,
  } = args

  const target = fireNumber(annualSpend, withdrawalRate)
  const years = yearsToTarget({ principal: invested, monthly, annualRate: realReturn }, target)

  const coastTarget = coastNumber({
    target,
    yearsToRetirement: Math.max(0, retireAge - age),
    realReturn,
  })
  const yearsToCoast = yearsToTarget(
    { principal: invested, monthly, annualRate: realReturn },
    coastTarget,
  )

  return {
    target,
    yearsToTarget: years,
    ageAtTarget: Number.isFinite(years) ? age + years : null,
    coastTarget,
    yearsToCoast,
    alreadyCoasting: invested >= coastTarget,
    savingsRate: takeHomeAnnual > 0 ? (monthly * 12) / takeHomeAnnual : 0,
  }
}

/* ------------------------------------------------------------------------- *
 * Runway
 * ------------------------------------------------------------------------- */

/**
 * Emergency runway, in months, and the gap to a chosen target.
 *
 * Returns months and a dollar gap. It does not opine on how many months is right —
 * three, six and twelve are all defensible depending on job security and household
 * structure, and the app is not in a position to know which applies.
 */
export function runway(args: {
  liquidSavings: number
  monthlyEssentials: number
  targetMonths: number
}): { months: number; gap: number; monthsToClose: (monthlySaving: number) => number } {
  const { liquidSavings, monthlyEssentials, targetMonths } = args
  const months = monthlyEssentials > 0 ? liquidSavings / monthlyEssentials : Infinity
  const gap = Math.max(0, targetMonths * monthlyEssentials - liquidSavings)

  return {
    months,
    gap,
    monthsToClose: (monthlySaving: number) =>
      monthlySaving > 0 ? Math.ceil(gap / monthlySaving) : Infinity,
  }
}

/* ------------------------------------------------------------------------- *
 * Unused tax-advantaged space
 * ------------------------------------------------------------------------- */

/**
 * What unused tax-advantaged room is worth this year.
 *
 * Deliberately conservative: it values the deduction at the marginal rate and stops
 * there. Projecting the compounded lifetime value of the deferral requires
 * assumptions about future tax rates that nobody has, and stating one would dress a
 * guess up as a calculation.
 */
export function unusedTaxSpace(args: {
  contributed: number
  limit: number
  marginalRate: number
}): { unused: number; deductionValueThisYear: number; percentUsed: number } {
  const { contributed, limit, marginalRate } = args
  const unused = Math.max(0, limit - contributed)
  return {
    unused,
    deductionValueThisYear: unused * marginalRate,
    percentUsed: limit > 0 ? Math.min(1, contributed / limit) : 0,
  }
}

/* ------------------------------------------------------------------------- *
 * Rent versus buy
 * ------------------------------------------------------------------------- */

export interface RentVsBuyResult {
  /** The year owning overtakes renting. null when it never does on these inputs. */
  breakEvenYear: number | null
  /** Cumulative net position by year: positive means owning is ahead. */
  series: { year: number; ownNetCost: number; rentNetCost: number }[]
}

/**
 * Rent versus buy, as a break-even year.
 *
 * Never "you should buy". The output is the year the cumulative cost of owning
 * falls below the cumulative cost of renting — which answers the only question that
 * actually depends on the numbers: how long you would need to stay.
 */
export function rentVsBuy(args: {
  homePrice: number
  downPayment: number
  mortgageRate: number
  termYears: number
  monthlyRent: number
  rentGrowth: number
  homeAppreciation: number
  /** Property tax, insurance, maintenance, as a share of home value per year. */
  ownershipCostRate: number
  /** Return the down payment would have earned if invested instead. */
  investmentReturn: number
  /** Agent fees and closing costs on sale, as a share of sale price. */
  sellingCostRate: number
  years?: number
}): RentVsBuyResult {
  const {
    homePrice,
    downPayment,
    mortgageRate,
    termYears,
    monthlyRent,
    rentGrowth,
    homeAppreciation,
    ownershipCostRate,
    investmentReturn,
    sellingCostRate,
    years = 30,
  } = args

  const loan = homePrice - downPayment
  const r = mortgageRate / 12
  const n = termYears * 12
  const payment = r > 0 ? (loan * r) / (1 - Math.pow(1 + r, -n)) : loan / n

  const series: RentVsBuyResult['series'] = []
  let breakEvenYear: number | null = null

  let balance = loan
  let ownCashOut = downPayment
  let rentCashOut = 0
  let rentThisMonth = monthlyRent
  // The renter invests the down payment, and that opportunity cost is the single
  // most commonly omitted term in this comparison.
  let renterPortfolio = downPayment
  const rInvest = monthlyRate(investmentReturn)

  for (let m = 1; m <= years * 12; m++) {
    // Owning.
    if (balance > 0) {
      const interest = balance * r
      balance = Math.max(0, balance + interest - payment)
      ownCashOut += payment
    }
    ownCashOut += (homePrice * ownershipCostRate) / 12

    // Renting.
    rentCashOut += rentThisMonth
    renterPortfolio *= 1 + rInvest
    if (m % 12 === 0) rentThisMonth *= 1 + rentGrowth

    if (m % 12 === 0) {
      const year = m / 12
      const homeValue = homePrice * Math.pow(1 + homeAppreciation, year)
      const equity = homeValue * (1 - sellingCostRate) - balance

      // Net cost of each path if you walked away at the end of this year.
      const ownNetCost = ownCashOut - equity
      const rentNetCost = rentCashOut - renterPortfolio

      series.push({ year, ownNetCost, rentNetCost })

      if (breakEvenYear === null && ownNetCost < rentNetCost) breakEvenYear = year
    }
  }

  return { breakEvenYear, series }
}

/* ------------------------------------------------------------------------- *
 * Match gap
 * ------------------------------------------------------------------------- */

/**
 * Unclaimed employer match, compounded to retirement.
 *
 * The annual figure is often small enough to shrug at. Compounded over a career it
 * usually is not, and that is the honest way to present it.
 */
export function matchGap(args: {
  salary: number
  currentRate: number
  matchLimit: number
  matchRate: number
  age: number
  retireAge: number
  annualReturn: number
}): { annualUnclaimed: number; byRetirement: number; extraMonthlyNeeded: number } {
  const { salary, currentRate, matchLimit, matchRate, age, retireAge, annualReturn } = args

  const shortfall = Math.max(0, matchLimit - currentRate)
  const annualUnclaimed = salary * shortfall * matchRate
  const years = Math.max(0, retireAge - age)

  return {
    annualUnclaimed,
    byRetirement: futureValue({
      principal: 0,
      monthly: annualUnclaimed / 12,
      annualRate: annualReturn,
      years,
    }).balance,
    extraMonthlyNeeded: (salary * shortfall) / 12,
  }
}

/** Re-exported so the calculators module is a single import for screens. */
export { payOffDebt, payoffVsInvest }
