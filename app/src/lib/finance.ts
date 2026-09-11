/**
 * The money maths.
 *
 * Every function here is pure and unit-tested (see finance.test.ts). Nothing in this
 * file knows about React, storage, or the DOM — a wrong number in a finance app is
 * the one bug that loses all trust, so the maths is kept isolated and verifiable.
 *
 * Conventions:
 *   - Rates are decimals per year. 7% is 0.07, never 7.
 *   - Money is a plain number of currency units (dollars), not cents. Values here are
 *     projections over decades, where float error is many orders of magnitude below
 *     the uncertainty in the rate assumption itself.
 *   - "Real" means inflation-adjusted into today's purchasing power.
 */

/** Months in a year. Named so the intent is readable at call sites. */
const MONTHS = 12

/**
 * Converts an annual nominal rate into the equivalent monthly rate, compounding
 * monthly. Note this is `(1+r)^(1/12) - 1`, not `r/12`: the latter is the APR
 * shortcut lenders use, and it quietly understates growth. We use the exact form so
 * a 7% assumption really returns 7% over a year.
 */
export function monthlyRate(annualRate: number): number {
  return Math.pow(1 + annualRate, 1 / MONTHS) - 1
}

/** Nominal annual rate (APR) to effective annual yield (APY), compounded n times. */
export function aprToApy(apr: number, compoundsPerYear = MONTHS): number {
  if (compoundsPerYear === Infinity) return Math.exp(apr) - 1
  return Math.pow(1 + apr / compoundsPerYear, compoundsPerYear) - 1
}

export interface GrowthInput {
  /** Money invested today. */
  principal: number
  /** Added at the end of every month. */
  monthly: number
  /** Expected nominal annual return, as a decimal. */
  annualRate: number
  /** How long the money stays invested. */
  years: number
  /** Annual fee drag, as a decimal (an 0.03% index fund is 0.0003). */
  expenseRatio?: number
  /** If given, results are also expressed in today's dollars. */
  inflation?: number
  /** Contributions grow by this much each year (a raise you keep investing). */
  contributionGrowth?: number
}

export interface GrowthPoint {
  /** Whole months since the start. */
  month: number
  year: number
  /** Total account value at this point. */
  balance: number
  /** Cumulative money the person actually put in. */
  contributed: number
  /** balance - contributed. The part compounding created. */
  growth: number
  /** Balance expressed in today's purchasing power, when inflation is supplied. */
  realBalance: number
}

/**
 * Simulates a portfolio month by month and returns the whole path.
 *
 * Month-by-month rather than a closed-form annuity formula because the chart needs
 * every point anyway, and because it lets contribution growth, fees and inflation
 * compose without a separate formula for each combination.
 */
export function growthSeries(input: GrowthInput): GrowthPoint[] {
  const {
    principal,
    monthly,
    annualRate,
    years,
    expenseRatio = 0,
    inflation = 0,
    contributionGrowth = 0,
  } = input

  // Fees come out of the return, so they compound against you exactly as returns
  // compound for you. This is the whole reason expense ratios matter so much.
  const netAnnual = annualRate - expenseRatio
  const rm = monthlyRate(netAnnual)
  const im = monthlyRate(inflation)

  const totalMonths = Math.round(years * MONTHS)
  const points: GrowthPoint[] = []

  let balance = principal
  let contributed = principal
  let contribution = monthly

  points.push({
    month: 0,
    year: 0,
    balance,
    contributed,
    growth: 0,
    realBalance: balance,
  })

  for (let m = 1; m <= totalMonths; m++) {
    balance = balance * (1 + rm) + contribution
    contributed += contribution
    // A raise applied once a year, at the anniversary.
    if (m % MONTHS === 0) contribution *= 1 + contributionGrowth

    points.push({
      month: m,
      year: m / MONTHS,
      balance,
      contributed,
      growth: balance - contributed,
      realBalance: balance / Math.pow(1 + im, m),
    })
  }

  return points
}

/** Just the end state, for when the path is not needed. */
export function futureValue(input: GrowthInput): GrowthPoint {
  const series = growthSeries(input)
  return series[series.length - 1]
}

/**
 * The headline number of the whole app: what a delay costs.
 *
 * Someone who starts in `delayYears` invests for that much less time but is otherwise
 * identical. The gap between the two end balances is the price of the delay — and it
 * is almost always shockingly larger than the contributions they skipped, which is
 * precisely the intuition this app exists to install.
 */
export function costOfWaiting(
  input: GrowthInput,
  delayYears: number,
): {
  startingNow: number
  startingLater: number
  gap: number
  /** Money not contributed during the delay. */
  contributionsSkipped: number
  /** gap - contributionsSkipped: growth that never happened. */
  growthForfeited: number
  /** How many dollars of end balance each dollar of delayed contribution cost. */
  multiple: number
} {
  const now = futureValue(input)
  const later = futureValue({
    ...input,
    // The later starter has the same horizon end-date, so fewer years invested,
    // and their lump sum sits in cash (earning nothing) during the delay.
    years: Math.max(0, input.years - delayYears),
  })

  const gap = now.balance - later.balance
  const contributionsSkipped = now.contributed - later.contributed

  return {
    startingNow: now.balance,
    startingLater: later.balance,
    gap,
    contributionsSkipped,
    growthForfeited: gap - contributionsSkipped,
    multiple: contributionsSkipped > 0 ? gap / contributionsSkipped : Infinity,
  }
}

/**
 * Years for money to double at a constant rate — the exact form.
 * `ruleOf72` is the mental-maths approximation people should memorise; this is what
 * the app checks it against so a lesson can show how close the shortcut really is.
 */
export function doublingTime(annualRate: number): number {
  if (annualRate <= 0) return Infinity
  return Math.log(2) / Math.log(1 + annualRate)
}

/** The classic shortcut: 72 / rate-as-a-percentage. */
export function ruleOf72(annualRate: number): number {
  if (annualRate <= 0) return Infinity
  return 72 / (annualRate * 100)
}

/**
 * What a fee costs over a lifetime.
 *
 * Expressed as a share of the final balance rather than a dollar figure, because the
 * share is the number that shocks people: a 1% fee does not cost 1%, it costs roughly
 * a quarter of the outcome over 40 years.
 */
export function feeDrag(
  input: Omit<GrowthInput, 'expenseRatio'>,
  expenseRatio: number,
): { withoutFee: number; withFee: number; lost: number; shareOfOutcome: number } {
  const withoutFee = futureValue({ ...input, expenseRatio: 0 }).balance
  const withFee = futureValue({ ...input, expenseRatio }).balance
  const lost = withoutFee - withFee
  return { withoutFee, withFee, lost, shareOfOutcome: withoutFee > 0 ? lost / withoutFee : 0 }
}

export interface DebtInput {
  balance: number
  /** Nominal annual rate on the debt. */
  apr: number
  /** Paid at the end of each month. */
  monthlyPayment: number
}

export interface DebtResult {
  /** Infinity when the payment never clears the interest. */
  months: number
  totalPaid: number
  totalInterest: number
  /** True when the payment is at or below the monthly interest charge. */
  neverPaysOff: boolean
  /** Remaining balance at the end of each month. */
  schedule: number[]
}

/**
 * Amortises a fixed-payment debt.
 *
 * Card APRs are quoted as nominal annual rates with monthly compounding, so here the
 * monthly rate really is `apr / 12` — deliberately different from `monthlyRate()`
 * above, which models an investment return. Getting this backwards is the classic
 * off-by-a-lot in debt calculators.
 */
export function payOffDebt({ balance, apr, monthlyPayment }: DebtInput): DebtResult {
  const rm = apr / MONTHS
  const firstMonthInterest = balance * rm

  if (monthlyPayment <= firstMonthInterest) {
    return {
      months: Infinity,
      totalPaid: Infinity,
      totalInterest: Infinity,
      neverPaysOff: true,
      schedule: [balance],
    }
  }

  const schedule: number[] = [balance]
  let remaining = balance
  let totalPaid = 0
  let months = 0

  // 100 years is a safety valve; the guard above already rules out true divergence.
  while (remaining > 0.005 && months < 1200) {
    const interest = remaining * rm
    const payment = Math.min(monthlyPayment, remaining + interest)
    remaining = remaining + interest - payment
    totalPaid += payment
    months++
    schedule.push(Math.max(0, remaining))
  }

  return {
    months,
    totalPaid,
    totalInterest: totalPaid - balance,
    neverPaysOff: false,
    schedule,
  }
}

/**
 * The question people actually ask: pay the debt down, or invest the money?
 *
 * Both branches are run over the same horizon and compared on net worth, so the
 * answer accounts for the fact that clearing a debt early frees the payment up to be
 * invested for the remaining time.
 */
export function payoffVsInvest(args: {
  debt: DebtInput
  /** Spare money each month, on top of the minimum payment. */
  extraMonthly: number
  investReturn: number
  /** Horizon to compare over. */
  years: number
}): {
  payoffFirst: { netWorth: number; debtClearedMonth: number; interestPaid: number }
  investInstead: { netWorth: number; debtClearedMonth: number; interestPaid: number }
  /** Positive means paying the debt down first wins. */
  advantageOfPayoff: number
  winner: 'payoff' | 'invest' | 'tie'
} {
  const { debt, extraMonthly, investReturn, years } = args
  const horizon = Math.round(years * MONTHS)
  const rInvest = monthlyRate(investReturn)

  /** Runs one strategy month by month and returns net worth at the horizon. */
  const simulate = (extraToDebt: number, extraToInvest: number) => {
    let remaining = debt.balance
    let invested = 0
    let interestPaid = 0
    let clearedMonth = Infinity
    const rDebt = debt.apr / MONTHS

    for (let m = 1; m <= horizon; m++) {
      invested *= 1 + rInvest

      if (remaining > 0.005) {
        const interest = remaining * rDebt
        interestPaid += interest
        const available = debt.monthlyPayment + extraToDebt
        const payment = Math.min(available, remaining + interest)
        remaining = remaining + interest - payment

        // Anything the debt did not need this month is invested instead.
        invested += available - payment + extraToInvest

        if (remaining <= 0.005 && clearedMonth === Infinity) clearedMonth = m
      } else {
        // Debt is gone: the whole payment plus the spare cash now compounds.
        invested += debt.monthlyPayment + extraToDebt + extraToInvest
      }
    }

    return { netWorth: invested - Math.max(0, remaining), debtClearedMonth: clearedMonth, interestPaid }
  }

  const payoffFirst = simulate(extraMonthly, 0)
  const investInstead = simulate(0, extraMonthly)
  const advantageOfPayoff = payoffFirst.netWorth - investInstead.netWorth

  return {
    payoffFirst,
    investInstead,
    advantageOfPayoff,
    winner:
      Math.abs(advantageOfPayoff) < 1 ? 'tie' : advantageOfPayoff > 0 ? 'payoff' : 'invest',
  }
}

/**
 * An employer 401(k) match, expressed as an immediate return.
 *
 * A "50% of the first 6%" match returns 50 cents per dollar the instant it lands —
 * before any market return at all. Framing it as a return rather than "free money"
 * makes it comparable to every other use of the same dollar, which is the point.
 */
export function employerMatch(args: {
  salary: number
  /** Share of salary the employee contributes, as a decimal. */
  employeeRate: number
  /** Employer pays this share of each matched dollar (0.5 = 50 cents on the dollar). */
  matchRate: number
  /** Matching stops above this share of salary. */
  matchLimit: number
}): {
  employeeContribution: number
  employerContribution: number
  /** Employer dollars per employee dollar, within the matched band. */
  instantReturn: number
  /** Employer money left on the table by not contributing up to the limit. */
  unclaimed: number
} {
  const { salary, employeeRate, matchRate, matchLimit } = args
  const employeeContribution = salary * employeeRate
  const matchedRate = Math.min(employeeRate, matchLimit)
  const employerContribution = salary * matchedRate * matchRate
  const maxEmployer = salary * matchLimit * matchRate

  return {
    employeeContribution,
    employerContribution,
    instantReturn: employeeContribution > 0 ? employerContribution / employeeContribution : 0,
    unclaimed: Math.max(0, maxEmployer - employerContribution),
  }
}

/**
 * The portfolio that sustains a given annual spend at a chosen withdrawal rate.
 * 4% is the convention from the Trinity study; the app always shows it as an
 * assumption the user can move, never as a law.
 */
export function fireNumber(annualSpend: number, withdrawalRate = 0.04): number {
  if (withdrawalRate <= 0) return Infinity
  return annualSpend / withdrawalRate
}

/**
 * Coast FIRE: the balance that, with no further contributions, grows into the
 * target by the retirement date. It is the most motivating milestone in personal
 * finance because it is reachable in years rather than decades.
 */
export function coastNumber(args: {
  target: number
  yearsToRetirement: number
  realReturn: number
}): number {
  const { target, yearsToRetirement, realReturn } = args
  return target / Math.pow(1 + realReturn, yearsToRetirement)
}

/**
 * Years until a portfolio reaches a target, given contributions.
 * Returns Infinity when the target is unreachable on these inputs.
 */
export function yearsToTarget(
  input: Omit<GrowthInput, 'years'>,
  target: number,
  maxYears = 80,
): number {
  if (input.principal >= target) return 0
  const series = growthSeries({ ...input, years: maxYears })
  const hit = series.find((p) => p.balance >= target)
  return hit ? hit.year : Infinity
}

/**
 * Converts a recurring expense into the portfolio it would have become.
 * This is the "latte factor" engine — used in the app to make a point about *large*
 * recurring costs (a car payment, a subscription stack), never to moralise about coffee.
 */
export function recurringCostAsPortfolio(args: {
  monthlyCost: number
  years: number
  annualRate: number
}): { invested: number; endBalance: number; growth: number } {
  const { monthlyCost, years, annualRate } = args
  const end = futureValue({ principal: 0, monthly: monthlyCost, annualRate, years })
  return { invested: end.contributed, endBalance: end.balance, growth: end.growth }
}
