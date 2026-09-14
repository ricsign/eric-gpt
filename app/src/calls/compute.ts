/**
 * The compute engine.
 *
 * One pure function per call. Each takes the drag position and the player's
 * profile and returns everything the screen and the receipt need. These run on
 * every drag frame, so everything here is closed-form: no month-by-month walks
 * over forty years.
 *
 * A note on why this file carries its own annuity helpers when lib/finance.ts
 * already compounds money: `growthSeries()` walks month by month because the
 * chart needs every point. On a drag frame we only ever need the endpoint, and
 * a 420-iteration loop per frame per dependent figure is the one thing the
 * contract in types.ts explicitly rules out. `fvAnnuity`/`fvLump` below are the
 * closed forms of exactly what `futureValue()` computes, and compute.test.ts
 * asserts they agree with it to the cent. Everything else — the match maths,
 * the fee drag, the monthly-rate convention — comes from lib/finance.ts.
 *
 * Two conventions inherited from lib/finance.ts and kept religiously:
 *   - Investment returns use `monthlyRate(r)` = (1+r)^(1/12)-1.
 *   - Debt uses apr/12, because that is how lenders actually quote and charge.
 */

import type { CallOutcome, ComputeFn, Profile } from './types'
import { employerMatch as matchMath, feeDrag, monthlyRate } from '../lib/finance'
import { BRACKETS_2026_SINGLE, FACTS } from '../data/facts'
import { duration, money, moneyCompact, percent } from '../lib/format'

/* ---- Shared assumptions --------------------------------------------------- */

/**
 * 7% nominal is the long-run US equity assumption the whole app is built on.
 * It is stated as an assumption on every call, never as a promise.
 */
const NOMINAL = 0.07
const RM = monthlyRate(NOMINAL)
const RETIRE_AT = 65

/** Cash yield used wherever money sits in savings rather than the market. */
const CASH = FACTS.hysaCompetitive.value
const CASH_M = monthlyRate(CASH)

const STD_DED = FACTS.standardDeductionSingle.value
const WAGE_BASE = FACTS.socialSecurityWageBase.value

/**
 * Horizon. Floored at a year so a profile at or past 65 still produces finite,
 * signed numbers instead of collapsing every call to zero.
 */
function monthsTo65(p: Profile): number {
  return Math.max(12, Math.round((RETIRE_AT - p.age) * 12))
}

/* ---- Closed-form money ---------------------------------------------------- */

/** FV of `monthly` paid at the end of each of `n` months. */
function fvAnnuity(monthly: number, n: number, rate = RM): number {
  if (n <= 0 || monthly === 0) return 0
  if (rate === 0) return monthly * n
  return (monthly * (Math.pow(1 + rate, n) - 1)) / rate
}

/** FV of a single amount left alone for `n` months. */
function fvLump(amount: number, n: number, rate = RM): number {
  if (n <= 0) return amount
  return amount * Math.pow(1 + rate, n)
}

/* ---- Closed-form debt ----------------------------------------------------- */

/**
 * Months to clear `balance` at monthly rate `i` paying `p` at each month end.
 * Fractional on purpose: a whole-month result would make the final partial
 * payment show up as a step in the drag, and the difference between two splits
 * is often less than a month.
 */
function monthsToClear(balance: number, i: number, p: number): number {
  if (balance <= 0) return 0
  if (p <= 0) return Infinity
  if (i === 0) return balance / p
  if (p <= balance * i) return Infinity
  return -Math.log(1 - (i * balance) / p) / Math.log(1 + i)
}

/** Balance after `n` months of paying `p`. Grows when `p` is under the interest. */
function balanceAfter(balance: number, i: number, p: number, n: number): number {
  if (n <= 0) return balance
  if (i === 0) return Math.max(0, balance - p * n)
  const g = Math.pow(1 + i, n)
  return Math.max(0, balance * g - (p * (g - 1)) / i)
}

/* ---- Tax ------------------------------------------------------------------ */

/** Federal tax on an already-deducted taxable income, 2026 single brackets. */
function federalTax(taxable: number): number {
  if (taxable <= 0) return 0
  let tax = 0
  let lower = 0
  for (const b of BRACKETS_2026_SINGLE) {
    if (taxable <= lower) break
    tax += (Math.min(taxable, b.upTo) - lower) * b.rate
    lower = b.upTo
  }
  return tax
}

/** FICA. OASDI stops at the wage base; Medicare does not. */
function fica(salary: number): number {
  return Math.min(salary, WAGE_BASE) * 0.062 + salary * 0.0145
}

/** Federal + FICA only. State tax is deliberately out of scope and said so. */
function takeHome(salary: number): number {
  return salary - federalTax(Math.max(0, salary - STD_DED)) - fica(salary)
}

/* ---- Receipt helpers ------------------------------------------------------ */

const line = (label: string, value: string, emphasis = false) =>
  emphasis ? { label, value, emphasis: true } : { label, value }

/* ==========================================================================
 * 1. employerMatch
 *
 * Value: employee contribution as a percent of salary, 0-15.
 *
 * Model. The most common US formula — 50 cents per employee dollar up to 6% of
 * pay (Vanguard, How America Saves 2026, FACTS.averageEmployerContribution).
 * Both streams are invested at 7% nominal to 65. There is no vesting schedule,
 * no salary growth and no contribution limit in the model: at 15% of a $62,000
 * salary nobody is near the $24,500 deferral cap, and vesting would add a
 * parameter the player cannot see or drag.
 *
 * at65 rises for the whole dial, because more invested is more invested. The
 * lesson is not in at65 — it is in blockTint. Blocks 1-6 are matched and tint
 * accent; 7-15 are the player's own money alone and tint plain. Benefit stops
 * dead at 6 while cost keeps climbing, so 6 is the cheapest position that
 * captures every employer dollar. That is the call's optimal.
 * ========================================================================== */

const MATCH_RATE = 0.5
const MATCH_LIMIT = 0.06

export const employerMatch: ComputeFn = (value, profile) => {
  const rate = Math.max(0, value) / 100
  const m = matchMath({
    salary: profile.salary,
    employeeRate: rate,
    matchRate: MATCH_RATE,
    matchLimit: MATCH_LIMIT,
  })

  const n = monthsTo65(profile)
  const cost = m.employeeContribution / 12
  const benefit = m.employerContribution
  const at65 = fvAnnuity((m.employeeContribution + m.employerContribution) / 12, n)

  return {
    cost,
    benefit,
    at65,
    breakdown: [
      line('YOU / MO', money(cost)),
      line('EMPLOYER / YR', money(benefit), true),
      line('LEFT ON TABLE', money(m.unclaimed)),
      line('ON YOUR DOLLAR', percent(m.instantReturn, 0)),
      line('AT 65', moneyCompact(at65)),
    ],
    // Indexed from 0, so index i is the i% position. 0% earns nothing, 1-6% are
    // matched, and everything past 6% is the player funding themselves.
    blockTint: (index) => (index >= 1 && index <= 6 ? 'accent' : 'plain'),
  }
}

/* ==========================================================================
 * 2. debtSplit
 *
 * Value: dollars of a fixed $500 monthly payment sent to the LOWER-rate card.
 *
 * Model. Two real-shaped balances: $4,200 at 24.99% and $2,800 at 11.99%. The
 * split holds until one card clears, then the entire $500 cascades onto the
 * survivor — which is what actually happens, and is why every split is finite
 * even when one card is being paid nothing at all. Minimum payments are ignored
 * on purpose: a minimum would put a floor under the dial and hide the lesson
 * that the last dollar sent to the cheap card is the expensive one.
 *
 * at65 counts only the payoff-date difference: finishing k months earlier means
 * k more months of $500 compounding before 65. The interest saved is not added
 * on top of that — it is the same money, and counting it twice would inflate
 * the hero number.
 *
 * Optimal is 0: every dollar to the 24.99% card.
 * ========================================================================== */

const CARD_HIGH = { balance: 4_200, apr: 0.2499 }
const CARD_LOW = { balance: 2_800, apr: 0.1199 }
const SPLIT_PAYMENT = 500

function splitPlan(toLow: number): { months: number; interest: number } {
  const toHigh = SPLIT_PAYMENT - toLow
  const iH = CARD_HIGH.apr / 12
  const iL = CARD_LOW.apr / 12

  const nH = monthsToClear(CARD_HIGH.balance, iH, toHigh)
  const nL = monthsToClear(CARD_LOW.balance, iL, toLow)
  const t1 = Math.min(nH, nL)

  // Interest on the card that clears is exactly what it paid minus principal;
  // on the card that does not, it is the growth plus whatever it did pay.
  const intH =
    nH <= t1
      ? toHigh * nH - CARD_HIGH.balance
      : balanceAfter(CARD_HIGH.balance, iH, toHigh, t1) - CARD_HIGH.balance + toHigh * t1
  const intL =
    nL <= t1
      ? toLow * nL - CARD_LOW.balance
      : balanceAfter(CARD_LOW.balance, iL, toLow, t1) - CARD_LOW.balance + toLow * t1

  const survivorHigh = nH > nL
  const rem = survivorHigh
    ? balanceAfter(CARD_HIGH.balance, iH, toHigh, t1)
    : balanceAfter(CARD_LOW.balance, iL, toLow, t1)
  const i2 = survivorHigh ? iH : iL

  const n2 = monthsToClear(rem, i2, SPLIT_PAYMENT)
  const int2 = rem > 0 ? SPLIT_PAYMENT * n2 - rem : 0

  return { months: t1 + n2, interest: intH + intL + int2 }
}

export const debtSplit: ComputeFn = (value, profile) => {
  const toLow = Math.min(SPLIT_PAYMENT, Math.max(0, value))
  const here = splitPlan(toLow)
  const worst = splitPlan(SPLIT_PAYMENT)

  const n = monthsTo65(profile)
  // Both plans free the $500 up on payoff; the earlier finisher simply gets
  // more months of it before 65.
  const at65 =
    fvAnnuity(SPLIT_PAYMENT, n - here.months) - fvAnnuity(SPLIT_PAYMENT, n - worst.months)

  const interestAvoided = worst.interest - here.interest
  const years = here.months / 12

  return {
    cost: SPLIT_PAYMENT,
    benefit: years > 0 ? interestAvoided / years : interestAvoided,
    at65,
    breakdown: [
      line('TO 24.99%', money(SPLIT_PAYMENT - toLow)),
      line('TO 11.99%', money(toLow)),
      line('DEBT FREE IN', duration(here.months)),
      line('INTEREST PAID', money(here.interest), true),
      line('AT 65', moneyCompact(at65)),
    ],
  }
}

/* ==========================================================================
 * 3. emergencyFund
 *
 * Value: months of essential costs held in cash, 0-12.
 *
 * Model. Essentials are 50% of take-home (federal + FICA, single filer, no
 * state tax — stated, not hidden). Holding the fund costs the spread between
 * cash at 4% (FDIC competitive range) and the market at 7%: 3% a year, forever.
 *
 * The payoff side needs a distribution, not a point estimate, or the answer is
 * always "hold nothing" or "hold everything". Shock severity is modelled as
 * exponential with a mean of 4 months of essentials, which makes the expected
 * uncovered dollars E[max(0, X-F)] = mean * e^(-F/mean) — closed form, smooth,
 * and correctly showing that the first month of cover buys far more than the
 * twelfth. Shock probability is 1/3 a year. Borrowing the uncovered part on a
 * 24.99% card repaid over 24 months costs 0.28 per dollar, which is computed
 * below rather than asserted.
 *
 * The mean severity and the annual probability are modelling priors, not
 * sourced statistics, and they are the two numbers that move the answer. They
 * are calibrated so the optimum lands at F* = mean * ln(p * borrowCost / drag)
 * = 4.5 months, inside the 3-6 month range the call declares. Anyone who thinks
 * their own job is less stable is right to sit at the top of that range.
 * ========================================================================== */

const SHOCK_P = 1 / 3
const SHOCK_MEAN_MONTHS = 4
const CARD_APR = CARD_HIGH.apr

/** Total interest per dollar borrowed at `apr` and repaid over `months`. */
function borrowCostPerDollar(apr: number, months: number): number {
  const i = apr / 12
  const pay = i / (1 - Math.pow(1 + i, -months))
  return pay * months - 1
}

const BORROW_COST = borrowCostPerDollar(CARD_APR, 24)

export const emergencyFund: ComputeFn = (value, profile) => {
  const months = Math.max(0, value)
  const essentials = (takeHome(profile.salary) / 12) * 0.5
  const fund = essentials * months
  const mean = essentials * SHOCK_MEAN_MONTHS

  const dragAnnual = fund * (NOMINAL - CASH)
  // Expected dollars still uncovered when a shock lands, against the same
  // figure with no fund at all — so benefit is what the cash actually bought.
  const uncovered = mean * Math.exp(-fund / mean)
  const benefit = SHOCK_P * BORROW_COST * (mean - uncovered)

  const net = benefit - dragAnnual
  const at65 = fvAnnuity(net / 12, monthsTo65(profile))

  return {
    cost: dragAnnual / 12,
    benefit,
    at65,
    breakdown: [
      line('ESSENTIALS / MO', money(essentials)),
      line('CASH HELD', money(fund)),
      line('CASH DRAG / MO', money(dragAnnual / 12)),
      line('SHOCK COVERED / YR', money(benefit)),
      line('NET AT 65', moneyCompact(at65), true),
    ],
    // Cover you do not have is exposure; cover past the point where the drag
    // outruns the protection is expensive insurance.
    blockTint: (index) => (index <= 2 ? 'loss' : index <= 6 ? 'accent' : 'plain'),
  }
}

/* ==========================================================================
 * 4. promoDeadline
 *
 * Value: months taken to clear a $3,000 0%-APR furniture offer, 1-24, with a
 * 12-month promotional window.
 *
 * Model. Equal principal payments of 3000/months. Inside the window the offer
 * is genuinely free. Miss it by a single month and the deferred-interest clause
 * fires: interest is charged retroactively at 26.99% on the balance carried
 * since day one, capitalised, and the remaining balance then amortises at that
 * rate. Retroactive interest over the window is the closed form of an
 * arithmetically declining balance, which is how issuers compute average daily
 * balance.
 *
 * The cash the player has not yet paid is assumed to sit in savings at the FDIC
 * competitive rate, which is why paying in one month is very slightly worse
 * than paying in twelve. That float is computed over the player's declared
 * schedule, not over the extended one a missed deadline creates — a
 * simplification worth a few dollars on a number that turns on a thousand.
 *
 * Optimal is 12: the last month that is still free.
 * ========================================================================== */

const PROMO_PRINCIPAL = 3_000
const PROMO_WINDOW = 12
const PROMO_APR = 0.2699

function promoPlan(months: number): { interest: number; float: number; actual: number } {
  const m = Math.max(1, Math.round(months))
  const pay = PROMO_PRINCIPAL / m

  // Interest earned on money not yet handed over, across the declared schedule.
  const float =
    CASH_M * (m * PROMO_PRINCIPAL - pay * ((m * (m - 1)) / 2))

  if (m <= PROMO_WINDOW) return { interest: 0, float, actual: m }

  const i = PROMO_APR / 12
  // Retroactive to day one, on the balance that was outstanding each month.
  const accrued =
    i * (PROMO_WINDOW * PROMO_PRINCIPAL - pay * ((PROMO_WINDOW * (PROMO_WINDOW - 1)) / 2))
  const carried = PROMO_PRINCIPAL - pay * PROMO_WINDOW + accrued

  const n2 = monthsToClear(carried, i, pay)
  const after = Number.isFinite(n2) ? pay * n2 - carried : 0

  return { interest: accrued + after, float, actual: PROMO_WINDOW + n2 }
}

export const promoDeadline: ComputeFn = (value, profile) => {
  const m = Math.min(24, Math.max(1, Math.round(value)))
  const here = promoPlan(m)
  const worst = promoPlan(24)

  const netCost = here.interest - here.float
  const avoided = worst.interest - worst.float - netCost
  const at65 = fvLump(avoided, monthsTo65(profile))

  return {
    cost: PROMO_PRINCIPAL / m,
    benefit: avoided / Math.max(1 / 12, here.actual / 12),
    at65,
    breakdown: [
      line('PAYMENT / MO', money(PROMO_PRINCIPAL / m)),
      line('DEFERRED INTEREST', money(here.interest), true),
      line('ACTUALLY CLEAR IN', duration(here.actual)),
      line('TOTAL PAID', money(PROMO_PRINCIPAL + here.interest)),
      line('AT 65', moneyCompact(at65)),
    ],
    // Index 0 is month 1. Everything inside the window is free; the block after
    // it is where the retroactive clause detonates.
    blockTint: (index) => (index < PROMO_WINDOW ? 'accent' : 'loss'),
  }
}

/* ==========================================================================
 * 5. anchorOffer
 *
 * Value: counter-offer as a percent above the initial offer, 0-20.
 *
 * Model. Upside is linear and permanent: one point of base compounds through
 * 3% raises for the rest of the career and is invested at 7% to 65. Downside is
 * convex: P(offer pulled) = 1 - e^(-(a/A)^2), and a pulled offer costs a
 * three-month search gap, also carried to 65. Expected value is
 * (1-p)*upside - p*cost, and its derivative gives A^2 = 2a^2 + 2a*(cost/upside)
 * at the optimum. A = 12.178 is that expression solved backwards at a = 8 for
 * the default profile, which is how the constant was chosen.
 *
 * Say plainly what this is: the withdrawal curve is a stylised shape, not a
 * measurement. Nobody publishes a credible P(pulled | ask) and this model does
 * not pretend to. It is calibrated so the optimum lands in the 5-10% band that
 * recruiters and negotiation coaches consistently recommend, and what it
 * teaches is the shape of the tradeoff — linear gain against convex risk — not
 * a probability forecast. The optimum drifts down with age in the model, which
 * is correct: fewer years of career left to amortise the same risk over.
 * ========================================================================== */

const ASK_RISK_SCALE = 12.178
const SEARCH_GAP_YEARS = 0.25
const RAISE = 0.03

/** Value at 65 of $1/yr of extra base pay that grows with raises. */
function raiseStream(years: number): number {
  const n = Math.max(1, Math.round(years))
  const x = (1 + RAISE) / (1 + NOMINAL)
  return Math.pow(1 + NOMINAL, n) * ((1 - Math.pow(x, n)) / (1 - x))
}

export const anchorOffer: ComputeFn = (value, profile) => {
  const ask = Math.max(0, value)
  const years = Math.max(1, RETIRE_AT - profile.age)
  const n = monthsTo65(profile)

  const pulled = 1 - Math.exp(-Math.pow(ask / ASK_RISK_SCALE, 2))
  const perPoint = 0.01 * profile.salary * raiseStream(years)
  const gapCost = fvLump(SEARCH_GAP_YEARS * profile.salary, n)

  const at65 = (1 - pulled) * perPoint * ask - pulled * gapCost
  const firstYear = (1 - pulled) * 0.01 * ask * profile.salary

  return {
    // No cash leaves the player's pocket here; what they spend is risk. Shown
    // as the expected search-gap loss spread over a year so it is comparable
    // to every other call's monthly cost.
    cost: (pulled * SEARCH_GAP_YEARS * profile.salary) / 12,
    benefit: firstYear,
    at65,
    breakdown: [
      line('ASK', percent(ask / 100, 0)),
      line('YEAR ONE', money(firstYear)),
      line('OFFER PULLED', percent(pulled, 0)),
      line('LIFETIME AT 65', moneyCompact(at65), true),
    ],
  }
}

/* ==========================================================================
 * 6. rothSplit
 *
 * Value: percent of contributions directed to Roth, 0-100.
 *
 * Model. The player directs 10% of gross pay (capped at the 2026 deferral
 * limit) at retirement. The pre-tax share reduces taxable income; the Roth
 * share is taxed on the way in, and the tax is computed bracket by bracket
 * against the 2026 single brackets rather than at a flat marginal rate, so a
 * contribution that straddles a bracket edge is handled correctly.
 *
 * Everything is projected in TODAY'S DOLLARS, at a 5% real return with 1% real
 * raises — because brackets are inflation-indexed, and comparing a nominal 2061
 * income to 2026 brackets would invent tax the player will never pay. This is
 * the only call whose at65 is real rather than nominal, and the receipt says so.
 *
 * The retirement rate is computed, not assumed: 4% of the pre-tax balance is
 * withdrawn on top of Social Security (40% replacement of final pay, 85%
 * taxable — the share that applies to anyone with meaningful withdrawals), and
 * the rate that matters is the marginal one on the withdrawal itself, not the
 * effective rate on everything. The 40% replacement is the one unsourced input
 * and it moves the answer, so it is stated.
 *
 * The honest result, and it is worth saying out loud: for a single filer this
 * model has pre-tax winning or tying at every salary. At the default $62,000
 * the two rates land on 12% exactly and the dial is flat to the dollar — a
 * genuine wash, which is the real lesson of the Roth argument. Above and below
 * that band pre-tax wins outright, so the optimal is 0.
 * ========================================================================== */

const REAL_RETURN = 0.05
const REAL_RAISE = 0.01
const SS_REPLACEMENT = 0.4
const SS_TAXABLE_SHARE = 0.85

/** Value at 65, in today's dollars, of $1/yr of contributions growing with raises. */
function realStream(years: number): number {
  const n = Math.max(1, Math.round(years))
  const x = (1 + REAL_RAISE) / (1 + REAL_RETURN)
  return Math.pow(1 + REAL_RETURN, n) * ((1 - Math.pow(x, n)) / (1 - x))
}

export const rothSplit: ComputeFn = (value, profile) => {
  const share = Math.min(1, Math.max(0, value / 100))
  const years = Math.max(1, RETIRE_AT - profile.age)

  const gross = Math.min(profile.salary * 0.1, FACTS.contrib401k.value)
  const trad = gross * (1 - share)
  const taxableNow = Math.max(0, profile.salary - STD_DED)

  const taxWithout = federalTax(taxableNow)
  const taxWithTrad = federalTax(Math.max(0, taxableNow - trad))
  const taxSaved = taxWithout - taxWithTrad
  // The Roth share is the slice of income left exposed after the pre-tax slice
  // comes out, so its tax is the difference across exactly that slice.
  const rothTax = taxWithTrad - federalTax(Math.max(0, taxableNow - gross))

  const g = realStream(years)
  const tradFv = trad * g
  const rothFv = (gross * share - rothTax) * g

  const ss = SS_REPLACEMENT * profile.salary * Math.pow(1 + REAL_RAISE, years)
  const ssTaxable = ss * SS_TAXABLE_SHARE
  const withdrawal = tradFv * 0.04
  const base = federalTax(Math.max(0, ssTaxable - STD_DED))
  const retireRate =
    withdrawal > 0
      ? (federalTax(Math.max(0, ssTaxable + withdrawal - STD_DED)) - base) / withdrawal
      : 0

  const at65 = tradFv * (1 - retireRate) + rothFv
  const cost = (gross - taxSaved) / 12

  return {
    cost,
    benefit: taxSaved,
    at65,
    breakdown: [
      line('TAKE-HOME HIT / MO', money(cost)),
      line('RATE NOW', percent(trad > 0 ? taxSaved / trad : 0, 1)),
      line('RATE AT 65', percent(retireRate, 1)),
      line('TAX SAVED / YR', money(taxSaved)),
      line("AT 65, TODAY'S $", moneyCompact(at65), true),
    ],
  }
}

/* ==========================================================================
 * 7. repairOrReplace
 *
 * Value: dollars spent repairing a $5,000 car, 0-4,000.
 *
 * Model. A repair buys life with diminishing returns and a hard ceiling: an old
 * car has a finite amount left in it no matter what you spend. Months bought =
 * 28 * (1 - e^(-R/1000)). Both constants are judgement calls — a $5,000 car
 * with a big repair behind it is good for a bit over two more years, and the
 * first $1,000 buys most of that.
 *
 * The alternative is a $22,000 used car on a 60-month loan at 7.5%, which is
 * $440.83 a month, computed here rather than quoted. Neither figure is in
 * facts.ts, so both are assumptions.
 *
 * The comparison is total cost of having wheels over the same 60 months:
 * spend R, drive it for L(R) months, then start the loan. Replacing today is
 * the R=0 case. The optimum falls out of the shape at R* = 1000 * ln(payment *
 * 28 / 1000) = $2,513, which on a $100 grid is $2,500 — half the car's value,
 * which is the rule of thumb the call is teaching.
 * ========================================================================== */

const CAR_VALUE = 5_000
const CAR_LIFE_CEILING = 28
const CAR_LIFE_SCALE = 1_000
const REPLACEMENT_PRICE = 22_000
const REPLACEMENT_TERM = 60
const REPLACEMENT_APR = 0.075

const REPLACEMENT_PAYMENT = (() => {
  const i = REPLACEMENT_APR / 12
  return (REPLACEMENT_PRICE * i) / (1 - Math.pow(1 + i, -REPLACEMENT_TERM))
})()

const carHorizonCost = (repair: number): number => {
  const life = CAR_LIFE_CEILING * (1 - Math.exp(-repair / CAR_LIFE_SCALE))
  return repair + REPLACEMENT_PAYMENT * Math.max(0, REPLACEMENT_TERM - life)
}

export const repairOrReplace: ComputeFn = (value, profile) => {
  const repair = Math.max(0, value)
  const life = CAR_LIFE_CEILING * (1 - Math.exp(-repair / CAR_LIFE_SCALE))
  const total = carHorizonCost(repair)
  const saved = carHorizonCost(0) - total

  const n = monthsTo65(profile)
  // The saving arrives month by month over the five years, then compounds.
  const at65 = fvLump(fvAnnuity(saved / REPLACEMENT_TERM, REPLACEMENT_TERM), n - REPLACEMENT_TERM)

  return {
    cost: total / REPLACEMENT_TERM,
    benefit: saved / 5,
    at65,
    breakdown: [
      line('REPAIR', money(repair)),
      line('SHARE OF VALUE', percent(repair / CAR_VALUE, 0)),
      line('BUYS YOU', duration(life)),
      line('5-YR COST OF WHEELS', money(total), true),
      line('AT 65', moneyCompact(at65)),
    ],
  }
}

/* ==========================================================================
 * 8. feeDragCall
 *
 * Value: expense ratio in basis points, 3-150.
 *
 * Model. A portfolio the player plausibly already has — one year of salary
 * invested, plus 10% of salary a month — run to 65 at 7% through
 * `feeDrag()` from lib/finance.ts, which is the only place in this file that
 * still walks month by month. That is two passes of the series per frame, a few
 * hundred microseconds, and it is worth it to keep the fee maths in one place.
 *
 * at65 is the wealth this expense ratio KEEPS against the worst fund on the
 * dial (150bp), so higher is better here exactly as it is on every other call.
 * The terminal wealth lost to the fee outright is on the receipt, emphasised,
 * because that is the number that lands: the share of the outcome, not the
 * share of the balance.
 *
 * The dial bottoms out at 3bp because that is what the cheapest broad index
 * funds actually charge. Optimal is the floor; there is no argument on the
 * other side.
 * ========================================================================== */

const WORST_ER = 0.015

export const feeDragCall: ComputeFn = (value, profile) => {
  const er = Math.max(0, value) / 10_000
  const n = monthsTo65(profile)
  const principal = profile.salary
  const monthly = (profile.salary * 0.1) / 12

  const input = { principal, monthly, annualRate: NOMINAL, years: n / 12 }
  const drag = feeDrag(input, er)

  const worstRm = monthlyRate(NOMINAL - WORST_ER)
  const worst = fvLump(principal, n, worstRm) + fvAnnuity(monthly, n, worstRm)

  return {
    cost: (principal * er) / 12,
    benefit: (WORST_ER - er) * principal,
    at65: drag.withFee - worst,
    breakdown: [
      line('EXPENSE RATIO', `${Math.round(value)} BP`),
      line('FEE / MO NOW', money((principal * er) / 12)),
      line('LOST TO FEES BY 65', money(drag.lost), true),
      line('SHARE OF OUTCOME', percent(drag.shareOfOutcome, 1)),
      line('YOU KEEP', moneyCompact(drag.withFee)),
    ],
    // Index 0 is 3bp. Broad index territory, then actively-managed territory,
    // then the funds that quietly take a quarter of the outcome.
    blockTint: (index) => (index + 3 <= 20 ? 'accent' : index + 3 >= 75 ? 'loss' : 'plain'),
  }
}

/* ==========================================================================
 * 9. rentVsBuy
 *
 * Value: years the player expects to stay, 0-15.
 *
 * Model. A $360,000 home, 20% down, 3% closing, 30 years at 6.5%, 1.5% a year
 * in tax/insurance/maintenance, 3% appreciation, 6% to sell. Rent starts at
 * $2,160 — 0.6% of price a month, the usual rent-to-price ratio — and grows
 * with the same 3%. None of these are in facts.ts; all are assumptions, and the
 * 6% selling cost and the down payment's opportunity cost are the two that
 * decide the answer.
 *
 * Two net worths at the end of the stay:
 *   buyer  = sale proceeds after 6% minus the mortgage balance
 *   renter = down payment and closing costs invested at 7%, plus every month
 *            the buyer spent more than the renter, also invested at 7%
 * The mortgage balance is the standard closed form; the cash-flow difference is
 * two geometric sums, settled mid-year. No amortisation table.
 *
 * Mortgage interest deduction is deliberately out: at these numbers the
 * standard deduction wins for a single filer, so modelling it would flatter
 * buying on a return almost nobody actually claims.
 *
 * Break-even lands just under five years, which is where the call's optimal of
 * 5-and-up comes from. The block tint is the model's own sign, not a rule typed
 * in by hand.
 * ========================================================================== */

const HOME = 360_000
const DOWN_SHARE = 0.2
const CLOSING_SHARE = 0.03
const MORT_APR = 0.065
const MORT_MONTHS = 360
const SELL_COST = 0.06
const OWN_COST = 0.015
const APPRECIATION = 0.03
const RENT_MONTHLY = 2_160

const MORT_PRINCIPAL = HOME * (1 - DOWN_SHARE)
const MORT_I = MORT_APR / 12
const MORT_GROWTH = Math.pow(1 + MORT_I, MORT_MONTHS)
const MORT_PAYMENT = (MORT_PRINCIPAL * MORT_I * MORT_GROWTH) / (MORT_GROWTH - 1)

/** Sum of q^k for k = 0..n-1. */
function geom(q: number, n: number): number {
  if (n <= 0) return 0
  return Math.abs(1 - q) < 1e-12 ? n : (1 - Math.pow(q, n)) / (1 - q)
}

/** Positive means buying is ahead at the end of a `years`-long stay. */
function buyMinusRent(years: number): { delta: number; cashGap: number } {
  const y = Math.max(0, years)
  const m = Math.round(y * 12)

  const owed =
    (MORT_PRINCIPAL * (MORT_GROWTH - Math.pow(1 + MORT_I, m))) / (MORT_GROWTH - 1)
  const buyer = (1 - SELL_COST) * HOME * Math.pow(1 + APPRECIATION, y) - owed

  // Annual cash the buyer spends over the renter: fixed payment plus running
  // costs that inflate, against rent that inflates at the same rate.
  const fixed = 12 * MORT_PAYMENT
  const inflating = OWN_COST * HOME - 12 * RENT_MONTHLY
  const n = Math.round(y)
  const cashGap = fixed * n + inflating * geom(1 + APPRECIATION, n)
  const invested =
    n > 0
      ? Math.pow(1 + NOMINAL, n - 0.5) *
        (fixed * geom(1 / (1 + NOMINAL), n) +
          inflating * geom((1 + APPRECIATION) / (1 + NOMINAL), n))
      : 0

  const renter =
    HOME * (DOWN_SHARE + CLOSING_SHARE) * Math.pow(1 + NOMINAL, y) + invested

  return { delta: buyer - renter, cashGap }
}

export const rentVsBuy: ComputeFn = (value, profile) => {
  const years = Math.min(15, Math.max(0, value))
  const { delta, cashGap } = buyMinusRent(years)
  const n = monthsTo65(profile)
  const at65 = fvLump(delta, n - Math.round(years * 12))

  return {
    cost: years > 0 ? cashGap / (years * 12) : MORT_PAYMENT + (OWN_COST * HOME) / 12 - RENT_MONTHLY,
    benefit: delta / Math.max(1, years),
    at65,
    breakdown: [
      line('OWNING / MO', money(MORT_PAYMENT + (OWN_COST * HOME) / 12)),
      line('RENTING / MO', money(RENT_MONTHLY)),
      line('COST TO SELL', money(SELL_COST * HOME * Math.pow(1 + APPRECIATION, years))),
      line('BUY MINUS RENT', money(delta), true),
      line('AT 65', moneyCompact(at65)),
    ],
    // The tint is the model evaluated at that block, not a rule typed by hand:
    // red until the sale covers what it cost to get in and back out.
    blockTint: (index) => (buyMinusRent(index).delta >= 0 ? 'accent' : 'loss'),
  }
}

/* ==========================================================================
 * 10. timingMarket
 *
 * Value: number of the market's best days missed by being out, 0-30.
 *
 * Model. Missing the ten best days roughly halves terminal wealth over a long
 * horizon — the finding every version of this study reproduces. Rather than
 * hardcode a drag, the code solves for it: the per-year drag that turns the
 * horizon's growth into exactly half is computed from the player's own horizon,
 * so a 30-year-old and a 55-year-old both get an internally consistent number.
 *
 * The shape is concave, drag = D * (1 - e^(-d/15)), because the best days
 * cluster — they land in the middle of the worst weeks, next to each other, and
 * the first few you miss are the ones that cost most. By 30 days the drag has
 * nearly saturated, leaving about 30% of the wealth, which is the right order
 * of magnitude against the published studies.
 *
 * What this does NOT model, and should be said: there is no upside to being
 * out. Missing the worst days would help by a similar amount. The honest claim
 * is that nobody reliably separates the two, not that being out is mechanically
 * a loss — and that is exactly why the optimum is zero days.
 * ========================================================================== */

const CLUSTER_TAU = 15
const HALVING_DAYS = 10

export const timingMarket: ComputeFn = (value, profile) => {
  const days = Math.max(0, value)
  const n = monthsTo65(profile)
  const years = n / 12

  // Drag at ten days that turns the whole horizon's growth into half of itself.
  const halving = NOMINAL - (Math.pow(0.5, 1 / years) * (1 + NOMINAL) - 1)
  const scale = halving / (1 - Math.exp(-HALVING_DAYS / CLUSTER_TAU))
  const drag = scale * (1 - Math.exp(-days / CLUSTER_TAU))
  const rate = Math.max(-0.9, NOMINAL - drag)

  const principal = profile.salary
  const monthly = (profile.salary * 0.1) / 12
  const rm = monthlyRate(rate)

  const at65 = fvLump(principal, n, rm) + fvAnnuity(monthly, n, rm)
  const perfect = fvLump(principal, n, RM) + fvAnnuity(monthly, n, RM)
  const gaveUp = perfect - at65

  return {
    cost: gaveUp / n,
    benefit: principal * rate,
    at65,
    breakdown: [
      line('DAYS MISSED', `${Math.round(days)}`),
      line('ANNUAL RETURN', percent(rate, 1)),
      line('YOU KEEP', percent(perfect > 0 ? at65 / perfect : 0, 0)),
      line('GAVE UP BY 65', money(gaveUp), true),
      line('AT 65', moneyCompact(at65)),
    ],
    blockTint: (index) => (index === 0 ? 'accent' : index <= 10 ? 'loss' : 'plain'),
  }
}

/* ---- Registry ------------------------------------------------------------- */

/** Keyed by `Call.compute`. */
export const COMPUTE: Record<string, ComputeFn> = {
  employerMatch,
  debtSplit,
  emergencyFund,
  promoDeadline,
  anchorOffer,
  rothSplit,
  repairOrReplace,
  feeDragCall,
  rentVsBuy,
  timingMarket,
}

/** Every outcome field is a real number, whatever the player drags it to. */
export function isSane(o: CallOutcome): boolean {
  return (
    Number.isFinite(o.cost) &&
    Number.isFinite(o.benefit) &&
    Number.isFinite(o.at65) &&
    o.breakdown.length >= 3 &&
    o.breakdown.length <= 6 &&
    o.breakdown.filter((b) => b.emphasis).length === 1 &&
    o.breakdown.every((b) => b.label.length > 0 && b.value.length > 0)
  )
}
