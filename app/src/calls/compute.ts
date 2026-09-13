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
