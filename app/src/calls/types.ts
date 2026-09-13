/**
 * The contract every call is built against.
 *
 * A "call" is one day's decision. The player drags a single continuous variable,
 * every dependent number recomputes on the same frame, and they lock it in. That
 * is the whole product: the lesson is discovered through the control's behaviour,
 * never explained before they have run into it.
 *
 * Adding a call means authoring one `Call` record and — occasionally — one pure
 * compute function. Nothing else. If a scenario cannot be expressed as a
 * draggable variable, it is an article, and it does not ship.
 */

/** The one thing we ask for, once, at first run. Everything else is derived. */
export interface Profile {
  /** Gross annual salary. Defaults to the national-median-ish figure on skip. */
  salary: number
  /** Age, used only where a projection horizon genuinely depends on it. */
  age: number
}

export const DEFAULT_PROFILE: Profile = { salary: 62_000, age: 30 }

/** One line on the receipt, and one row in the outcome breakdown. */
export interface BreakdownLine {
  /** Left side. Monospace, uppercase, dotted leader to the value. */
  label: string
  /** Right side, already formatted. */
  value: string
  /** Marks the line the whole call turns on, so the receipt can weight it. */
  emphasis?: boolean
}

/**
 * What a compute function returns for a given drag position.
 *
 * Every field is recomputed on every drag frame, so these must stay cheap —
 * closed-form arithmetic, no month-by-month loops over 40 years.
 */
export interface CallOutcome {
  /** What this choice costs the player per month, in dollars. */
  cost: number
  /** What it captures or avoids per year. Free money, avoided interest, etc. */
  benefit: number
  /** Projected value of this decision at 65. The hero number. */
  at65: number
  /** Receipt line items for this position. */
  breakdown: BreakdownLine[]
  /**
   * Optional per-block tint for the control, indexed from 0.
   *
   * This is how the teaching happens without words: in the employer-match call
   * blocks 1-6 fill accent (matched) and 7-15 fill white (unmatched), so the
   * player *sees* the match stop while their monthly cost keeps climbing.
   */
  blockTint?: (index: number) => 'accent' | 'plain' | 'loss'
}

export type ComputeFn = (value: number, profile: Profile) => CallOutcome

/** The draggable variable. Always continuous, always quantized to `step`. */
export interface CallVariable {
  key: string
  min: number
  max: number
  step: number
  /** Rendered immediately after the value in the 64px readout. */
  unit: string
  /** Lowercase caption under the readout. */
  label: string
  /** Where the control starts. Never the optimal value. */
  start: number
}

/** A fixed parameter tile. `{{salary}}` interpolates from the profile. */
export interface CallFact {
  k: string
  v: string
}

export type Domain =
  | 'retirement'
  | 'debt'
  | 'savings'
  | 'credit'
  | 'income'
  | 'tax'
  | 'vehicles'
  | 'investing'
  | 'housing'

export interface Call {
  /** Monotonic. Also the deep-link path: /142 */
  id: number
  /** Maximum 12 words. Three lines of display type. */
  title: string
  domain: Domain
  variable: CallVariable
  fixed: CallFact[]
  /** Name of the pure function in compute.ts. */
  compute: string
  /**
   * The best play. A number, or a range when more than one position is
   * defensible — an emergency fund is right anywhere from three to six months,
   * and pretending otherwise would be false precision.
   */
  optimal: number | { min: number; max: number }
  /** Under 10 words, imperative mood. Filed to the player's Rules forever. */
  rule: string
  /**
   * Seed distribution across the variable's range, from real-world priors.
   *
   * Used until enough live results exist to aggregate, and blended with them
   * after. Length must equal the number of steps in the variable.
   */
  crowd: number[]
  /** Teaser headline for the next call. Shown on the receipt. */
  tomorrow: string
  /** Shown on demand. Never present an estimate as a promise. */
  assumptions: string
}

export type Verdict = 'optimal' | 'short' | 'over'

/** A locked-in answer. Immutable once written. */
export interface CallResult {
  callId: number
  value: number
  verdict: Verdict
  /** Signed: what this play is worth at 65 against the optimal play. */
  delta: number
  at65: number
  /** Day key (YYYY-MM-DD) it was played, in the player's local time. */
  day: string
  /** True for a replay after the day closed. Does not move the Tab. */
  practice: boolean
}

/** Where a value sits against the call's optimal. */
export function judge(value: number, optimal: Call['optimal']): Verdict {
  if (typeof optimal === 'number') {
    if (value === optimal) return 'optimal'
    return value < optimal ? 'short' : 'over'
  }
  if (value < optimal.min) return 'short'
  if (value > optimal.max) return 'over'
  return 'optimal'
}

/** The single number a call is scored against. */
export function optimalValue(optimal: Call['optimal']): number {
  return typeof optimal === 'number' ? optimal : (optimal.min + optimal.max) / 2
}

/** Number of discrete positions on a variable — also the crowd array length. */
export function stepCount(v: CallVariable): number {
  return Math.round((v.max - v.min) / v.step) + 1
}

/** Substitutes profile values into a fact tile. */
export function resolveFact(v: string, profile: Profile, money: (n: number) => string): string {
  return v.replace('{{salary}}', money(profile.salary))
}
