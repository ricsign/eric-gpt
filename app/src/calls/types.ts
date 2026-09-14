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
  /**
   * Fill colour for one step POSITION on the control: 0 is `variable.min`,
   * 1 is one `step` above it, and so on. It is not the index of a drawn block —
   * the bar draws at most fifteen of those however many steps a call has, and
   * translates positions to them itself. Author against the values you know.
   */
  blockTint?: (position: number) => 'accent' | 'plain' | 'loss'
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
  /** The best play. See `Optimal`. */
  optimal: Optimal
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

/**
 * Where the best play sits.
 *
 * A plain number where one position is right. A range where several are — an
 * emergency fund is defensible anywhere from three to six months, and the
 * employer-match call has no upper bound at all, because saving *more* than the
 * match cap is not a mistake and calling it "Overshot" would be false advice.
 *
 * A function where the answer genuinely depends on who is asking. Roth versus
 * traditional is the honest case: the crossover moves with the player's bracket,
 * and pretending it is a constant would teach the wrong rule to whoever is on
 * the wrong side of it.
 */
export type Optimal =
  | number
  | { min: number; max: number }
  | ((profile: Profile) => number | { min: number; max: number })

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

/** Collapses a possibly profile-dependent optimum to a concrete one. */
export function resolveOptimal(
  optimal: Optimal,
  profile: Profile,
): number | { min: number; max: number } {
  return typeof optimal === 'function' ? optimal(profile) : optimal
}

/** Where a value sits against the call's optimal. */
export function judge(value: number, optimal: Optimal, profile: Profile): Verdict {
  const o = resolveOptimal(optimal, profile)
  if (typeof o === 'number') {
    if (value === o) return 'optimal'
    return value < o ? 'short' : 'over'
  }
  if (value < o.min) return 'short'
  if (value > o.max) return 'over'
  return 'optimal'
}

/**
 * The value a play should be measured against.
 *
 * For a play that is already optimal this is the play itself, so the loss shown
 * is exactly zero. Measuring a correct answer against the midpoint of a range —
 * which is what an earlier version did — told someone who banked three months of
 * expenses that they had left money behind, which is both wrong and the single
 * most trust-destroying thing this screen could say.
 *
 * For a play outside the range it is the nearest edge: the closest correct
 * answer they could have given, not the most extreme one.
 */
export function referenceValue(value: number, optimal: Optimal, profile: Profile): number {
  const o = resolveOptimal(optimal, profile)
  if (typeof o === 'number') return o
  if (value < o.min) return o.min
  if (value > o.max) return o.max
  return value
}

/** A single representative optimum, for display and for seeding a comparison. */
export function optimalValue(optimal: Optimal, profile: Profile): number {
  const o = resolveOptimal(optimal, profile)
  return typeof o === 'number' ? o : (o.min + o.max) / 2
}

/** Number of discrete positions on a variable — also the crowd array length. */
export function stepCount(v: CallVariable): number {
  return Math.round((v.max - v.min) / v.step) + 1
}

/** Substitutes profile values into a fact tile. */
export function resolveFact(v: string, profile: Profile, money: (n: number) => string): string {
  return v.replace('{{salary}}', money(profile.salary))
}
