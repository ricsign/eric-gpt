/**
 * The crowd.
 *
 * Every call ships with an authored seed distribution drawn from real-world
 * priors, because a histogram that says "1 player answered, it was you" is not
 * a crowd and teaches nothing. There is no backend yet, so the only live signal
 * is this device's own answers. The seed therefore has to carry the chart early
 * and get out of the way later, without a visible seam on the day real data
 * overtakes it.
 *
 * Everything here is pure and works in weights, not percentages: the seeds in
 * the registry are authored as relative sizes and only their ratios matter.
 */

/**
 * The best play, as the scorer needs it.
 *
 * A point, or the band of answers that are all equally right. Structurally the
 * same as the registry's `Optimal` once a profile-dependent optimum has been
 * resolved to a concrete one — this module deliberately does not import that
 * type, because it must not need a profile to draw a chart.
 */
export type Optimal = number | { min: number; max: number }

/**
 * How many real observations the seed is worth.
 *
 * This is a Dirichlet prior expressed in the same units as the data, which is
 * what makes the decay automatic rather than a hand-tuned fade: the seed holds
 * SEED_WEIGHT / (SEED_WEIGHT + observations) of the chart, so it is the whole
 * picture at 0 observations, half of it at 120, and under a tenth past 1,100.
 * Nothing ever switches over — the blend moves one observation at a time.
 *
 * 120 is chosen so a single device, which will only ever contribute a handful of
 * answers to one call, can never visibly bend an authored distribution, while a
 * real aggregate of thousands drowns it out completely.
 */
export const SEED_WEIGHT = 120

/** Output is on a 0-100 scale, so one player is one unit of the crowd. */
const CROWD_SCALE = 100

/**
 * Percentile bounds.
 *
 * Callers render this at whole-percent precision, and "better than 100% of
 * players" is false in both directions: you are counted in your own tie group,
 * so you are never better than everyone, and never worse than everyone either.
 * These are the largest and smallest values that still round to 99 and 1.
 */
const MAX_PERCENTILE = 99.4
const MIN_PERCENTILE = 0.6

/**
 * Tolerance on bucket distances.
 *
 * A 0.1-step lattice does not land on its own values in binary, and two answers
 * that are equally good but differ by 2e-16 must not be scored apart.
 */
const FUZZ = 1e-9

/** Weights must be finite and positive; anything else is dropped, not trusted. */
function sanitize(source: number[], length: number): number[] {
  const out = new Array<number>(length).fill(0)
  for (let i = 0; i < length; i++) {
    const w = source[i]
    if (Number.isFinite(w) && w > 0) out[i] = w
  }
  return out
}

function total(weights: number[]): number {
  let sum = 0
  for (const w of weights) sum += w
  return sum
}

/** The value a bucket index stands for. */
function valueAt(index: number, min: number, step: number): number {
  return min + index * step
}

/**
 * How far a play is from being right.
 *
 * A band scores zero anywhere inside it. Three months of expenses and six months
 * of expenses are both correct, and ranking one above the other — which is what
 * measuring against the midpoint of the band does — tells someone who gave a
 * right answer that they were closer to wrong than their neighbour. An unbounded
 * band (`max: Infinity`, where contributing more is never a mistake) works out of
 * the same arithmetic.
 */
function distanceTo(value: number, optimal: Optimal): number {
  if (typeof optimal === 'number') return Math.abs(value - optimal)
  if (value < optimal.min) return optimal.min - value
  if (value > optimal.max) return value - optimal.max
  return 0
}

/**
 * The seed distribution blended with what this device has actually observed.
 *
 * Returns weights summing to 100 — a percent-scale crowd, so one player is one
 * unit of it however the seed was authored. An all-zero result means there is
 * genuinely nothing to draw.
 *
 * The seed defines the shape of the variable, so its length wins: a `local`
 * array of a different length is a stale record from a previous version of the
 * call and only the buckets that still exist are read.
 */
export function blendCrowd(seed: number[], local: number[]): number[] {
  const length = seed.length || local.length
  if (length === 0) return []

  const s = sanitize(seed, length)
  const l = sanitize(local, length)
  const seedTotal = total(s)

  // Rescale the authored shape to exactly SEED_WEIGHT of mass. Observations are
  // then added at face value, which is the entire decay mechanism: the seed's
  // contribution is fixed while theirs keeps growing.
  // A call with no authored seed is legal but rare — fall through to whatever
  // was observed rather than inventing a uniform crowd nobody belongs to.
  const scale = seedTotal > 0 ? SEED_WEIGHT / seedTotal : 0

  const blended = new Array<number>(length).fill(0)
  for (let i = 0; i < length; i++) blended[i] = s[i] * scale + l[i]

  const mass = total(blended)
  if (mass <= 0) return blended

  for (let i = 0; i < length; i++) blended[i] = (blended[i] / mass) * CROWD_SCALE
  return blended
}

/**
 * What share of players did worse than this answer, 0-100.
 *
 * "Worse" is distance from the optimal play, not a higher or a lower number:
 * overshooting by two steps and falling short by two steps are the same quality
 * of call, and a percentile that preferred one would be scoring direction
 * instead of judgement. Pass the whole band where several answers are right, not
 * its midpoint, or every correct answer but the middle one is scored as a miss.
 *
 * Ties take the midpoint — half of everyone who played exactly as well as you
 * counts as behind you — and the player is always added to their own tie group.
 * That second part is what keeps the figure honest on a sparse distribution:
 * you cannot be better than 100% of players when you are one of them, and a
 * bucket nobody else picked cannot hand you a clean sweep.
 */
export function percentileOf(
  value: number,
  distribution: number[],
  min: number,
  step: number,
  optimal: Optimal,
): number {
  const mine = distanceTo(value, optimal)
  // An unanswerable question gets the no-information answer, not NaN.
  if (!Number.isFinite(mine)) return 50

  let worse = 0
  let tied = 0
  let better = 0
  let mass = 0

  for (let i = 0; i < distribution.length; i++) {
    const w = distribution[i]
    if (!Number.isFinite(w) || w <= 0) continue

    mass += w
    const d = distanceTo(valueAt(i, min, step), optimal)
    if (d > mine + FUZZ) worse += w
    else if (d < mine - FUZZ) better += w
    else tied += w
  }

  // No crowd at all: the no-information answer rather than a division by zero.
  if (mass <= 0) return 50

  // The player themselves, at one percent of whatever crowd was handed over.
  // Sized off the mass rather than fixed at 1 so the figure cannot move just
  // because a caller passed raw authored weights instead of blendCrowd's
  // hundred-unit output — a percentile that changed when every weight was
  // doubled would be reporting the scale of the array, not the player's call.
  tied += mass / CROWD_SCALE

  const pct = ((worse + tied / 2) / (worse + tied + better)) * 100
  return Math.min(MAX_PERCENTILE, Math.max(MIN_PERCENTILE, pct))
}

/**
 * Where the crowd piled up — the value of the heaviest bucket.
 *
 * Returns NaN when there is nothing to summarise. A distribution with no mass
 * has no mode, and returning `min` would paint a phantom spike on the left edge
 * of every empty chart. Ties go to the lower value so the answer is stable as
 * observations arrive, rather than flicking between two equal peaks.
 */
export function modeOf(distribution: number[], min: number, step: number): number {
  let best = -Infinity
  let index = -1

  for (let i = 0; i < distribution.length; i++) {
    const w = distribution[i]
    if (!Number.isFinite(w) || w <= 0) continue
    if (w > best) {
      best = w
      index = i
    }
  }

  return index < 0 ? NaN : valueAt(index, min, step)
}
