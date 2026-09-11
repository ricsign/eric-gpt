/**
 * Comparing a finger-drawn prediction against the truth.
 *
 * Pure and tested, because the number this produces — "you guessed 61% low" — is
 * both the app's central pedagogical claim and the headline on every share card.
 * It has to be right, and it has to be stable: the same stroke must always produce
 * the same percentage.
 */

export interface StrokePoint {
  /** 0..1 across the domain, strictly non-decreasing. */
  x: number
  /** 0..1 up the range, where 1 is the top of the axis. */
  y: number
}

export interface CurveComparison {
  predictedEnd: number
  actualEnd: number
  /** Signed relative error at the endpoint. Negative means the guess was low. */
  endpointError: number
  /** Mean absolute relative error across the domain — measures shape, not just the end. */
  shapeError: number
}

/**
 * Resamples a stroke onto an even grid of `steps + 1` points.
 *
 * A finger produces points at whatever rate the browser fires pointermove, which
 * varies with device, speed and load. Resampling makes the comparison independent
 * of how fast the line was drawn.
 */
export function resampleStroke(stroke: StrokePoint[], steps: number): number[] {
  if (stroke.length === 0 || steps < 1) return []
  if (stroke.length === 1) return Array.from({ length: steps + 1 }, () => stroke[0].y)

  const out: number[] = []
  let cursor = 0

  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    while (cursor < stroke.length - 2 && stroke[cursor + 1].x < t) cursor++

    const a = stroke[cursor]
    const b = stroke[cursor + 1]
    const span = b.x - a.x
    // Zero-width segments happen when the finger pauses; hold the value rather
    // than dividing by zero.
    const f = span > 0 ? Math.min(1, Math.max(0, (t - a.x) / span)) : 0
    out.push(a.y + (b.y - a.y) * f)
  }
  return out
}

/**
 * Scores a stroke against the truth.
 *
 * The shape error deliberately skips the first 15% of the domain: every curve in
 * this app starts at or near zero, where a relative error is arbitrarily large
 * and says nothing about whether the person understood the shape.
 */
export function compareCurve(
  stroke: StrokePoint[],
  truth: number[],
  yMax: number,
): CurveComparison | null {
  if (truth.length < 2 || stroke.length < 2 || yMax <= 0) return null

  const steps = truth.length - 1
  const drawn = resampleStroke(stroke, steps).map((n) => n * yMax)
  if (drawn.length !== truth.length) return null

  const predictedEnd = drawn[steps]
  const actualEnd = truth[steps]

  const from = Math.floor(steps * 0.15)
  let acc = 0
  let n = 0
  for (let i = from; i <= steps; i++) {
    if (truth[i] <= 0) continue
    acc += Math.abs(drawn[i] - truth[i]) / truth[i]
    n++
  }

  return {
    predictedEnd,
    actualEnd,
    endpointError: actualEnd > 0 ? (predictedEnd - actualEnd) / actualEnd : 0,
    shapeError: n > 0 ? acc / n : 0,
  }
}

/**
 * The share headline. Kept here rather than in the component so the exact wording
 * that ends up burned into an image is covered by a test.
 */
export function describeError(endpointError: number): string {
  const pct = Math.round(Math.abs(endpointError) * 100)
  if (pct <= 8) return 'I got it almost exactly right.'
  return `I guessed ${pct}% ${endpointError < 0 ? 'low' : 'high'}.`
}
