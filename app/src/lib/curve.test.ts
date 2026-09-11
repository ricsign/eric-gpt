import { describe, expect, it } from 'vitest'
import { compareCurve, describeError, resampleStroke, type StrokePoint } from './curve'

/** A straight line from (0,0) to (1,1), sampled coarsely like a real finger would. */
const diagonal: StrokePoint[] = [
  { x: 0, y: 0 },
  { x: 0.37, y: 0.37 },
  { x: 0.62, y: 0.62 },
  { x: 1, y: 1 },
]

describe('resampleStroke', () => {
  it('returns steps + 1 samples', () => {
    expect(resampleStroke(diagonal, 10)).toHaveLength(11)
  })

  it('interpolates a straight line back to a straight line', () => {
    // The resample must not distort the shape, regardless of where the original
    // points happened to land.
    const out = resampleStroke(diagonal, 20)
    out.forEach((v, i) => expect(v).toBeCloseTo(i / 20, 6))
  })

  it('is independent of how fast the line was drawn', () => {
    const sparse: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ]
    const dense: StrokePoint[] = Array.from({ length: 60 }, (_, i) => ({
      x: i / 59,
      y: i / 59,
    }))
    const a = resampleStroke(sparse, 24)
    const b = resampleStroke(dense, 24)
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 6))
  })

  it('holds the value through a pause, rather than dividing by zero', () => {
    const paused: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 0.5, y: 0.5 },
      { x: 0.5, y: 0.5 },
      { x: 1, y: 1 },
    ]
    expect(resampleStroke(paused, 10).every(Number.isFinite)).toBe(true)
  })

  it('survives a single point and an empty stroke', () => {
    expect(resampleStroke([{ x: 0, y: 0.4 }], 4)).toEqual([0.4, 0.4, 0.4, 0.4, 0.4])
    expect(resampleStroke([], 4)).toEqual([])
  })
})

describe('compareCurve', () => {
  /** An exponential truth, the shape the app exists to teach. */
  const exponential = Array.from({ length: 49 }, (_, i) => Math.pow(1.07, (i / 48) * 40) * 1000)
  const yMax = exponential[exponential.length - 1] * 1.15

  it('a perfect trace scores near zero error', () => {
    const perfect: StrokePoint[] = exponential.map((v, i) => ({
      x: i / 48,
      y: v / yMax,
    }))
    const r = compareCurve(perfect, exponential, yMax)!
    expect(Math.abs(r.endpointError)).toBeLessThan(0.001)
    expect(r.shapeError).toBeLessThan(0.001)
  })

  it('the canonical wrong answer — a line to the contributions total — reads as a large underestimate', () => {
    // This is the whole finding: the intuitive prediction is a straight line
    // ending near the sum of what you put in, which is far below where compound
    // growth actually lands.
    const contributionsEnd = exponential[exponential.length - 1] * 0.3
    const naive: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 1, y: contributionsEnd / yMax },
    ]
    const r = compareCurve(naive, exponential, yMax)!
    expect(r.endpointError).toBeLessThan(-0.5)
    expect(r.shapeError).toBeGreaterThan(0.2)
  })

  it('separates getting the endpoint right from getting the shape right', () => {
    // A straight line drawn to the correct final value still has the wrong shape:
    // it runs far above the truth for most of the domain. The shape score has to
    // catch that, or someone can guess the destination and learn nothing.
    const actualEnd = exponential[exponential.length - 1]
    const straightToTheRightPlace: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 1, y: actualEnd / yMax },
    ]
    const r = compareCurve(straightToTheRightPlace, exponential, yMax)!
    expect(Math.abs(r.endpointError)).toBeLessThan(0.001)
    expect(r.shapeError).toBeGreaterThan(0.5)
  })

  it('reports the sign correctly for an overestimate', () => {
    const tooHigh: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ]
    // Truth flat at half the axis; the stroke ends at the top.
    const flat = Array.from({ length: 49 }, () => 500)
    const r = compareCurve(tooHigh, flat, 1000)!
    expect(r.endpointError).toBeGreaterThan(0)
    expect(r.predictedEnd).toBeCloseTo(1000, 6)
    expect(r.actualEnd).toBe(500)
  })

  it('ignores the near-zero opening of the domain in the shape score', () => {
    // Every curve starts at zero, where a relative error is arbitrarily large and
    // says nothing about whether the shape was understood.
    const truth = Array.from({ length: 49 }, (_, i) => (i === 0 ? 0.0001 : i * 100))
    const traced: StrokePoint[] = truth.map((v, i) => ({ x: i / 48, y: v / 4800 }))
    const r = compareCurve(traced, truth, 4800)!
    expect(Number.isFinite(r.shapeError)).toBe(true)
    expect(r.shapeError).toBeLessThan(0.05)
  })

  it('refuses to score inputs it cannot score', () => {
    expect(compareCurve([], exponential, yMax)).toBeNull()
    expect(compareCurve(diagonal, [1], yMax)).toBeNull()
    expect(compareCurve(diagonal, exponential, 0)).toBeNull()
  })

  it('is deterministic — the same stroke always gives the same number', () => {
    const a = compareCurve(diagonal, exponential, yMax)!
    const b = compareCurve(diagonal, exponential, yMax)!
    expect(a).toEqual(b)
  })
})

describe('describeError', () => {
  it('states the direction, because low and high are different mistakes', () => {
    expect(describeError(-0.61)).toBe('I guessed 61% low.')
    expect(describeError(0.4)).toBe('I guessed 40% high.')
  })

  it('does not nitpick a close guess', () => {
    expect(describeError(0.03)).toContain('almost exactly right')
    expect(describeError(-0.08)).toContain('almost exactly right')
  })

  it('contains no dollar figure — the share headline must disclose nothing', () => {
    for (const e of [-0.9, -0.3, 0, 0.25, 1.4]) {
      expect(describeError(e)).not.toMatch(/\$/)
    }
  })
})
