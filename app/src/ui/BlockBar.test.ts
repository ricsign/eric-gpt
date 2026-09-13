import { describe, expect, it } from 'vitest'
import {
  blockState,
  positionCount,
  positionToValue,
  stepValue,
  valueToFraction,
} from './BlockBar'

/** Fractions dense enough to hit every block boundary and every rounding edge. */
const SWEEP = Array.from({ length: 1001 }, (_, i) => i / 1000)

/** Digits after the point in a number's own string form. */
const decimals = (n: number) => {
  const s = String(n)
  const dot = s.indexOf('.')
  return dot < 0 ? 0 : s.length - dot - 1
}

/** The real variables this control ships against, plus the awkward cases. */
const RANGES: { min: number; max: number; step: number; name: string }[] = [
  { min: 0, max: 15, step: 1, name: '401k percent' },
  { min: 0, max: 12, step: 0.5, name: 'months of expenses' },
  { min: 0, max: 3, step: 0.1, name: 'expense ratio' },
  { min: 25, max: 70, step: 1, name: 'retirement age' },
  { min: 0, max: 2000, step: 25, name: 'monthly payment' },
  { min: 0, max: 10, step: 3, name: 'step does not divide the range' },
  { min: 300, max: 850, step: 7, name: 'credit score, uneven step' },
  { min: -5000, max: 5000, step: 250, name: 'signed range' },
]

describe('positionToValue', () => {
  it('clamps every fraction, in range or not, into [min,max]', () => {
    for (const r of RANGES) {
      for (const f of [...SWEEP, -1, -0.0001, 1.0001, 99, NaN, Infinity, -Infinity]) {
        const v = positionToValue(f, r.min, r.max, r.step)
        expect(Number.isFinite(v), `${r.name} @ ${f}`).toBe(true)
        expect(v, `${r.name} @ ${f}`).toBeGreaterThanOrEqual(r.min)
        expect(v, `${r.name} @ ${f}`).toBeLessThanOrEqual(r.max)
      }
    }
  })

  it('lands on a step boundary, with no float dust in the rendered number', () => {
    for (const r of RANGES) {
      for (const f of SWEEP) {
        const v = positionToValue(f, r.min, r.max, r.step)
        if (v === r.max) continue // the tail position need not be on a step

        const k = Math.round((v - r.min) / r.step)
        expect(Math.abs(v - (r.min + k * r.step)), `${r.name} @ ${f} -> ${v}`).toBeLessThan(1e-9)
        // The value is printed straight into the 64px readout and used as an
        // object key in the crowd histogram, so 0.30000000000000004 is a bug
        // even though it is within a rounding error of 0.3.
        expect(decimals(v), `${r.name} @ ${f} -> ${v}`).toBeLessThanOrEqual(decimals(r.step))
      }
    }
  })

  it('is monotonic across the bar', () => {
    for (const r of RANGES) {
      let prev = -Infinity
      for (const f of SWEEP) {
        const v = positionToValue(f, r.min, r.max, r.step)
        expect(v, r.name).toBeGreaterThanOrEqual(prev)
        prev = v
      }
    }
  })

  it('reaches both ends', () => {
    for (const r of RANGES) {
      expect(positionToValue(0, r.min, r.max, r.step), r.name).toBe(r.min)
      expect(positionToValue(1, r.min, r.max, r.step), r.name).toBe(r.max)
    }
  })

  it('round-trips through a fraction unchanged — the anti-jitter invariant', () => {
    // A value that moved when it was re-projected and re-read would oscillate
    // under a finger held still, because every frame re-derives it.
    for (const r of RANGES) {
      for (const f of SWEEP) {
        const v = positionToValue(f, r.min, r.max, r.step)
        const back = positionToValue(valueToFraction(v, r.min, r.max), r.min, r.max, r.step)
        expect(back, `${r.name} @ ${f}`).toBe(v)
      }
    }
  })

  it('is idempotent under repeated projection', () => {
    for (const r of RANGES) {
      const v = positionToValue(0.3737, r.min, r.max, r.step)
      let x = v
      for (let i = 0; i < 25; i++) {
        x = positionToValue(valueToFraction(x, r.min, r.max), r.min, r.max, r.step)
      }
      expect(x, r.name).toBe(v)
    }
  })

  it('quantizes to the nearest position, not the one below', () => {
    // 0-15 in 1s: the midpoint between 7 and 8 is 7.5/15 = 0.5.
    expect(positionToValue(0.49, 0, 15, 1)).toBe(7)
    expect(positionToValue(0.51, 0, 15, 1)).toBe(8)
    expect(positionToValue(0.4, 0, 10, 1)).toBe(4)
  })

  it('gives an uneven max its own position without inventing an off-lattice value', () => {
    // 0-10 by 3 => 0,3,6,9,10. 10 is reachable; 9.5 and 11 are not.
    const seen = new Set(SWEEP.map((f) => positionToValue(f, 0, 10, 3)))
    expect([...seen].sort((a, b) => a - b)).toEqual([0, 3, 6, 9, 10])
  })

  it('survives a degenerate range or step instead of returning NaN', () => {
    expect(positionToValue(0.5, 5, 5, 1)).toBe(5)
    expect(positionToValue(0.5, 10, 0, 1)).toBe(10)
    for (const bad of [0, -1, NaN, Infinity]) {
      const v = positionToValue(0.5, 0, 10, bad)
      expect(Number.isFinite(v)).toBe(true)
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(10)
    }
  })

  it('holds every invariant on a range that divides only in decimal', () => {
    // 0.1 steps are exact in decimal and never in binary; this is where a naive
    // (value - min) / step accumulates error until the top step disappears.
    const values = SWEEP.map((f) => positionToValue(f, 0, 3, 0.1))
    expect(values[0]).toBe(0)
    expect(values[values.length - 1]).toBe(3)
    expect(new Set(values).size).toBe(31)
    expect(values).toContain(0.3)
    expect(values).toContain(2.9)
  })
})

describe('valueToFraction', () => {
  it('maps the ends to 0 and 1', () => {
    expect(valueToFraction(0, 0, 15)).toBe(0)
    expect(valueToFraction(15, 0, 15)).toBe(1)
    expect(valueToFraction(300, 300, 850)).toBe(0)
    expect(valueToFraction(850, 300, 850)).toBe(1)
  })

  it('is linear in between and handles a signed range', () => {
    expect(valueToFraction(7.5, 0, 15)).toBe(0.5)
    expect(valueToFraction(0, -5000, 5000)).toBe(0.5)
    expect(valueToFraction(-2500, -5000, 5000)).toBe(0.25)
  })

  it('clamps out-of-range values rather than overflowing the bar', () => {
    expect(valueToFraction(-3, 0, 15)).toBe(0)
    expect(valueToFraction(40, 0, 15)).toBe(1)
  })

  it('returns 0 for a degenerate range or a non-finite value', () => {
    expect(valueToFraction(5, 5, 5)).toBe(0)
    expect(valueToFraction(5, 10, 0)).toBe(0)
    expect(valueToFraction(NaN, 0, 15)).toBe(0)
  })
})

describe('blockState', () => {
  it('is empty at min and filled at max, for any block count', () => {
    for (const n of [1, 3, 15, 40]) {
      for (let i = 0; i < n; i++) {
        expect(blockState(i, n, 0, 0, 15), `empty ${i}/${n}`).toBe('empty')
        expect(blockState(i, n, 15, 0, 15), `filled ${i}/${n}`).toBe('filled')
      }
    }
  })

  it('fills one block per step when blocks match the range', () => {
    // The employer-match call: 15 blocks, 0-15%, so block k is the k-th point.
    for (let v = 0; v <= 15; v++) {
      const filled = Array.from({ length: 15 }, (_, i) => blockState(i, 15, v, 0, 15)).filter(
        (s) => s === 'filled',
      ).length
      expect(filled, `value ${v}`).toBe(v)
    }
  })

  it('never un-fills a block as the value rises', () => {
    for (const r of RANGES) {
      for (let i = 0; i < 15; i++) {
        let prev = 'empty'
        for (const f of SWEEP) {
          const s = blockState(i, 15, positionToValue(f, r.min, r.max, r.step), r.min, r.max)
          if (prev === 'filled') expect(s, `${r.name} block ${i} @ ${f}`).toBe('filled')
          prev = s
        }
      }
    }
  })

  it('fills a contiguous run from the left, never a gap', () => {
    for (const f of SWEEP) {
      const states = Array.from({ length: 15 }, (_, i) =>
        blockState(i, 15, positionToValue(f, 0, 2000, 25), 0, 2000),
      )
      const firstEmpty = states.indexOf('empty')
      if (firstEmpty >= 0) {
        expect(states.slice(firstEmpty).every((s) => s === 'empty'), `@ ${f}`).toBe(true)
      }
    }
  })

  it('treats an out-of-bounds index as empty', () => {
    expect(blockState(-1, 15, 15, 0, 15)).toBe('empty')
    expect(blockState(15, 15, 15, 0, 15)).toBe('empty')
    expect(blockState(0, 0, 15, 0, 15)).toBe('empty')
  })
})

describe('positionCount', () => {
  it('counts both ends', () => {
    expect(positionCount(0, 15, 1)).toBe(16)
    expect(positionCount(0, 12, 0.5)).toBe(25)
    expect(positionCount(0, 3, 0.1)).toBe(31)
  })

  it('adds one slot for a max that no step lands on', () => {
    // 0,3,6,9,10
    expect(positionCount(0, 10, 3)).toBe(5)
  })
})

describe('stepValue', () => {
  it('moves one position at a time and stops at both ends', () => {
    expect(stepValue(7, 1, 0, 15, 1)).toBe(8)
    expect(stepValue(7, -1, 0, 15, 1)).toBe(6)
    expect(stepValue(15, 1, 0, 15, 1)).toBe(15)
    expect(stepValue(0, -1, 0, 15, 1)).toBe(0)
  })

  it('keeps fractional steps clean', () => {
    expect(stepValue(0.2, 1, 0, 3, 0.1)).toBe(0.3)
    let v = 0
    for (let i = 0; i < 30; i++) v = stepValue(v, 1, 0, 3, 0.1)
    expect(v).toBe(3)
  })

  it('steps down from an uneven max onto the last real step, not past it', () => {
    // 0,3,6,9,10: arrowing off 10 must land on 9, not on 7.
    expect(stepValue(10, -1, 0, 10, 3)).toBe(9)
    expect(stepValue(9, 1, 0, 10, 3)).toBe(10)
    expect(stepValue(850, -1, 300, 850, 7)).toBe(846)
  })

  it('pages without ever leaving the range', () => {
    for (const r of RANGES) {
      const page = Math.max(1, Math.round((positionCount(r.min, r.max, r.step) - 1) / 10))
      for (const from of [r.min, r.max, positionToValue(0.5, r.min, r.max, r.step)]) {
        for (const d of [page, -page, 999, -999]) {
          const v = stepValue(from, d, r.min, r.max, r.step)
          expect(v, `${r.name} ${from}+${d}`).toBeGreaterThanOrEqual(r.min)
          expect(v, `${r.name} ${from}+${d}`).toBeLessThanOrEqual(r.max)
        }
      }
    }
  })

  it('agrees with the drag: a keyboard value is always a draggable value', () => {
    for (const r of RANGES) {
      const reachable = new Set(SWEEP.map((f) => positionToValue(f, r.min, r.max, r.step)))
      let v = r.min
      while (v < r.max) {
        v = stepValue(v, 1, r.min, r.max, r.step)
        expect(reachable.has(v), `${r.name} -> ${v}`).toBe(true)
      }
      expect(v).toBe(r.max)
    }
  })

  it('handles a non-finite current value by starting from min', () => {
    expect(stepValue(NaN, 1, 0, 15, 1)).toBe(1)
  })
})
