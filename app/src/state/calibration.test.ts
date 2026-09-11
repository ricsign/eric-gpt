import { describe, expect, it } from 'vitest'
import { calibration, type Prediction } from './store'

const p = (error: number, day = '2026-09-11'): Prediction => ({ concept: 'egb', error, day })

describe('calibration', () => {
  it('reports nothing before the first prediction', () => {
    const c = calibration([])
    expect(c.median).toBeNull()
    expect(c.count).toBe(0)
    expect(c.improving).toBe(false)
  })

  it('is the median, not the mean', () => {
    // One wild first guess must not dominate the number for weeks, which is
    // exactly what a mean would let it do.
    const withOutlier = calibration([p(0.1), p(0.2), p(0.3), p(9.0)])
    expect(withOutlier.median).toBeCloseTo(0.25, 6)

    const mean = (0.1 + 0.2 + 0.3 + 9.0) / 4
    expect(withOutlier.median).toBeLessThan(mean)
  })

  it('handles an odd count', () => {
    expect(calibration([p(0.5), p(0.1), p(0.3)]).median).toBeCloseTo(0.3, 6)
  })

  it('counts every prediction', () => {
    expect(calibration([p(0.1), p(0.2)]).count).toBe(2)
  })

  it('detects improvement once there is enough history', () => {
    // Early guesses badly wrong, recent ones close.
    const improving = calibration([p(0.9), p(0.8), p(0.85), p(0.7), p(0.1), p(0.12), p(0.08)])
    expect(improving.improving).toBe(true)
    expect(improving.recent!).toBeLessThan(improving.median!)
  })

  it('does not claim improvement from a short history', () => {
    // Three good guesses is luck, not calibration. Claiming otherwise would make
    // the one honest metric in the app dishonest.
    expect(calibration([p(0.9), p(0.1), p(0.1)]).improving).toBe(false)
  })

  it('does not claim improvement when accuracy is getting worse', () => {
    expect(
      calibration([p(0.05), p(0.08), p(0.1), p(0.6), p(0.7), p(0.8), p(0.9)]).improving,
    ).toBe(false)
  })

  it('a perfect run reports zero error', () => {
    expect(calibration([p(0), p(0), p(0)]).median).toBe(0)
  })
})
