import { describe, expect, it } from 'vitest'
import { drillIndex, drillNumber, EPOCH } from './daily'

describe('drillNumber', () => {
  it('the epoch is puzzle 1', () => {
    expect(drillNumber(EPOCH)).toBe(1)
  })

  it('advances by exactly one per day', () => {
    expect(drillNumber('2026-01-02')).toBe(2)
    expect(drillNumber('2026-01-31')).toBe(31)
    expect(drillNumber('2026-02-01')).toBe(32)
  })

  it('crosses a leap day correctly', () => {
    // 2028 is a leap year; Mar 1 must be one past Feb 29.
    expect(drillNumber('2028-03-01') - drillNumber('2028-02-29')).toBe(1)
  })

  it('is stable across a year boundary', () => {
    expect(drillNumber('2027-01-01') - drillNumber('2026-12-31')).toBe(1)
  })
})

describe('drillIndex', () => {
  it('always lands inside the bank', () => {
    for (let n = 1; n < 500; n++) {
      const i = drillIndex(n, 37)
      expect(i).toBeGreaterThanOrEqual(0)
      expect(i).toBeLessThan(37)
    }
  })

  it('visits every question exactly once per cycle', () => {
    // The property that matters: no question repeats until the bank is exhausted.
    const size = 40
    const seen = new Set<number>()
    for (let n = 0; n < size; n++) seen.add(drillIndex(n, size))
    expect(seen.size).toBe(size)
  })

  it('does not put the same question on the same weekday', () => {
    const size = 40
    expect(drillIndex(1, size)).not.toBe(drillIndex(8, size))
  })

  it('is deterministic — the same day is the same question, always', () => {
    expect(drillIndex(212, 40)).toBe(drillIndex(212, 40))
  })

  it('survives an empty bank rather than dividing by zero', () => {
    expect(drillIndex(5, 0)).toBe(0)
  })
})
