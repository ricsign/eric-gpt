import { describe, expect, it } from 'vitest'
import { EPOCH, ROLLOVER_HOUR, callIndex, callNumber, compoundDay, formatCountdown, msUntilNextCall, nextQuestionId, unlockedToday } from './schedule'

/** Local-time Date, so these tests exercise the same path the app does. */
const at = (y: number, m: number, d: number, h = 12, min = 0) =>
  new Date(y, m - 1, d, h, min, 0, 0)

describe('compoundDay', () => {
  it('is the calendar day after the rollover hour', () => {
    expect(compoundDay(at(2026, 9, 13, 6, 0))).toBe('2026-09-13')
    expect(compoundDay(at(2026, 9, 13, 23, 59))).toBe('2026-09-13')
  })

  it('is still yesterday before the rollover hour', () => {
    // The whole point: someone playing at 00:30 is finishing their evening.
    expect(compoundDay(at(2026, 9, 13, 0, 30))).toBe('2026-09-12')
    expect(compoundDay(at(2026, 9, 13, 5, 59))).toBe('2026-09-12')
  })

  it('turns over exactly at the rollover hour, not a minute either side', () => {
    expect(compoundDay(at(2026, 9, 13, ROLLOVER_HOUR - 1, 59))).toBe('2026-09-12')
    expect(compoundDay(at(2026, 9, 13, ROLLOVER_HOUR, 0))).toBe('2026-09-13')
  })

  it('walks back across a month boundary', () => {
    expect(compoundDay(at(2026, 10, 1, 2, 0))).toBe('2026-09-30')
  })

  it('walks back across a year boundary', () => {
    expect(compoundDay(at(2027, 1, 1, 3, 0))).toBe('2026-12-31')
  })
})

describe('callNumber', () => {
  it('the epoch is call 1', () => {
    expect(callNumber(EPOCH)).toBe(1)
  })

  it('advances by exactly one per day', () => {
    expect(callNumber('2026-09-02')).toBe(2)
    expect(callNumber('2026-10-01')).toBe(31)
  })

  it('crosses a leap day correctly', () => {
    expect(callNumber('2028-03-01') - callNumber('2028-02-29')).toBe(1)
  })

  it('never repeats a number, so No.142 is stable forever', () => {
    const seen = new Set<number>()
    const d = new Date(2026, 8, 1)
    for (let i = 0; i < 400; i++) {
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
        d.getDate(),
      ).padStart(2, '0')}`
      seen.add(callNumber(key))
      d.setDate(d.getDate() + 1)
    }
    expect(seen.size).toBe(400)
  })
})

describe('callIndex', () => {
  it('plays the library in authored order on the first pass', () => {
    // Calls 1-10 are a deliberate onboarding sequence; shuffling them would
    // hand a brand-new player the hardest call first.
    for (let n = 1; n <= 10; n++) expect(callIndex(n, 10)).toBe(n - 1)
  })

  it('stays inside the library for any call number', () => {
    for (let n = 1; n < 2000; n++) {
      const i = callIndex(n, 10)
      expect(i).toBeGreaterThanOrEqual(0)
      expect(i).toBeLessThan(10)
    }
  })

  it('does not simply repeat the authored order on later cycles', () => {
    const first = Array.from({ length: 10 }, (_, i) => callIndex(i + 1, 10))
    const second = Array.from({ length: 10 }, (_, i) => callIndex(i + 11, 10))
    expect(second).not.toEqual(first)
  })

  it('visits every call within a cycle rather than favouring a few', () => {
    const seen = new Set(Array.from({ length: 10 }, (_, i) => callIndex(i + 11, 10)))
    expect(seen.size).toBe(10)
  })

  it('survives an empty library instead of dividing by zero', () => {
    expect(callIndex(5, 0)).toBe(0)
  })
})

describe('the countdown', () => {
  it('counts to the next rollover, not to midnight', () => {
    const ms = msUntilNextCall(at(2026, 9, 13, 22, 0))
    expect(ms / 3_600_000).toBeCloseTo(8, 5)
  })

  it('counts to this morning when the player is up before rollover', () => {
    const ms = msUntilNextCall(at(2026, 9, 13, 2, 0))
    expect(ms / 3_600_000).toBeCloseTo(4, 5)
  })

  it('is never negative', () => {
    for (const h of [0, 5, 6, 7, 23]) {
      expect(msUntilNextCall(at(2026, 9, 13, h, 30))).toBeGreaterThan(0)
    }
  })

  it('formats as three fixed segments so it never reflows', () => {
    expect(formatCountdown(0)).toBe('00:00:00')
    expect(formatCountdown(3_600_000)).toBe('01:00:00')
    expect(formatCountdown(8 * 3_600_000 + 7 * 60_000 + 5_000)).toBe('08:07:05')
    expect(formatCountdown(-5000)).toBe('00:00:00')
  })
})

describe('the queue follows the player, not the calendar', () => {
  const IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  it('starts everyone at question one, whenever they arrive', () => {
    // The bug this replaces: callIndex returned authored order only for the
    // ten days after EPOCH, so from 2026-09-11 a brand-new player's first
    // screen was whatever the shuffling stride landed on. Arrival date must
    // not decide which question you meet first.
    expect(nextQuestionId(IDS, new Set())).toBe(1)
  })

  it('walks the set in authored order as answers land', () => {
    expect(nextQuestionId(IDS, new Set([1]))).toBe(2)
    expect(nextQuestionId(IDS, new Set([1, 2, 3]))).toBe(4)
    // A gap is filled before moving on: the order is the curriculum.
    expect(nextQuestionId(IDS, new Set([1, 3, 4]))).toBe(2)
  })

  it('returns null once the set is finished, because ten is all there is', () => {
    expect(nextQuestionId(IDS, new Set(IDS))).toBe(null)
  })

  it('spends one day per answer, and practice never spends one', () => {
    const today = compoundDay(new Date('2026-09-14T12:00:00'))
    expect(unlockedToday([], new Date('2026-09-14T12:00:00'))).toBe(true)
    expect(unlockedToday([today], new Date('2026-09-14T12:00:00'))).toBe(false)
    // Rollover is 6am, so late the same evening it is still spent...
    expect(unlockedToday([today], new Date('2026-09-14T23:30:00'))).toBe(false)
    // ...and still spent just before the rollover next morning.
    expect(unlockedToday([today], new Date('2026-09-15T05:30:00'))).toBe(false)
    expect(unlockedToday([today], new Date('2026-09-15T06:30:00'))).toBe(true)
  })
})
