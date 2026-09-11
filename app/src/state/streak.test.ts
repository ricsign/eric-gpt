import { describe, expect, it } from 'vitest'
import { advanceStreak, type Streak } from './store'

const fresh: Streak = { current: 0, longest: 0, lastDay: null, freezes: 2 }

describe('advanceStreak', () => {
  it('starts a streak on the first session', () => {
    const s = advanceStreak(fresh, '2026-09-11')
    expect(s.current).toBe(1)
    expect(s.longest).toBe(1)
    expect(s.lastDay).toBe('2026-09-11')
  })

  it('a second session the same day changes nothing', () => {
    const one = advanceStreak(fresh, '2026-09-11')
    expect(advanceStreak(one, '2026-09-11')).toBe(one)
  })

  it('consecutive days increment', () => {
    let s = advanceStreak(fresh, '2026-09-11')
    s = advanceStreak(s, '2026-09-12')
    s = advanceStreak(s, '2026-09-13')
    expect(s.current).toBe(3)
    expect(s.longest).toBe(3)
  })

  it('one missed day is forgiven, spending a freeze', () => {
    let s = advanceStreak(fresh, '2026-09-11')
    s = advanceStreak(s, '2026-09-12')
    const before = s.freezes
    // Skips the 13th entirely.
    s = advanceStreak(s, '2026-09-14')
    expect(s.current).toBe(3)
    expect(s.freezes).toBe(before - 1)
  })

  it('two missed days resets, even with freezes left', () => {
    let s = advanceStreak(fresh, '2026-09-11')
    s = advanceStreak(s, '2026-09-12')
    s = advanceStreak(s, '2026-09-16')
    expect(s.current).toBe(1)
    expect(s.freezes).toBe(2)
  })

  it('resets once the freezes run out', () => {
    let s: Streak = { current: 5, longest: 5, lastDay: '2026-09-01', freezes: 0 }
    s = advanceStreak(s, '2026-09-03')
    expect(s.current).toBe(1)
  })

  it('keeps the longest streak after a reset', () => {
    let s: Streak = { current: 21, longest: 21, lastDay: '2026-09-01', freezes: 0 }
    s = advanceStreak(s, '2026-09-20')
    expect(s.current).toBe(1)
    expect(s.longest).toBe(21)
  })

  it('earns a freeze back every seventh consecutive day, capped at two', () => {
    let s: Streak = { current: 5, longest: 5, lastDay: '2026-09-01', freezes: 0 }
    s = advanceStreak(s, '2026-09-02') // 6
    expect(s.freezes).toBe(0)
    s = advanceStreak(s, '2026-09-03') // 7 -> earns one
    expect(s.current).toBe(7)
    expect(s.freezes).toBe(1)

    // Run to 14 and confirm the cap holds rather than accumulating.
    let day = 3
    while (s.current < 21) {
      day++
      s = advanceStreak(s, `2026-09-${String(day).padStart(2, '0')}`)
    }
    expect(s.freezes).toBeLessThanOrEqual(2)
  })

  it('ignores a clock that has gone backwards', () => {
    const s: Streak = { current: 4, longest: 4, lastDay: '2026-09-11', freezes: 2 }
    // A device-time edit or a timezone jump must not corrupt the count.
    expect(advanceStreak(s, '2026-09-09')).toBe(s)
  })
})
