import { describe, expect, it } from 'vitest'
import { drillIndex, drillNumber, glyph, MAX_ATTEMPTS, shareText, EPOCH } from './daily'

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

describe('the share string', () => {
  it('reports the attempt it was solved on', () => {
    const text = shareText({ number: 212, attempts: [false, true], solved: true, streak: 1 })
    expect(text).toContain('#212')
    expect(text).toContain(`2/${MAX_ATTEMPTS}`)
  })

  it('marks an unsolved drill with X', () => {
    const text = shareText({
      number: 9,
      attempts: [false, false, false],
      solved: false,
      streak: 0,
    })
    expect(text).toContain(`X/${MAX_ATTEMPTS}`)
  })

  it('shows the streak only once it is worth showing', () => {
    const one = shareText({ number: 1, attempts: [true], solved: true, streak: 1 })
    const many = shareText({ number: 1, attempts: [true], solved: true, streak: 14 })
    expect(one).not.toContain('🔥')
    expect(many).toContain('14🔥')
  })

  it('carries a bare domain and no tracking link', () => {
    const text = shareText({ number: 5, attempts: [true], solved: true, streak: 3 })
    expect(text).toContain('compound.money')
    expect(text).not.toContain('http')
    expect(text).not.toContain('utm')
    expect(text).not.toContain('?')
  })

  it('discloses nothing about the user or the answer', () => {
    const text = shareText({ number: 5, attempts: [false, false, true], solved: true, streak: 3 })
    // Spoiler-free is the whole design: squares say how many tries, never which
    // option was chosen. And there is no dollar figure anywhere.
    expect(text).not.toMatch(/\$/)
    expect(glyph({ number: 5, attempts: [false, false, true], solved: true, streak: 3 })).toBe(
      '⬛⬛🟩',
    )
  })

  it('puts the domain on its own line', () => {
    const lines = shareText({ number: 5, attempts: [true], solved: true, streak: 0 }).split('\n')
    expect(lines).toHaveLength(3)
    expect(lines[2]).toBe('compound.money')
  })
})
