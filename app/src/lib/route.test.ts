import { describe, expect, it } from 'vitest'
import { callNoFromPath, pathForCall, urlForCall } from './route'

describe('callNoFromPath', () => {
  it('reads a bare call number', () => {
    expect(callNoFromPath('/142')).toBe(142)
    expect(callNoFromPath('/1')).toBe(1)
  })

  it('tolerates a trailing slash', () => {
    expect(callNoFromPath('/142/')).toBe(142)
  })

  it('rejects everything that is not a call', () => {
    // The root is today's call, not a deep link, and must not resolve to one.
    for (const p of ['/', '', '/abc', '/142a', '/0', '/-3', '/1.5', '/142/extra', '/tab']) {
      expect(callNoFromPath(p), p).toBeNull()
    }
  })

  it('rejects a number too large to be a real call', () => {
    // Guards against a crafted path forcing a huge index computation.
    expect(callNoFromPath('/99999999999')).toBeNull()
  })

  it('round-trips with pathForCall', () => {
    for (const n of [1, 7, 142, 9999]) {
      expect(callNoFromPath(pathForCall(n))).toBe(n)
    }
  })
})

describe('urlForCall', () => {
  it('builds an absolute url with no query string or tracking', () => {
    const url = urlForCall(142, 'https://compound.day')
    expect(url).toBe('https://compound.day/142')
    expect(url).not.toContain('?')
    expect(url).not.toContain('utm')
  })
})
