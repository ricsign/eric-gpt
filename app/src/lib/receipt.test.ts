import { describe, expect, it } from 'vitest'
import {
  BARCODE_MAX_WIDTH,
  BARCODE_MIN_WIDTH,
  BARCODE_MODULES,
  MAX_LINES,
  TEAR_SEGMENTS,
  barcodeWidths,
  receiptCode,
  receiptLines,
  receiptSeed,
  receiptText,
  tearClipPath,
  tearPath,
  type ReceiptInput,
} from './receipt'

const SALARY = 62_000

const BASE: ReceiptInput = {
  callNo: 13,
  date: 'SEP 13',
  title: 'Your boss pays 50c for every dollar you save.',
  verdict: 'short',
  value: 3,
  unit: '%',
  breakdown: [
    // A compute function is free to put the player's pay on the receipt — it is
    // their own screen. The share text must still never carry it off the device.
    { label: 'PAY', value: '$62,000' },
    { label: 'YOUR COST', value: '$155/mo' },
    { label: 'MATCH CAPTURED', value: '$930' },
    { label: 'LEFT ON THE TABLE', value: '$930', emphasis: true },
    { label: 'INSTANT RETURN', value: '50%' },
    { label: 'YEARS TO 65', value: '35' },
    { label: 'ONE LINE TOO MANY', value: '—' },
  ],
  at65: 412_900,
  delta: -188_400,
}

const withPatch = (patch: Partial<ReceiptInput>): ReceiptInput => ({ ...BASE, ...patch })

describe('barcodeWidths', () => {
  it('is deterministic for one seed', () => {
    expect(barcodeWidths('13:3:412900')).toEqual(barcodeWidths('13:3:412900'))
  })

  it('produces the declared number of modules, all drawable', () => {
    const widths = barcodeWidths(receiptSeed(BASE))
    expect(widths).toHaveLength(BARCODE_MODULES)
    // A zero-width module would leave a gap in the band; anything wider than the
    // cap swallows its neighbours and the barcode reads as a solid block.
    for (const w of widths) {
      expect(Number.isInteger(w)).toBe(true)
      expect(w).toBeGreaterThanOrEqual(BARCODE_MIN_WIDTH)
      expect(w).toBeLessThanOrEqual(BARCODE_MAX_WIDTH)
    }
  })

  it('is not the same barcode for every player', () => {
    // The whole point of deriving bars from the result: if two different plays on
    // the same call produced identical bars, the barcode would be wallpaper.
    const a = barcodeWidths(receiptSeed(BASE))
    const b = barcodeWidths(receiptSeed(withPatch({ value: 6, at65: 601_300, delta: 0 })))
    expect(a).not.toEqual(b)
  })

  it('separates results that differ only in one field', () => {
    const seen = new Set<string>()
    for (const value of [0, 1, 2, 3, 4, 5, 6, 7, 8]) {
      seen.add(barcodeWidths(receiptSeed(withPatch({ value }))).join(''))
    }
    expect(seen.size).toBe(9)
  })

  it('uses the whole range rather than clustering on one width', () => {
    const widths = barcodeWidths(receiptSeed(BASE))
    expect(new Set(widths).size).toBeGreaterThan(2)
  })
})

describe('tearPath', () => {
  const seed = 'receipt/top'

  it('is deterministic for one seed', () => {
    expect(tearPath(seed, 100, TEAR_SEGMENTS)).toEqual(tearPath(seed, 100, TEAR_SEGMENTS))
  })

  it('returns one more point than segments, spanning the full width', () => {
    const pts = tearPath(seed, 360, 15)
    expect(pts).toHaveLength(16)
    expect(pts[0].x).toBe(0)
    expect(pts[pts.length - 1].x).toBe(360)
  })

  it('stays inside the box it is clipping', () => {
    for (const s of ['a', 'b', 'c', 'd', 'e']) {
      for (const p of tearPath(s, 1080, 15)) {
        expect(p.x).toBeGreaterThanOrEqual(0)
        expect(p.x).toBeLessThanOrEqual(1080)
        expect(p.y).toBeGreaterThanOrEqual(0)
        expect(p.y).toBeLessThanOrEqual(1)
      }
    }
  })

  it('never doubles back, so the polygon cannot self-intersect', () => {
    // A non-monotonic x would fold the clip path into a bowtie and punch a hole
    // through the card — visible instantly, and only on some seeds.
    for (const s of ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) {
      const pts = tearPath(s, 1080, 15)
      for (let i = 1; i < pts.length; i++) {
        expect(pts[i].x).toBeGreaterThan(pts[i - 1].x)
      }
    }
  })

  it('tears differently for different seeds', () => {
    expect(tearPath('a', 100, 15)).not.toEqual(tearPath('b', 100, 15))
  })

  it('is not a zigzag — depth varies rather than alternating', () => {
    const ys = tearPath('irregular', 100, 40).map((p) => p.y)
    const shallow = ys.filter((y) => y < 0.3).length
    const deep = ys.filter((y) => y > 0.6).length
    expect(shallow).toBeGreaterThan(0)
    expect(deep).toBeGreaterThan(0)
    // Most of a torn edge is shallow fibre; a uniform distribution reads as teeth.
    expect(shallow).toBeGreaterThan(deep)
  })

  it('scales x with width and leaves y normalised', () => {
    const small = tearPath('s', 100, 12)
    const large = tearPath('s', 1000, 12)
    small.forEach((p, i) => {
      expect(large[i].x).toBeCloseTo(p.x * 10, 6)
      expect(large[i].y).toBe(p.y)
    })
  })
})

describe('tearClipPath', () => {
  it('closes a polygon with both edges torn from the same receipt seed', () => {
    const css = tearClipPath(receiptSeed(BASE), 11)
    expect(css.startsWith('polygon(')).toBe(true)
    expect(css.split(',')).toHaveLength((TEAR_SEGMENTS + 1) * 2)
    // The bottom edge is measured from the far side so the card can be any height.
    expect(css).toContain('calc(100% - ')
    expect(css).toBe(tearClipPath(receiptSeed(BASE), 11))
  })

  it('gives a different card a different tear', () => {
    expect(tearClipPath(receiptSeed(BASE), 11)).not.toBe(
      tearClipPath(receiptSeed(withPatch({ callNo: 14 })), 11),
    )
  })
})

describe('receiptLines', () => {
  it('opens with the play and never overflows the card', () => {
    const lines = receiptLines(BASE)
    expect(lines).toHaveLength(MAX_LINES)
    expect(lines[0]).toEqual({ label: 'YOUR PLAY', value: '3%' })
  })

  it('formats a money variable as money rather than a bare number', () => {
    const lines = receiptLines(withPatch({ unit: '', value: 350 }))
    expect(lines[0].value).toBe('$350')
  })

  it('does not print the play twice when a compute function already supplied it', () => {
    const lines = receiptLines(
      withPatch({ breakdown: [{ label: 'Your play', value: '3%' }, { label: 'PAY', value: '$1' }] }),
    )
    expect(lines.filter((l) => l.label.toUpperCase() === 'YOUR PLAY')).toHaveLength(1)
  })
})

describe('receiptText', () => {
  const EMOJI =
    /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}\u{1F1E6}-\u{1F1FF}\u{200D}\u{20E3}\u{2190}-\u{21FF}]/u

  const CASES: ReceiptInput[] = [
    BASE,
    withPatch({ verdict: 'optimal', value: 6, at65: 601_300, delta: 0 }),
    withPatch({ verdict: 'over', value: 15, at65: 900_000, delta: 298_700 }),
    withPatch({ callNo: 1, unit: '', value: 500 }),
  ]

  it('is exactly five lines', () => {
    // Five is the ceiling iMessage and WhatsApp show without a "read more"; a
    // sixth line is the same as not sending the last one.
    for (const input of CASES) {
      expect(receiptText(input).split('\n')).toHaveLength(5)
    }
  })

  it('has no empty lines', () => {
    for (const input of CASES) {
      for (const line of receiptText(input).split('\n')) {
        expect(line.trim().length).toBeGreaterThan(0)
      }
    }
  })

  it('contains no emoji', () => {
    for (const input of CASES) {
      expect(EMOJI.test(receiptText(input))).toBe(false)
    }
  })

  it('carries a bare domain and never a URL', () => {
    for (const input of CASES) {
      const text = receiptText(input)
      expect(text).toContain('compound.day')
      expect(text.toLowerCase()).not.toContain('http')
      expect(text).not.toContain('?')
      expect(text).not.toContain('utm')
    }
  })

  it('names the call number', () => {
    expect(receiptText(BASE)).toContain('No.13')
    expect(receiptText(withPatch({ callNo: 142 }))).toContain('No.142')
  })

  it('leads with the verdict and the headline number', () => {
    expect(receiptText(BASE)).toContain('LEFT BEHIND')
    expect(receiptText(BASE)).toContain('$413K at 65')
    expect(receiptText(CASES[1])).toContain('OPTIMAL PLAY')
    expect(receiptText(CASES[2])).toContain('OVERSHOT')
  })

  it('never discloses the player’s salary', () => {
    // The share artifact is a game result. Anything that could be reverse-read as
    // personal financial data stays on the device — including via a breakdown
    // line a compute function put on the on-screen card.
    for (const input of CASES) {
      const text = receiptText(input)
      expect(text).not.toContain(String(SALARY))
      expect(text).not.toContain('62,000')
      expect(text).not.toContain('$62K')
      for (const line of input.breakdown) expect(text).not.toContain(line.value)
    }
  })
})

describe('receiptCode', () => {
  it('is fixed width so the barcode digits never reflow', () => {
    const widths = new Set(
      [BASE, withPatch({ callNo: 1, value: 0, at65: 0 }), withPatch({ callNo: 9999, value: 15 })].map(
        (i) => receiptCode(i).length,
      ),
    )
    expect(widths.size).toBe(1)
  })

  it('keeps a fractional position instead of rounding it away', () => {
    expect(receiptCode(withPatch({ value: 7.5 }))).not.toBe(receiptCode(withPatch({ value: 7 })))
  })
})

// TEMP-SHOT
import { writeFileSync } from 'node:fs'
import { barcodeUnits, deltaLine, stampText, DOMAIN } from './receipt'
import { money } from './format'
it('shot', () => {
  const render = (input: ReceiptInput) => {
    const seed = receiptSeed(input)
    const w = barcodeWidths(seed)
    const units = barcodeUnits(w)
    let x = 0
    const rects = w
      .map((bw, i) => {
        const at = x
        x += bw
        return i % 2 === 0 ? `<rect x="${at}" y="0" width="${bw}" height="10"/>` : ''
      })
      .join('')
    const d = deltaLine(input)
    const items = receiptLines(input)
      .map(
        (l) =>
          `<li class="receipt-item" ${l.emphasis ? 'data-emphasis' : ''}><span class="receipt-k">${l.label}</span><span class="receipt-leader"></span><span class="receipt-v num">${l.value}</span></li>`,
      )
      .join('')
    return `<div class="receipt-lift"><figure class="receipt" style="clip-path:${tearClipPath(seed, 11)}">
<div class="receipt-brand"><span class="receipt-brand-text">COMPOUND</span></div>
<div class="receipt-meta"><span>No.${input.callNo}</span><span>${input.date}</span></div>
<div class="receipt-rule"></div>
<h2 class="receipt-title">${input.title}</h2>
<div class="receipt-rule"></div>
<ul class="receipt-items">${items}</ul>
<div class="receipt-rule receipt-rule--double"></div>
<div class="receipt-total"><span class="receipt-k">Total at 65</span><strong class="receipt-hero num">${money(input.at65)}</strong></div>
<div class="receipt-item receipt-delta" data-verdict="${input.verdict}"><span class="receipt-k">${d.label}</span><span class="receipt-leader"></span><span class="receipt-v num">${d.value}</span></div>
<div class="receipt-stamp" data-verdict="${input.verdict}"><span class="receipt-stamp-text">*** ${stampText(input.verdict)} ***</span></div>
<svg class="receipt-barcode" viewBox="0 0 ${units} 10" preserveAspectRatio="none">${rects}</svg>
<div class="receipt-foot"><span class="num">${receiptCode(input)}</span><span class="receipt-domain">${DOMAIN}</span></div>
</figure></div>`
  }
  const html = `<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="/src/styles/fonts.css"><link rel="stylesheet" href="/src/styles/tokens.css"><link rel="stylesheet" href="/src/styles/base.css"><link rel="stylesheet" href="/src/ui/Receipt.css">
<style>body{background:var(--bg);display:flex;gap:24px;padding:24px;align-items:flex-start}
.col{width:362px}.thumb{width:120px}</style></head><body>
<div class="col">${render(BASE)}</div>
<div class="thumb">${render(BASE)}</div>
<div class="col">${render(withPatch({ verdict: 'optimal', value: 6, at65: 601300, delta: 0 }))}</div>
<div class="thumb">${render(withPatch({ verdict: 'optimal', value: 6, at65: 601300, delta: 0 }))}</div>
</body></html>`
  writeFileSync('/home/user/eric-gpt/app/public/__shot.html', html)
})
