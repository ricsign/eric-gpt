import { describe, expect, it } from 'vitest'
import {
  BARCODE_MAX_WIDTH,
  BARCODE_MIN_WIDTH,
  BARCODE_MODULES,
  MAX_LINES,
  TEAR_SEGMENTS,
  barcodeWidths,
  deltaLine,
  receiptCode,
  receiptLines,
  receiptSeed,
  receiptText,
  stampText,
  tearClipPath,
  tearPath,
  type ReceiptInput,
} from './receipt'
import { CALLS } from '../calls/registry'
import { COMPUTE } from '../calls/compute'
import { judge, referenceValue, stepCount, type Verdict } from '../calls/types'

/**
 * Every receipt the product can actually produce.
 *
 * The salary slider runs $15k-$400k and the age the horizon is measured from
 * moves with the player, so the widest figures on the card come from the corners
 * of that space, not from the median profile every other fixture uses. Several
 * invariants below only break out there.
 */
function* everyReceipt(): Generator<ReceiptInput> {
  for (const salary of [15_000, 62_000, 400_000]) {
    for (const age of [22, 30, 55]) {
      const profile = { salary, age }
      for (const call of CALLS) {
        const compute = COMPUTE[call.compute]
        const v = call.variable
        for (let i = 0; i < stepCount(v); i++) {
          const value = v.min + i * v.step
          const mine = compute(value, profile)
          const best = compute(referenceValue(value, call.optimal, profile), profile)
          yield {
            callNo: call.id,
            date: 'SEP 13',
            title: call.title,
            verdict: judge(value, call.optimal, profile),
            value,
            unit: v.unit,
            breakdown: mine.breakdown,
            at65: mine.at65,
            delta: mine.at65 - best.at65,
          }
        }
      }
    }
  }
}

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

  it('starts and ends on ink', () => {
    // Even indices are bars. An even module count would end the band on paper,
    // and a barcode whose right edge is blank floats off the card — visible only
    // once, in whatever was already shared.
    expect(BARCODE_MODULES % 2).toBe(1)
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

  it('still states the play when the line that stated it falls off the end', () => {
    // The check used to run on the whole breakdown and the cut ran afterwards,
    // so a breakdown longer than the card could satisfy the check with a line
    // that was then thrown away — leaving a receipt that never says what was
    // chosen. No call ships more than six lines today; one will.
    const lines = receiptLines(
      withPatch({
        value: 9,
        unit: '%',
        breakdown: [
          { label: 'ONE', value: '$1' },
          { label: 'TWO', value: '$2' },
          { label: 'THREE', value: '$3' },
          { label: 'FOUR', value: '$4' },
          { label: 'FIVE', value: '$5' },
          { label: 'SIX', value: '$6' },
          { label: 'RATE', value: '9%' },
        ],
      }),
    )
    expect(lines).toHaveLength(MAX_LINES)
    expect(lines[0]).toEqual({ label: 'YOUR PLAY', value: '9%' })
  })

  it('still names the play when a call supplies no breakdown at all', () => {
    const lines = receiptLines(withPatch({ breakdown: [] }))
    expect(lines).toEqual([{ label: 'YOUR PLAY', value: '3%' }])
  })
})

describe('the receipt states the play exactly once', () => {
  // The receipt is the thing that leaves the app. A number printed twice under
  // two labels reads as two findings, and the reader has no way to tell that
  // "YOUR PLAY 63bps" and "EXPENSE RATIO 63 BP" are the same fact.
  it('drops the generic line when a compute line already states the position', () => {
    const lines = receiptLines({
      ...BASE,
      value: 63,
      unit: 'bps',
      breakdown: [
        { label: 'EXPENSE RATIO', value: '63 BP' },
        { label: 'FEE / MO NOW', value: '$48' },
        { label: 'LOST TO FEES BY 65', value: '$349,976', emphasis: true },
      ],
    })
    expect(lines.map((l) => l.label)).not.toContain('YOUR PLAY')
    expect(lines[0]).toEqual({ label: 'EXPENSE RATIO', value: '63 BP' })
  })

  it('keeps the generic line when nothing else states the position', () => {
    const lines = receiptLines({
      ...BASE,
      value: 8,
      unit: '%',
      breakdown: [
        { label: 'YEAR ONE', value: '$4,960' },
        { label: 'OFFER PULLED', value: '35%' },
        { label: 'LIFETIME AT 65', value: '$1.2M', emphasis: true },
      ],
    })
    expect(lines[0]).toEqual({ label: 'YOUR PLAY', value: '8%' })
  })

  it('never lets the emphasised line stand in for the play', () => {
    // A call whose headline figure happens to equal the position must still say
    // what was chosen, or the receipt loses the play entirely.
    const lines = receiptLines({
      ...BASE,
      value: 12,
      unit: ' MONTHS',
      breakdown: [
        { label: 'PAYMENT / MO', value: '$250' },
        { label: 'DEFERRED INTEREST', value: '12', emphasis: true },
      ],
    })
    expect(lines[0]).toEqual({ label: 'YOUR PLAY', value: '12 MONTHS' })
  })

  it('states the position exactly once on every real call', () => {
    // Not "no two lines share a number" — at 3% on call 1 the employer adds
    // $930 and the player misses exactly $930, which are two different facts
    // that happen to be equal. What must never happen is the *position* being
    // printed twice, because that is one fact wearing two labels.
    for (const call of CALLS) {
      const compute = COMPUTE[call.compute]
      for (const value of [call.variable.min, call.variable.start, call.variable.max]) {
        const { breakdown } = compute(value, { salary: SALARY, age: 30 })
        const lines = receiptLines({ ...BASE, value, unit: call.variable.unit, breakdown })
        const stating = lines.filter((l) => {
          const first = l.value.match(/-?\d[\d,]*(?:\.\d+)?/)
          return first !== null && Number(first[0].replace(/,/g, '')) === value
        })
        // The generic line is only there to fill a gap. If a compute line
        // already carries the position, adding "YOUR PLAY" alongside it prints
        // one fact twice — which is the whole bug this guards.
        // At zero every line reads as the play, so the generic one stays —
        // see the guard in statesPlay.
        if (value !== 0 && stating.some((l) => l.label !== 'YOUR PLAY')) {
          expect(
            lines.map((l) => l.label),
            `call ${call.id} at ${value}: ${stating.map((l) => l.label).join(' / ')}`,
          ).not.toContain('YOUR PLAY')
        }
        // Whatever else it says, it has to say what was chosen.
        // Compared with the thousands separators stripped: the play is written
        // "$3,200" on the receipt and 3200 in the record.
        expect(
          lines.some((l) => l.value.replace(/,/g, '').includes(String(value))),
          `call ${call.id} at ${value} never states the play`,
        ).toBe(true)
      }
    }
  })
})

describe('receiptText', () => {
  // Pictographs, dingbats, flag letters, variation selectors, the keycap combiner
  // and the ZWJ that joins emoji sequences — plus arrows, which are not emoji but
  // render as one in several chat clients. Written as alternatives rather than one
  // class because a combining mark inside a character class does not mean what it
  // looks like it means.
  const EMOJI =
    /[\u{1F000}-\u{1FAFF}]|[\u{2600}-\u{27BF}]|[\u{2B00}-\u{2BFF}]|[\u{1F1E6}-\u{1F1FF}]|[\u{2190}-\u{21FF}]|[\u{FE00}-\u{FE0F}]|\u{20E3}|\u{200D}/u

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
  it('is one fixed width across every receipt the product can produce', () => {
    // The foot is a single no-wrap flex row inside a clip-path, so a code that
    // grows a digit pushes COMPOUND.DAY off the paper. Sampling three tidy
    // fixtures said this held; the real ranges say otherwise — the repair call
    // reaches $4,000 (six digits once scaled to hundredths) and a 22-year-old on
    // the top of the salary slider clears eight figures at 65.
    const widths = new Set<number>()
    let widest: ReceiptInput | null = null
    for (const input of everyReceipt()) {
      widths.add(receiptCode(input).length)
      if (!widest || input.at65 > widest.at65) widest = input
    }
    expect(widths.size, `saw code lengths ${[...widths].join(', ')}`).toBe(1)
    // Proof the sweep actually reached the sizes that used to overflow, rather
    // than agreeing with itself on a narrow sample.
    expect(widest!.at65).toBeGreaterThan(9_999_999)
  })

  it('survives a figure past its own field width without reflowing', () => {
    const huge = receiptCode(withPatch({ at65: 9.9e12, value: 999_999 }))
    expect(huge.length).toBe(receiptCode(BASE).length)
    expect(huge).not.toContain('NaN')
  })

  it('keeps a fractional position instead of rounding it away', () => {
    expect(receiptCode(withPatch({ value: 7.5 }))).not.toBe(receiptCode(withPatch({ value: 7 })))
  })

  it('separates two positions a single step apart on every call', () => {
    // The code is the only place the exact position is written down. Two
    // adjacent stops sharing one code would make it decoration.
    for (const call of CALLS) {
      const { min, step } = call.variable
      const a = receiptCode(withPatch({ callNo: call.id, value: min }))
      const b = receiptCode(withPatch({ callNo: call.id, value: min + step }))
      expect(a, `call ${call.id}`).not.toBe(b)
    }
  })
})

describe('deltaLine', () => {
  it('writes a true minus, not a hyphen', () => {
    // U+002D next to a dollar sign at 13px reads as the dash in the label above
    // it. The distinction is invisible in a diff, which is why it is asserted.
    const value = deltaLine(BASE).value
    expect(value.codePointAt(0)).toBe(0x2212)
    expect(value).toBe('−$188,400')
  })

  it('is even, not a loss of nothing, when the money rounds away', () => {
    // `money` rounds to the dollar. Without a guard a forty-cent gap prints as
    // "−$0" — a loss, stated in red, of nothing.
    const line = deltaLine(withPatch({ verdict: 'short', delta: -0.4 }))
    expect(line.value).toBe('EVEN')
    expect(line.tone).toBe('even')
  })

  it('takes its colour from the money, not from the verdict', () => {
    // An `over` play still loses money on every call in the product. Keying the
    // tone off `verdict === 'short'` left that loss in neutral ink underneath a
    // red stamp — the two marks contradicting each other about one number.
    expect(deltaLine(withPatch({ verdict: 'over', delta: -12_252 })).tone).toBe('loss')
    expect(deltaLine(withPatch({ verdict: 'optimal', delta: 0 })).tone).toBe('even')
  })
})

describe('stampText', () => {
  it('gives every verdict its own word, and never the loss word for a win', () => {
    const verdicts: Verdict[] = ['optimal', 'short', 'over']
    const words = verdicts.map(stampText)
    expect(new Set(words).size).toBe(verdicts.length)
    for (const w of words) expect(w).toBe(w.toUpperCase())
    expect(stampText('optimal')).not.toMatch(/BEHIND|OVERSHOT/)
  })
})

describe('every receipt the product can produce', () => {
  it('fits the card and states the play', () => {
    for (const input of everyReceipt()) {
      const lines = receiptLines(input)
      expect(lines.length, `call ${input.callNo} at ${input.value}`).toBeGreaterThan(0)
      expect(lines.length).toBeLessThanOrEqual(MAX_LINES)
      expect(
        lines.some((l) => l.value.replace(/,/g, '').includes(String(input.value))),
        `call ${input.callNo} at ${input.value} never states the play`,
      ).toBe(true)
      expect(receiptText(input).split('\n')).toHaveLength(5)
    }
  })

  it('never stamps a loss beside a gain', () => {
    // The stamp is a judgement and the figure beneath it is a fact; if a call
    // ever paid more for a wrong answer at 65, the card would read
    // "MONEY LEFT BEHIND" directly above "+$12,000" and mean nothing. Today no
    // call does, and this is the guard on that staying true.
    for (const input of everyReceipt()) {
      if (input.verdict === 'optimal') continue
      expect(
        deltaLine(input).tone,
        `call ${input.callNo} at ${input.value} rewards a non-optimal play`,
      ).not.toBe('gain')
    }
  })
})
