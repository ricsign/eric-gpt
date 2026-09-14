import { describe, expect, it } from 'vitest'
import {
  DOMAIN,
  PEN_SEGMENTS,
  TEAR_SEGMENTS,
  answerNote,
  answerText,
  dialText,
  isMoneyDial,
  napkinSeed,
  napkinText,
  penPathData,
  penStroke,
  shareHandle,
  shareLink,
  sharedFacts,
  tearClipPath,
  tearPath,
  type NapkinInput,
} from './receipt'
import { CALLS } from '../calls/registry'
import { resolveOptimal, type Profile } from '../calls/types'

const ORIGIN = 'napkin.example'

/**
 * The profiles the corners of the product live at.
 *
 * The salary slider runs $15k-$400k and one answer — the April refund — is a
 * function of the tax bill, so the widest and narrowest phrasings come from the
 * ends of that range, not from the median profile every other fixture uses.
 */
const PROFILES: Profile[] = [
  { salary: 15_000, age: 22 },
  { salary: 62_000, age: 30 },
  { salary: 400_000, age: 55 },
]

/** Every napkin the product can actually produce, guess shown and hidden. */
function* everyNapkin(): Generator<{ input: NapkinInput; profile: Profile }> {
  for (const profile of PROFILES) {
    for (const [i, call] of CALLS.entries()) {
      const v = call.variable
      for (const withGuess of [false, true]) {
        yield {
          profile,
          input: {
            questionNo: i + 1,
            date: 'SEP 13',
            scene: call.title,
            question: call.question,
            givens: sharedFacts(call.fixed),
            answer: answerText(resolveOptimal(call.optimal, profile), v),
            note: answerNote(v),
            rule: call.rule,
            guess: withGuess ? dialText(v.start, v) : undefined,
          },
        }
      }
    }
  }
}

const BASE: NapkinInput = {
  questionNo: 4,
  date: 'SEP 13',
  scene: 'The sofa is 0% interest. Until it is not.',
  question: 'How fast do you have to clear it?',
  givens: [
    { k: 'ON THE STORE CARD', v: '$3,000' },
    { k: 'MISS THE DEADLINE', v: '26.99% FROM DAY ONE' },
  ],
  answer: '12mo or less',
  note: 'to pay it off in full',
  rule: 'Treat 0% as a deadline, not a discount.',
}

const withPatch = (patch: Partial<NapkinInput>): NapkinInput => ({ ...BASE, ...patch })

describe('where a share points', () => {
  it('is never the domain that belongs to another company', () => {
    // compound.day was a string literal in receipt.ts and it is a live AI
    // workspace product, so every successful share this app ever made handed
    // the recipient to a competitor. The literal is the bug; this is the guard
    // on it never coming back.
    expect(DOMAIN).not.toContain('compound.day')
    for (const { input } of everyNapkin()) {
      expect(napkinText(input, ORIGIN)).not.toContain('compound.day')
    }
  })

  it('prints a bare host and the question, with nothing to track', () => {
    const handle = shareHandle(4, ORIGIN)
    expect(handle).toBe('napkin.example/4')
    expect(handle.toLowerCase()).not.toContain('http')
    expect(handle).not.toContain('?')
    expect(handle).not.toContain('utm')
  })

  it('sends the text, the card and the copied link to the same place', () => {
    // Three code paths used to derive their address three ways — a literal, a
    // literal again, and location.origin — so a build served anywhere but the
    // canonical host disagreed with itself in the one place nobody checks.
    for (const n of [1, 4, 10]) {
      expect(napkinText(withPatch({ questionNo: n }), ORIGIN)).toContain(shareHandle(n, ORIGIN))
      expect(shareLink(n, ORIGIN).endsWith(shareHandle(n, ORIGIN))).toBe(true)
    }
  })

  it('copies an absolute URL, and keeps a configured scheme', () => {
    expect(shareLink(4, ORIGIN)).toBe('https://napkin.example/4')
    expect(shareLink(4, 'https://napkin.example')).toBe('https://napkin.example/4')
    expect(shareLink(4, 'https://napkin.example/')).toBe('https://napkin.example/4')
    // A local origin is served over http and rewriting it to https would make
    // every shared link from a dev build dead on arrival.
    expect(shareLink(4, 'http://localhost:5173')).toBe('http://localhost:5173/4')
  })

  it('says nothing rather than something broken when there is no origin', () => {
    // Server-side, or a test: `location` does not exist and nothing was
    // configured. A half-formed address on the card is worse than no address.
    expect(shareHandle(4, '')).toBe('')
    expect(napkinText(BASE, '')).not.toContain('//')
    for (const line of napkinText(BASE, '').split('\n')) {
      expect(line.trim().length).toBeGreaterThan(0)
    }
  })
})

describe('sharedFacts', () => {
  it('drops every given the profile would fill in', () => {
    // The share artifact carries a question and its answer. Three of the ten
    // question records set a tile to the reader's own pay, and those tiles are
    // dropped rather than resolved — the rule is "no placeholder survives",
    // not "not the salary one", so a new token cannot leak by being new.
    for (const call of CALLS) {
      for (const fact of sharedFacts(call.fixed)) {
        expect(fact.v, `question ${call.id}`).not.toContain('{{')
      }
    }
  })

  it('leaves every question with a setup', () => {
    // The givens are what make the answer surprising instead of arbitrary. A
    // question that lost both tiles would ship a card with no numbers on it
    // but the answer.
    for (const call of CALLS) {
      expect(sharedFacts(call.fixed).length, `question ${call.id}`).toBeGreaterThan(0)
    }
  })
})

describe('isMoneyDial', () => {
  it('separates the dollar dials from the one that counts days', () => {
    // Five dials carry an empty unit, because the readout prints the unit
    // straight after the figure and "3200$" is not a thing. Four of them are
    // dollars and the fifth counts the market's best days; nothing in the
    // record says which is which. Renaming a dial breaks this rather than
    // quietly printing "$4" for four missed days.
    const bare = CALLS.filter((c) => c.variable.unit === '')
    expect(bare).toHaveLength(5)
    expect(bare.filter((c) => !isMoneyDial(c.variable)).map((c) => c.variable.key)).toEqual([
      'daysMissed',
    ])
  })

  it('never calls a dial with a unit money', () => {
    for (const call of CALLS) {
      if (call.variable.unit !== '') expect(isMoneyDial(call.variable)).toBe(false)
    }
  })
})

describe('dialText', () => {
  it('reads a negative refund as a bill, not as negative money', () => {
    // The refund dial runs through zero. "-$2,000 back in April" is a double
    // negative that reads as a refund, which is the opposite of the truth.
    const refund = CALLS.find((c) => c.variable.key === 'refund')!.variable
    expect(dialText(-2000, refund)).toBe('owe $2,000')
    expect(dialText(0, refund)).toBe('$0')
    expect(dialText(3000, refund)).toBe('$3,000')
  })

  it('puts a unit straight after the figure, the way the dial does', () => {
    const months = CALLS.find((c) => c.variable.key === 'payoffMonths')!.variable
    expect(dialText(18, months)).toBe('18mo')
  })
})

describe('answerText', () => {
  /**
   * The exact words that leave the app, for all ten.
   *
   * Locked rather than derived, because this is copy: a change here is a
   * change to the sentence a stranger reads, and it should have to be typed
   * out on purpose. The April refund is a regular expression because its
   * answer is a function of the tax bill and genuinely differs per person.
   */
  const GOLDEN: Record<number, string | RegExp> = {
    1: '6% or more',
    2: '$0',
    3: '3mo to 6mo',
    4: '12mo or less',
    5: '6% to 10%',
    6: /^(\$0|owe \$[\d,]+)$/,
    7: '$2,000 to $3,000',
    8: '$3',
    9: '5yr or more',
    10: '0',
  }

  it('says each answer the way the dial said it', () => {
    const profile = { salary: 62_000, age: 30 }
    for (const call of CALLS) {
      const said = answerText(resolveOptimal(call.optimal, profile), call.variable)
      const want = GOLDEN[call.id]
      if (typeof want === 'string') expect(said, `question ${call.id}`).toBe(want)
      else expect(said, `question ${call.id}`).toMatch(want)
    }
  })

  it('states a band that reaches the end of the dial as an open one', () => {
    // Saving more than the employer match is not a mistake, and "6% to 15%"
    // says it is — 15% is where the control stops, not where the answer does.
    const match = CALLS.find((c) => c.variable.key === 'contribution')!
    expect(answerText(resolveOptimal(match.optimal, PROFILES[1]), match.variable)).toBe(
      '6% or more',
    )
    const promo = CALLS.find((c) => c.variable.key === 'payoffMonths')!
    expect(answerText(resolveOptimal(promo.optimal, PROFILES[1]), promo.variable)).toBe(
      '12mo or less',
    )
  })

  it('states a band with room on both sides as a span', () => {
    const fund = CALLS.find((c) => c.variable.key === 'months')!
    expect(answerText(resolveOptimal(fund.optimal, PROFILES[1]), fund.variable)).toBe('3mo to 6mo')
  })

  it('is a readable figure on every question, at every profile', () => {
    for (const profile of PROFILES) {
      for (const call of CALLS) {
        const said = answerText(resolveOptimal(call.optimal, profile), call.variable)
        expect(said, `question ${call.id}`).toMatch(/\d/)
        expect(said).not.toContain('NaN')
        expect(said).not.toContain('undefined')
        if (isMoneyDial(call.variable)) expect(said).toContain('$')
      }
    }
  })
})

describe('answerNote', () => {
  it('does not say the unit twice', () => {
    // The dial prints a bare number above its caption, so four captions open
    // with the word "dollars". The card prints "$3", and leaving the word in
    // gives "$3, dollars a year per $10,000" — which reads as a typo.
    for (const call of CALLS) {
      const note = answerNote(call.variable)
      if (isMoneyDial(call.variable)) {
        expect(note.toLowerCase(), `question ${call.id}`).not.toMatch(/^dollars\b/)
      }
      expect(note.length).toBeGreaterThan(0)
    }
  })

  it('leaves a captioned unit alone', () => {
    const match = CALLS.find((c) => c.variable.key === 'contribution')!.variable
    expect(answerNote(match)).toBe('of your pay, into retirement')
    const fee = CALLS.find((c) => c.variable.key === 'expenseRatio')!.variable
    expect(answerNote(fee)).toBe('a year per $10,000 invested')
  })
})

describe('napkinSeed', () => {
  it('follows from what is printed, not from who printed it', () => {
    // Two people sharing the same fact share the same paper, which is correct
    // now that the card is about the fact. It also means turning the optional
    // guess on does not re-tear the paper under the sender's hands.
    expect(napkinSeed(BASE)).toBe(napkinSeed(withPatch({ guess: '18mo' })))
    expect(napkinSeed(BASE)).not.toBe(napkinSeed(withPatch({ answer: '6mo or less' })))
    expect(napkinSeed(BASE)).not.toBe(napkinSeed(withPatch({ questionNo: 5 })))
  })
})

describe('tearPath', () => {
  const seed = 'napkin/top'

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
  it('closes a polygon with both edges torn from the same napkin seed', () => {
    const css = tearClipPath(napkinSeed(BASE), 11)
    expect(css.startsWith('polygon(')).toBe(true)
    expect(css.split(',')).toHaveLength((TEAR_SEGMENTS + 1) * 2)
    // The bottom edge is measured from the far side so the card can be any height.
    expect(css).toContain('calc(100% - ')
    expect(css).toBe(tearClipPath(napkinSeed(BASE), 11))
  })

  it('gives a different card a different tear', () => {
    expect(tearClipPath(napkinSeed(BASE), 11)).not.toBe(
      tearClipPath(napkinSeed(withPatch({ questionNo: 5 })), 11),
    )
  })
})

describe('penStroke', () => {
  it('is deterministic for one seed', () => {
    expect(penStroke('a', 100)).toEqual(penStroke('a', 100))
  })

  it('draws a different line for a different card', () => {
    // The whole point of deriving the mark from the card: two different facts
    // that drew the same stroke would make the stroke wallpaper.
    expect(penStroke(napkinSeed(BASE), 100)).not.toEqual(
      penStroke(napkinSeed(withPatch({ answer: '6mo or less' })), 100),
    )
  })

  it('runs left to right without doubling back', () => {
    for (const s of ['a', 'b', 'c', 'd']) {
      const pts = penStroke(s, 240)
      expect(pts).toHaveLength(PEN_SEGMENTS + 1)
      for (let i = 1; i < pts.length; i++) {
        expect(pts[i].x).toBeGreaterThan(pts[i - 1].x)
      }
      expect(pts[0].x).toBe(0)
      expect(pts[pts.length - 1].x).toBeCloseTo(240, 6)
    }
  })

  it('starts and ends on the baseline', () => {
    // A stroke that stops mid-wobble looks cut off rather than lifted, and the
    // amount it is off by is exactly the amount it looks broken by.
    for (const s of ['a', 'b', 'c', 'd']) {
      const pts = penStroke(s, 100)
      expect(pts[0].y).toBe(0)
      expect(pts[pts.length - 1].y).toBe(0)
    }
  })

  it('stays inside the band the renderer sized for it', () => {
    for (const s of ['a', 'b', 'c', 'd', 'e', 'f']) {
      for (const p of penStroke(s, 100, 60)) {
        expect(Math.abs(p.y)).toBeLessThanOrEqual(1)
      }
    }
  })

  it('wanders rather than jumps', () => {
    // White noise at this amplitude reads as a zigzag, which is the one thing a
    // pen stroke must not look like. Each point steps from the last one.
    for (const s of ['a', 'b', 'c', 'd', 'e', 'f']) {
      const pts = penStroke(s, 100, 60)
      for (let i = 1; i < pts.length; i++) {
        expect(Math.abs(pts[i].y - pts[i - 1].y)).toBeLessThan(0.5)
      }
    }
  })

  it('is not a straight line', () => {
    const ys = penStroke('wobble', 100, 40).map((p) => p.y)
    expect(Math.max(...ys.map(Math.abs))).toBeGreaterThan(0.2)
  })

  it('scales x with width and leaves y normalised', () => {
    const small = penStroke('s', 100)
    const large = penStroke('s', 1000)
    small.forEach((p, i) => {
      expect(large[i].x).toBeCloseTo(p.x * 10, 6)
      expect(large[i].y).toBe(p.y)
    })
  })
})

describe('penPathData', () => {
  it('is one move and then lines, centred on the baseline it is given', () => {
    const d = penPathData(penStroke('a', 100, 3), 1)
    expect(d.startsWith('M')).toBe(true)
    expect(d.match(/M/g)).toHaveLength(1)
    expect(d.match(/L/g)).toHaveLength(3)
    expect(d).not.toContain('NaN')
    // Both ends sit on the baseline, so the path opens and closes at `mid`.
    expect(d.startsWith('M0.00,1.000')).toBe(true)
    expect(d.endsWith('1.000')).toBe(true)
  })
})

describe('napkinText', () => {
  // Pictographs, dingbats, flag letters, variation selectors, the keycap combiner
  // and the ZWJ that joins emoji sequences — plus arrows, which are not emoji but
  // render as one in several chat clients. Written as alternatives rather than one
  // class because a combining mark inside a character class does not mean what it
  // looks like it means.
  const EMOJI =
    /[\u{1F000}-\u{1FAFF}]|[\u{2600}-\u{27BF}]|[\u{2B00}-\u{2BFF}]|[\u{1F1E6}-\u{1F1FF}]|[\u{2190}-\u{21FF}]|[\u{FE00}-\u{FE0F}]|\u{20E3}|\u{200D}/u

  it('is exactly five lines, none of them empty', () => {
    // Five is the ceiling iMessage and WhatsApp show without a "read more"; a
    // sixth line is the same as not sending the last one.
    for (const { input } of everyNapkin()) {
      const lines = napkinText(input, ORIGIN).split('\n')
      expect(lines).toHaveLength(5)
      for (const line of lines) expect(line.trim().length).toBeGreaterThan(0)
    }
  })

  it('contains no emoji', () => {
    for (const { input } of everyNapkin()) {
      expect(EMOJI.test(napkinText(input, ORIGIN))).toBe(false)
    }
  })

  it('leads with the question and then answers it', () => {
    const text = napkinText(BASE, ORIGIN)
    expect(text).toContain('How fast do you have to clear it?')
    expect(text).toContain('The answer: 12mo or less, to pay it off in full.')
  })

  it('never carries the sender’s score', () => {
    // The thing the whole rewrite exists to remove. No projection, no gap to
    // the right answer, no verdict — and not the guess either: five lines are
    // all spoken for and the fact is worth more than the confession.
    for (const { input } of everyNapkin()) {
      const text = napkinText(withPatch({ ...input, guess: '18mo' }), ORIGIN)
      expect(text).not.toContain('18mo')
      expect(text.toLowerCase()).not.toContain('at 65')
      expect(text.toLowerCase()).not.toContain('left behind')
    }
  })

  it('never discloses the reader’s pay', () => {
    // Anything that could be reverse-read as personal financial data stays on
    // the device. `sharedFacts` makes that structural; this is the proof.
    for (const { input, profile } of everyNapkin()) {
      const text = napkinText(input, ORIGIN)
      expect(text).not.toContain(String(profile.salary))
      expect(text).not.toContain(profile.salary.toLocaleString('en-US'))
      expect(text).not.toContain('{{')
    }
  })

  it('carries the disclaimer', () => {
    for (const { input } of everyNapkin()) {
      expect(napkinText(input, ORIGIN)).toContain('Estimates, not advice.')
      expect(napkinText(input, '')).toContain('Estimates, not advice.')
    }
  })

  it('ends on an invitation with nothing to track', () => {
    const last = napkinText(BASE, ORIGIN).split('\n')[4]
    expect(last).toContain(shareHandle(4, ORIGIN))
    expect(last.toLowerCase()).not.toContain('http')
    expect(last).not.toContain('?')
    expect(last).not.toContain('utm')
  })
})

describe('every napkin the product can produce', () => {
  it('says what was asked, what the answer is, and where to try it', () => {
    for (const { input } of everyNapkin()) {
      const where = `question ${input.questionNo}`
      expect(input.question.endsWith('?'), where).toBe(true)
      expect(input.answer.length, where).toBeGreaterThan(0)
      expect(input.note.length, where).toBeGreaterThan(0)
      expect(input.rule.length, where).toBeGreaterThan(0)
      expect(input.givens.length, where).toBeGreaterThan(0)
      for (const g of input.givens) expect(g.v, where).not.toContain('{{')
      expect(napkinText(input, ORIGIN)).toContain(input.answer)
    }
  })
})
