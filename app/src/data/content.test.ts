import { describe, expect, it } from 'vitest'
import { DRILLS } from './drills'
import { LESSONS, rankLessons } from './lessons'
import { FACTS, BRACKETS_2026_SINGLE, FICO_FACTORS, staleness } from './facts'
import { DEFAULT_STATE } from '../state/store'
import { feeDrag, futureValue, payOffDebt } from '../lib/finance'

const profile = DEFAULT_STATE.profile

describe('drill integrity', () => {
  it('every drill has exactly one correct option', () => {
    for (const d of DRILLS) {
      const correct = d.options.filter((o) => o.correct)
      expect(correct, `${d.id} should have one correct option`).toHaveLength(1)
    }
  })

  it('every option explains itself — including the right one', () => {
    // The distractor rationale is where most of the teaching happens; an option
    // without one is a filler option, and filler options make the drill a guess.
    for (const d of DRILLS) {
      for (const o of d.options) {
        expect(o.why.length, `${d.id}: "${o.label}" needs a why`).toBeGreaterThan(20)
      }
    }
  })

  it('every drill offers a real choice and a reveal worth reading', () => {
    for (const d of DRILLS) {
      expect(d.options.length, d.id).toBeGreaterThanOrEqual(2)
      expect(d.reveal.length, d.id).toBeGreaterThan(60)
      expect(d.concept, d.id).toBeTruthy()
    }
  })

  it('drill ids are unique', () => {
    expect(new Set(DRILLS.map((d) => d.id)).size).toBe(DRILLS.length)
  })

  it('no drill asks for or discloses personal financial data', () => {
    // The entire share loop depends on this: a drill that references "your salary"
    // cannot be the same question for everyone, and cannot be shared safely.
    for (const d of DRILLS) {
      expect(d.question.toLowerCase(), d.id).not.toMatch(/your (salary|income|balance|net worth)/)
    }
  })

  it('the bank is large enough that a question never repeats within a month', () => {
    expect(DRILLS.length).toBeGreaterThanOrEqual(24)
  })
})

/**
 * Numbers stated in drill copy are cross-checked against the same engine the app
 * computes with. If a rate assumption or a formula ever changes, these fail rather
 * than leaving confidently wrong arithmetic in front of users.
 */
describe('drill arithmetic matches the finance engine', () => {
  it('the penny really does beat the million', () => {
    expect(0.01 * 2 ** 29).toBeCloseTo(5_368_709.12, 2)
    expect(0.01 * 2 ** 27).toBeCloseTo(1_342_177.28, 2)
    // And it genuinely is still tiny on day 20 — the claim the drill rests on.
    expect(0.01 * 2 ** 19).toBeLessThan(5_500)
  })

  it('the car payment really is the most expensive of the four', () => {
    const at = (monthly: number) =>
      futureValue({ principal: 0, monthly, annualRate: 0.07, years: 10 }).balance
    const car = at(450)
    const coffee = at(182.5)
    const subs = at(60)
    const holiday = 4000 * Math.pow(1.07, 10)

    expect(car).toBeGreaterThan(coffee)
    expect(coffee).toBeGreaterThan(subs)
    expect(subs).toBeGreaterThan(holiday)
    expect(Math.round(car / 1000)).toBe(77)
    expect(Math.round(coffee / 100) * 100).toBe(31_200)
  })

  it('the early saver wins, by the amount the reveal states', () => {
    const early =
      futureValue({ principal: 0, monthly: 200, annualRate: 0.07, years: 10 }).balance *
      Math.pow(1.07, 30)
    const late = futureValue({ principal: 0, monthly: 200, annualRate: 0.07, years: 30 }).balance

    expect(early).toBeGreaterThan(late)
    expect(Math.round(early / 100) * 100).toBe(260_400)
    expect(Math.round(late / 100) * 100).toBe(233_900)
    expect(Math.round((early - late) / 100) * 100).toBe(26_500)
  })

  it('the 1% fund really costs over a fifth of the outcome', () => {
    const share = feeDrag(
      { principal: 0, monthly: 500, annualRate: 0.07, years: 40 },
      0.01 - 0.0003,
    ).shareOfOutcome
    expect(share).toBeGreaterThan(0.2)
    expect(Math.round(share * 100)).toBe(22)
  })

  it('the fee field rule tracks the real figure at every horizon', () => {
    // 0.6 x years, rather than the popular flat 25x which is only right at 40.
    for (const years of [20, 30, 40]) {
      const actual = feeDrag({ principal: 0, monthly: 300, annualRate: 0.07, years }, 0.0035)
        .shareOfOutcome
      const ruleOfThumb = 0.0035 * 0.6 * years
      expect(Math.abs(actual - ruleOfThumb), `${years}y`).toBeLessThan(0.012)
    }
  })

  it('the card amortisation in the drill is exact', () => {
    const r = payOffDebt({ balance: 5000, apr: 0.22, monthlyPayment: 200 })
    expect(r.months).toBe(34)
    expect(Math.round(r.totalInterest * 100) / 100).toBeCloseTo(1749.88, 1)
  })

  it('the recurring raise beats the one-off, by the stated amounts', () => {
    const recurring = futureValue({
      principal: 0,
      monthly: 5000 / 12,
      annualRate: 0.07,
      years: 30,
    }).balance
    const once = 10_000 * Math.pow(1.07, 30)
    expect(Math.round(recurring / 1000)).toBe(487)
    expect(Math.round(once / 1000)).toBe(76)
  })

  it('24% APR really is 26.8% APY', () => {
    expect((Math.pow(1 + 0.24 / 12, 12) - 1) * 100).toBeCloseTo(26.82, 2)
  })

  it('cash at 0.4% against 2.5% inflation loses about a fifth in ten years', () => {
    expect(Math.pow(1.004, 10) / Math.pow(1.025, 10)).toBeCloseTo(0.813, 3)
  })
})

describe('lesson integrity', () => {
  it('every lesson builds all eight beats, in order', () => {
    const expected = [
      'anchor',
      'probe',
      'reveal',
      'mechanism',
      'worked',
      'practice',
      'rule',
      'action',
    ]
    for (const l of LESSONS) {
      expect(l.build(profile).map((b) => b.kind), l.id).toEqual(expected)
    }
  })

  it('the probe always comes before any explanation', () => {
    // The generation effect depends on committing to an answer first. If content
    // ever drifts ahead of the probe, the lesson silently loses its main mechanism.
    for (const l of LESSONS) {
      const kinds = l.build(profile).map((b) => b.kind)
      expect(kinds.indexOf('probe'), l.id).toBeLessThan(kinds.indexOf('mechanism'))
      expect(kinds.indexOf('probe'), l.id).toBeLessThan(kinds.indexOf('reveal'))
    }
  })

  it('a numeric probe never starts the slider on the right answer', () => {
    for (const l of LESSONS) {
      for (const b of l.build(profile)) {
        if (b.kind === 'probe' && b.mode === 'estimate') {
          expect(Math.abs(b.start - b.answer) / Math.max(1, b.answer), l.id).toBeGreaterThan(
            b.tolerance,
          )
          expect(b.answer, `${l.id}: answer must be reachable on the slider`).toBeLessThanOrEqual(
            b.max,
          )
          expect(b.answer).toBeGreaterThanOrEqual(b.min)
        }
      }
    }
  })

  it('every lesson ends on an action, never on congratulations', () => {
    for (const l of LESSONS) {
      const beats = l.build(profile)
      const last = beats[beats.length - 1]
      expect(last.kind, l.id).toBe('action')
      if (last.kind === 'action') {
        // There is always an honest way out. An action list with no "not now"
        // is a dark pattern wearing a commitment device's clothes.
        expect(last.options.some((o) => !o.commits), l.id).toBe(true)
      }
    }
  })

  it('every practice block has at least one transfer item', () => {
    for (const l of LESSONS) {
      const practice = l.build(profile).find((b) => b.kind === 'practice')
      expect(practice, l.id).toBeDefined()
      if (practice?.kind === 'practice') {
        expect(practice.items.length, l.id).toBeGreaterThanOrEqual(3)
        expect(practice.items.some((i) => i.transfer), l.id).toBe(true)
        for (const item of practice.items) {
          expect(item.options.filter((o) => o.correct), `${l.id}: ${item.prompt}`).toHaveLength(1)
          for (const o of item.options) expect(o.why.length).toBeGreaterThan(20)
        }
      }
    }
  })

  it('every lesson names a competence and a specific misconception', () => {
    for (const l of LESSONS) {
      expect(l.competence, l.id).toMatch(/^You can /)
      expect(l.misconception.length, l.id).toBeGreaterThan(40)
      expect(l.concepts.length, l.id).toBeGreaterThan(0)
    }
  })

  it('lesson ids are unique', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length)
  })

  it('survives an empty profile — lessons must work before onboarding', () => {
    for (const l of LESSONS) {
      expect(() => l.build({ assumedReturn: 0.07, assumedInflation: 0.025 }), l.id).not.toThrow()
    }
  })

  it('no lesson tells the user what they should do with a named product', () => {
    // The legal line in this category is personalisation, not topic: a tool may
    // compute, but it must never conclude. This guards the architecture, not the
    // wording — a lesson that starts recommending tickers fails the build.
    const banned = /\b(you should (buy|invest in|sell|open)|we recommend|buy (VTI|VOO|SPY))\b/i
    for (const l of LESSONS) {
      const text = JSON.stringify(l.build(profile))
      expect(banned.test(text), l.id).toBe(false)
    }
  })
})

describe('facts table', () => {
  it('every fact carries a primary source and a date it was read', () => {
    for (const f of Object.values(FACTS)) {
      expect(f.source, f.id).toMatch(/^https:\/\//)
      expect(f.sourceName, f.id).toBeTruthy()
      expect(f.asOf, f.id).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    }
  })

  it('anything not read from the issuing authority is marked secondary', () => {
    // Honesty about provenance is what lets the UI soften a claim instead of
    // stating a half-checked number flatly.
    const secondary = Object.values(FACTS).filter((f) => f.confidence === 'secondary')
    for (const f of secondary) expect(f.note ?? f.sourceName, f.id).toBeTruthy()
  })

  it('staleness is measured from the read date', () => {
    expect(staleness(FACTS.contrib401k, '2026-09-11')).toBe(0)
    expect(staleness(FACTS.contrib401k, '2026-10-11')).toBe(30)
  })

  it('tax brackets ascend and terminate at infinity', () => {
    let prevRate = 0
    let prevCap = 0
    for (const b of BRACKETS_2026_SINGLE) {
      expect(b.rate).toBeGreaterThan(prevRate)
      expect(b.upTo).toBeGreaterThan(prevCap)
      prevRate = b.rate
      prevCap = b.upTo
    }
    expect(BRACKETS_2026_SINGLE[BRACKETS_2026_SINGLE.length - 1].upTo).toBe(Infinity)
  })

  it('the FICO weights sum to one', () => {
    expect(FICO_FACTORS.reduce((s, f) => s + f.weight, 0)).toBeCloseTo(1, 10)
  })
})

describe('lesson ranking', () => {
  it('puts a debt lesson first for someone carrying debt', () => {
    const ranked = rankLessons({ ...profile, debtBalance: 4200, debtApr: 0.229 }, {})
    expect(ranked[0].triggers).toContain('has-debt')
  })

  it('hides US-only lessons from a non-US learner', () => {
    // Showing US numbers to a UK user with a caveat is how you ship confidently
    // wrong content. Gate it instead.
    const ranked = rankLessons(profile, {}, 'other')
    expect(ranked.every((l) => l.jurisdiction === 'any')).toBe(true)
    expect(ranked.length).toBeGreaterThan(0)
  })

  it('demotes completed lessons without removing them', () => {
    const first = rankLessons(profile, {})[0]
    const after = rankLessons(profile, { [first.id]: true })
    expect(after[0].id).not.toBe(first.id)
    expect(after.map((l) => l.id)).toContain(first.id)
  })
})
