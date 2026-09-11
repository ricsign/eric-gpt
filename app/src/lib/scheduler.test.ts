import { describe, expect, it } from 'vitest'
import {
  dueQueue,
  initialProcedure,
  initialReview,
  isDue,
  nextScenario,
  PROCEDURE_LADDER,
  recallProbability,
  review,
  reviewProcedure,
  type ReviewState,
} from './scheduler'

const DAY = '2026-09-11'

/** Advances a day key by n days, mirroring the scheduler's own UTC arithmetic. */
const plus = (key: string, n: number) => {
  const [y, m, d] = key.split('-').map(Number)
  const date = new Date(Date.UTC(y, m - 1, d))
  date.setUTCDate(date.getUTCDate() + n)
  return date.toISOString().slice(0, 10)
}

describe('recallProbability', () => {
  it('is 1 the moment after a review', () => {
    expect(recallProbability(5, 0)).toBe(1)
  })

  it('decays monotonically', () => {
    let prev = 1
    for (const d of [1, 2, 5, 10, 30, 100]) {
      const p = recallProbability(5, d)
      expect(p).toBeLessThan(prev)
      prev = p
    }
  })

  it('is at the 0.9 retention target after exactly one stability-day', () => {
    // Stability is defined as the point where recall hits the target, so this is
    // the invariant the whole schedule rests on.
    expect(recallProbability(10, 10)).toBeCloseTo(0.9, 2)
  })

  it('more stable memories decay more slowly', () => {
    expect(recallProbability(30, 14)).toBeGreaterThan(recallProbability(3, 14))
  })
})

describe('initialReview', () => {
  it('grades the first answer into very different intervals', () => {
    const forgot = initialReview('forgot', DAY)
    const easy = initialReview('easy', DAY)
    expect(easy.stability).toBeGreaterThan(forgot.stability * 10)
    expect(easy.due > forgot.due).toBe(true)
  })

  it('never schedules anything for the same day', () => {
    // A card re-shown within the session teaches recognition, not recall.
    for (const g of ['forgot', 'hard', 'good', 'easy'] as const) {
      expect(initialReview(g, DAY).due > DAY).toBe(true)
    }
  })

  it('a failed first attempt does not count as a rep', () => {
    expect(initialReview('forgot', DAY).reps).toBe(0)
    expect(initialReview('good', DAY).reps).toBe(1)
  })

  it('harder first answers start at a higher difficulty', () => {
    expect(initialReview('forgot', DAY).difficulty).toBeGreaterThan(
      initialReview('easy', DAY).difficulty,
    )
  })
})

describe('review', () => {
  const base = initialReview('good', DAY)

  it('successful reviews grow stability', () => {
    const next = review(base, 'good', base.due)
    expect(next.stability).toBeGreaterThan(base.stability)
    expect(next.reps).toBe(base.reps + 1)
  })

  it('intervals lengthen across a run of successes', () => {
    let s = base
    let prev = 0
    for (let i = 0; i < 6; i++) {
      const gap = Math.round(s.stability)
      s = review(s, 'good', plus(s.lastReview, Math.max(1, gap)))
      const interval = s.stability
      expect(interval).toBeGreaterThan(prev)
      prev = interval
    }
    // Six good reviews should push this out to months, not days.
    expect(s.stability).toBeGreaterThan(60)
  })

  it('rewards a review that happened when recall was already shaky', () => {
    // Same card, same grade — only the delay differs. The late review should
    // produce the bigger gain. This is the spacing effect, and it is the reason
    // to schedule on recall probability at all.
    const early = review(base, 'good', plus(DAY, 1))
    const late = review(base, 'good', plus(DAY, 12))
    expect(late.stability).toBeGreaterThan(early.stability)
  })

  it('a lapse cuts stability but does not reset it', () => {
    let s = base
    for (let i = 0; i < 4; i++) s = review(s, 'good', plus(s.lastReview, 10))
    const before = s.stability
    const after = review(s, 'forgot', plus(s.lastReview, 5))

    expect(after.stability).toBeLessThan(before)
    // Relearning is faster than learning from scratch.
    expect(after.stability).toBeGreaterThan(0)
    expect(after.lapses).toBe(1)
    expect(after.reps).toBe(s.reps)
  })

  it('raises difficulty on failure and lowers it on easy recall', () => {
    expect(review(base, 'forgot', plus(DAY, 3)).difficulty).toBeGreaterThan(base.difficulty)
    expect(review(base, 'easy', plus(DAY, 3)).difficulty).toBeLessThan(base.difficulty)
  })

  it('a harder concept grows stability more slowly than an easy one', () => {
    const hardCard: ReviewState = { ...base, difficulty: 9 }
    const easyCard: ReviewState = { ...base, difficulty: 2 }
    const day = plus(DAY, 4)
    expect(review(easyCard, 'good', day).stability).toBeGreaterThan(
      review(hardCard, 'good', day).stability,
    )
  })

  it('keeps difficulty and stability inside their bounds', () => {
    let s = base
    for (let i = 0; i < 40; i++) s = review(s, 'forgot', plus(s.lastReview, 1))
    expect(s.difficulty).toBeLessThanOrEqual(10)
    expect(s.stability).toBeGreaterThanOrEqual(0.4)

    let t = base
    for (let i = 0; i < 40; i++) t = review(t, 'easy', plus(t.lastReview, 400))
    expect(t.difficulty).toBeGreaterThanOrEqual(1)
    expect(t.stability).toBeLessThanOrEqual(365)
  })

  it('a repeatedly failed concept keeps coming back within days', () => {
    let s = base
    for (let i = 0; i < 5; i++) s = review(s, 'forgot', plus(s.lastReview, 2))
    // Not "in a week" — something being actively forgotten needs to be seen soon.
    expect(new Date(s.due).getTime() - new Date(s.lastReview).getTime()).toBeLessThanOrEqual(
      3 * 86_400_000,
    )
  })
})

describe('the queue', () => {
  const make = (due: string, stability = 5): { id: string; review: ReviewState } => ({
    id: due,
    review: { stability, difficulty: 5, reps: 2, lapses: 0, lastReview: plus(due, -3), due },
  })

  it('isDue is true on the due day and after', () => {
    expect(isDue(make(DAY).review, DAY)).toBe(true)
    expect(isDue(make(plus(DAY, 1)).review, DAY)).toBe(false)
    expect(isDue(make(plus(DAY, -1)).review, DAY)).toBe(true)
  })

  it('returns only due items, most overdue first', () => {
    const q = dueQueue(
      [make(plus(DAY, 2)), make(plus(DAY, -1)), make(plus(DAY, -9)), make(DAY)],
      DAY,
    )
    expect(q.map((i) => i.id)).toEqual([plus(DAY, -9), plus(DAY, -1), DAY])
  })

  it('breaks ties toward the memory closest to being lost', () => {
    const strong = make(DAY, 60)
    const weak = make(DAY, 1)
    const q = dueQueue([strong, weak], DAY)
    expect(q[0]).toBe(weak)
  })

  it('is empty when nothing is due', () => {
    expect(dueQueue([make(plus(DAY, 3))], DAY)).toEqual([])
  })
})

describe('procedural ladder', () => {
  it('starts at the first rung, due tomorrow', () => {
    const p = initialProcedure('card-vs-loan', DAY)
    expect(p.rung).toBe(0)
    expect(p.due).toBe(plus(DAY, PROCEDURE_LADDER[0]))
  })

  it('climbs to month-scale gaps over a run of correct answers', () => {
    let p = initialProcedure('a', DAY)
    for (let i = 0; i < 5; i++) p = reviewProcedure(p, true, `s${i}`, p.due)
    expect(p.rung).toBe(PROCEDURE_LADDER.length - 1)
    // The whole point of the ladder: the final gap is months, not weeks.
    expect(PROCEDURE_LADDER[p.rung]).toBeGreaterThanOrEqual(90)
  })

  it('a miss steps back one rung rather than resetting', () => {
    let p = initialProcedure('a', DAY)
    for (let i = 0; i < 4; i++) p = reviewProcedure(p, true, `s${i}`, p.due)
    const high = p.rung
    p = reviewProcedure(p, false, 'sx', p.due)
    expect(p.rung).toBe(high - 1)
  })

  it('cannot fall below the first rung', () => {
    let p = initialProcedure('a', DAY)
    for (let i = 0; i < 5; i++) p = reviewProcedure(p, false, `s${i}`, p.due)
    expect(p.rung).toBe(0)
  })

  it('records every scenario it has shown, without duplicates', () => {
    let p = initialProcedure('a', DAY)
    p = reviewProcedure(p, true, 'b', p.due)
    p = reviewProcedure(p, true, 'b', p.due)
    expect(p.seenScenarios).toEqual(['a', 'b'])
  })

  it('always offers an unseen surface scenario while one exists', () => {
    const pool = ['a', 'b', 'c']
    let p = initialProcedure('a', DAY)
    expect(nextScenario(p, pool)).toBe('b')
    p = reviewProcedure(p, true, 'b', p.due)
    expect(nextScenario(p, pool)).toBe('c')
  })

  it('recycles rather than returning nothing once the pool is exhausted', () => {
    const pool = ['a', 'b']
    let p = initialProcedure('a', DAY)
    p = reviewProcedure(p, true, 'b', p.due)
    expect(pool).toContain(nextScenario(p, pool))
  })

  it('has nothing to offer from an empty pool', () => {
    expect(nextScenario(initialProcedure('a', DAY), [])).toBeNull()
  })
})
