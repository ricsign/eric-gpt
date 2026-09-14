import { describe, expect, it } from 'vitest'
import { COMPUTE, SCENARIO } from './compute'
import { CALLS } from './registry'
import { DEFAULT_PROFILE, judge, optimalValue, resolveOptimal, stepCount, type Profile } from './types'

/** The best reachable position inside a call's optimal window. */
function bestInOptimal(
  call: (typeof CALLS)[number],
  profile: Profile,
  score: (v: number) => number,
): number {
  const o = resolveOptimal(call.optimal, profile)
  const inside = sweep(call).filter((v) =>
    typeof o === 'number' ? v === o : v >= o.min && v <= o.max,
  )
  // A range wider than one step has several correct answers; the player is
  // entitled to the best of them, not the midpoint.
  return inside.reduce((best, v) => (score(v) > score(best) ? v : best), inside[0])
}

/**
 * The seam tests.
 *
 * Every other test file checks one module against its own intentions. These
 * check the modules against *each other*, which is where the real bugs live: a
 * call record that says the two cards hold $2,400 and $900 while its compute
 * function models $4,200 and $2,800 will pass both of those files' own tests and
 * still show the player numbers that do not add up.
 *
 * If any of these fail, the app is teaching something false. They are the most
 * important tests in the codebase.
 */

/** A spread of players, so nothing passes only at the default salary. */
const PROFILES: Profile[] = [
  { salary: 28_000, age: 22 },
  { salary: 62_000, age: 30 },
  { salary: 145_000, age: 41 },
  { salary: 320_000, age: 55 },
]

/** Every reachable position on a call's control. */
function sweep(call: (typeof CALLS)[number]): number[] {
  const { min, max, step } = call.variable
  const out: number[] = []
  for (let v = min; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step)
  return out
}

describe('every call resolves to a real compute function', () => {
  it('no call references a function that does not exist', () => {
    for (const call of CALLS) {
      expect(COMPUTE[call.compute], `call ${call.id} -> ${call.compute}`).toBeTypeOf('function')
    }
  })

  it('no compute function is orphaned', () => {
    // An unused compute function is either a call that was cut without cleanup
    // or a call record pointing at the wrong name.
    const used = new Set(CALLS.map((c) => c.compute))
    for (const name of Object.keys(COMPUTE)) {
      expect(used.has(name), `${name} is never used by any call`).toBe(true)
    }
  })
})

describe('the declared optimal really is the best play', () => {
  it.each(CALLS.map((c) => [c.id, c.title, c] as const))(
    'call %i — %s',
    (_id, _title, call) => {
      for (const profile of PROFILES) {
        const compute = COMPUTE[call.compute]
        const positions = sweep(call)

        // The best reachable position by end value at 65.
        let bestValue = positions[0]
        let bestAt65 = -Infinity
        for (const v of positions) {
          const at65 = compute(v, profile).at65
          if (at65 > bestAt65) {
            bestAt65 = at65
            bestValue = v
          }
        }

        // The call's own claim must agree with its own maths. If it does not,
        // the outcome screen calls a correct answer "money left behind".
        expect(
          judge(bestValue, call.optimal, profile),
          `call ${call.id} at salary ${profile.salary}: maths peaks at ${bestValue}, ` +
            `but the record claims ${JSON.stringify(resolveOptimal(call.optimal, profile))}`,
        ).toBe('optimal')
      }
    },
  )
})

describe('no call can produce a broken number', () => {
  it.each(CALLS.map((c) => [c.id, c] as const))('call %i is finite everywhere', (_id, call) => {
    const compute = COMPUTE[call.compute]
    for (const profile of PROFILES) {
      for (const v of sweep(call)) {
        const o = compute(v, profile)
        for (const [k, n] of Object.entries({ cost: o.cost, benefit: o.benefit, at65: o.at65 })) {
          expect(Number.isFinite(n), `call ${call.id} @ ${v} salary ${profile.salary}: ${k}=${n}`).toBe(
            true,
          )
        }
        // A negative monthly cost would render as "-$120/mo", which reads as the
        // app paying the player.
        expect(o.cost, `call ${call.id} @ ${v}`).toBeGreaterThanOrEqual(0)
      }
    }
  })
})

describe('the breakdown is receipt-ready at every position', () => {
  it.each(CALLS.map((c) => [c.id, c] as const))('call %i', (_id, call) => {
    const compute = COMPUTE[call.compute]
    for (const v of sweep(call)) {
      const { breakdown } = compute(v, { salary: 62_000, age: 30 })

      // The receipt has room for five or six lines and no more.
      expect(breakdown.length, `call ${call.id} @ ${v}`).toBeGreaterThanOrEqual(3)
      expect(breakdown.length, `call ${call.id} @ ${v}`).toBeLessThanOrEqual(6)

      // Exactly one line carries the weight, so the receipt knows what to stress.
      expect(
        breakdown.filter((l) => l.emphasis).length,
        `call ${call.id} @ ${v} emphasis count`,
      ).toBe(1)

      for (const line of breakdown) {
        expect(line.label.trim().length, `call ${call.id} @ ${v}`).toBeGreaterThan(0)
        expect(String(line.value).trim().length, `call ${call.id} @ ${v}`).toBeGreaterThan(0)
        // Labels are set in a narrow monospace column on the receipt.
        expect(line.label.length, `call ${call.id}: "${line.label}" is too long`).toBeLessThanOrEqual(
          28,
        )
      }
    }
  })
})

describe('the control agrees with the histogram', () => {
  it('every crowd array is exactly as long as the control has positions', () => {
    // Drift here silently misaligns every bar in the histogram against the
    // value it claims to represent — the chart still renders, it just lies.
    for (const call of CALLS) {
      expect(call.crowd.length, `call ${call.id}`).toBe(stepCount(call.variable))
    }
  })

  it('the sweep and stepCount agree', () => {
    for (const call of CALLS) {
      expect(sweep(call).length, `call ${call.id}`).toBe(stepCount(call.variable))
    }
  })
})

describe('the control starts somewhere that teaches', () => {
  it('never starts on the optimal answer', () => {
    // If the control opened on the right answer the player would never have to
    // move it, and the gesture — which is the entire product — would teach
    // nothing. Checked at every profile, because a profile-dependent optimum
    // could drift onto the start position for some players and not others.
    for (const call of CALLS) {
      for (const profile of PROFILES) {
        expect(
          judge(call.variable.start, call.optimal, profile),
          `call ${call.id} at salary ${profile.salary}`,
        ).not.toBe('optimal')
      }
    }
  })

  it('starts on a reachable step', () => {
    for (const call of CALLS) {
      const { min, max, step, start } = call.variable
      expect(start).toBeGreaterThanOrEqual(min)
      expect(start).toBeLessThanOrEqual(max)
      expect(Math.abs((start - min) / step - Math.round((start - min) / step))).toBeLessThan(1e-9)
    }
  })
})

describe('moving the control actually changes something', () => {
  it.each(CALLS.map((c) => [c.id, c] as const))('call %i is not a flat line', (_id, call) => {
    // A call whose numbers do not move as you drag is an article wearing a
    // control, and it does not ship.
    const compute = COMPUTE[call.compute]
    const values = sweep(call).map((v) => compute(v, { salary: 62_000, age: 30 }).at65)
    const spread = Math.max(...values) - Math.min(...values)
    expect(spread, `call ${call.id} at65 spread`).toBeGreaterThan(1)
  })
})

describe('the optimal play beats the starting position', () => {
  it.each(CALLS.map((c) => [c.id, c] as const))('call %i rewards moving', (_id, call) => {
    // Every call must leave the player better off than where they started, or
    // locking in without touching anything would be as good as playing.
    const compute = COMPUTE[call.compute]
    for (const profile of PROFILES) {
      const score = (v: number) => compute(v, profile).at65
      const atStart = score(call.variable.start)
      // Against the best answer available inside the optimal window, not its
      // midpoint — for a wide range the midpoint can genuinely be worse than a
      // start position sitting near one of the edges.
      const atBest = score(bestInOptimal(call, profile, score))
      expect(atBest, `call ${call.id} at salary ${profile.salary}`).toBeGreaterThan(atStart)
    }
  })
})

describe('the card states the scenario the maths actually uses', () => {
  /**
   * Pulls every figure out of a card row: "$4,200 AT 24.99%" -> [4200, 24.99].
   * The `{{salary}}` token carries no digits, so it falls out on its own.
   */
  const numbers = (text: string): number[] =>
    (text.match(/\d[\d,]*(?:\.\d+)?/g) ?? []).map((n) => Number(n.replace(/,/g, '')))

  // A card rounds to the dollar and to two decimal places; the model does not.
  const has = (pool: number[], n: number) =>
    pool.some((p) => Math.abs(p - n) <= Math.max(0.51, Math.abs(p) * 0.001))

  it.each(CALLS.map((c) => [c.id, c.title, c] as const))(
    'call %i — %s',
    (_id, _title, call) => {
      const pool = SCENARIO[call.compute]
      expect(pool, `no SCENARIO entry for ${call.compute}`).toBeDefined()
      for (const row of call.fixed) {
        for (const n of numbers(row.v)) {
          expect(
            has(pool, n),
            `card row ${row.k} says ${n}, which ${call.compute} does not model ` +
              `(it models ${pool.join(', ')})`,
          ).toBe(true)
        }
      }
    },
  )
})

describe('every question says what it is asking, and what to do about it', () => {
  // The two fields added when six reviewers independently reported that they
  // could not tell what the screen wanted from them. A headline, a bar and a
  // number is not a question, and an answer with no errand attached is
  // trivia. These guard the authoring rules that make them useful.

  it.each(CALLS.map((c) => [c.id, c] as const))('question %i is answerable by its own dial', (_id, call) => {
    expect(call.question.trim().endsWith('?'), call.question).toBe(true)
    // Second person. A question about "the player" or "a saver" is a case
    // study; this product only works if it is about the person holding it.
    expect(/\b(you|your)\b/i.test(call.question), call.question).toBe(true)
    // Short enough to read in one pass under a headline.
    expect(call.question.length, call.question).toBeLessThanOrEqual(70)
    // It must not contain the answer.
    const answer = String(optimalValue(call.optimal, DEFAULT_PROFILE))
    expect(call.question.includes(answer) && answer.length > 1).toBe(false)
  })

  it.each(CALLS.map((c) => [c.id, c] as const))('action %i is an errand, not a principle', (_id, call) => {
    const first = call.action.split(/\s+/)[0].toLowerCase().replace(/[^a-z]/g, '')
    // An instruction opens on a verb. "You should consider reviewing..." is
    // the register this is written to avoid.
    expect(DOING_VERBS.has(first), `action ${call.id} opens on "${first}"`).toBe(true)
    // Long enough to be specific, short enough to do in an evening.
    expect(call.action.length).toBeGreaterThan(40)
    expect(call.action.length).toBeLessThanOrEqual(200)
    // The rule is the idea and the action is the errand; if they are the same
    // sentence, one of them is not doing its job.
    expect(call.action).not.toBe(call.rule)
  })

  it('never speaks the vocabulary that made the old build unreadable', () => {
    // Every one of these was flagged by the plain-language review as a term a
    // normal adult would not know or would misread. They are banned from all
    // player-facing strings, which is everything on a Call record except the
    // compute key.
    const banned = /\b(bps|basis points?|safe harbou?r|optimal|overshot|the tab|expense ratio)\b/i
    for (const call of CALLS) {
      for (const [what, text] of [
        ['title', call.title],
        ['question', call.question],
        ['rule', call.rule],
        ['action', call.action],
        ['label', call.variable.label],
        ['assumptions', call.assumptions],
        ...call.fixed.map((f) => [`fixed ${f.k}`, `${f.k} ${f.v}`] as const),
      ] as const) {
        const hit = text.match(banned)
        expect(hit?.[0], `call ${call.id} ${what}: "${text}"`).toBeUndefined()
      }
    }
  })
})

/** Verbs an instruction may open on. Anything else is advice, not an errand. */
const DOING_VERBS = new Set([
  'add', 'ask', 'bank', 'book', 'call', 'cancel', 'change', 'check', 'choose',
  'clear', 'compare', 'count', 'divide', 'find', 'fix', 'get', 'give', 'log',
  'look', 'make', 'move', 'open', 'pay', 'practise', 'put', 'read', 'search',
  'set', 'sign', 'split', 'start', 'stop', 'switch', 'take', 'tell', 'work',
  'write',
])

describe('nothing the maths renders speaks the old vocabulary', () => {
  // The record's own strings are guarded above, but breakdown lines are built
  // at runtime out of template literals, and one of them was emitting "126 BP"
  // onto the reveal. It was invisible for a while only because an unrelated
  // rule on the reveal happened to drop that row — a side effect, not a guard.
  // This checks what a player can actually read, at every position of every
  // dial, which is the only place the check is worth anything.
  const BANNED = /\b(bps|bp|basis points?|safe harbou?r|overshot|optimal|the tab|expense ratio)\b/i

  it.each(CALLS.map((c) => [c.id, c] as const))('question %i renders clean copy', (_id, call) => {
    const compute = COMPUTE[call.compute]
    for (const profile of PROFILES) {
      for (let i = 0; i < stepCount(call.variable); i++) {
        const value = call.variable.min + i * call.variable.step
        for (const l of compute(value, profile).breakdown) {
          const text = `${l.label} ${l.value}`
          const hit = text.match(BANNED)
          expect(
            hit?.[0],
            `question ${call.id} at ${value} (salary ${profile.salary}) renders "${text}"`,
          ).toBeUndefined()
        }
      }
    }
  })
})
