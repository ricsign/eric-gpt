/**
 * Review scheduling.
 *
 * A half-life model in the FSRS family rather than SM-2's ease-factor ladder.
 * The reasons matter for this app specifically:
 *
 *  - It schedules on *recall probability*, so "due" means "you are about to forget
 *    this", not "an interval elapsed". That is the right trigger for money concepts
 *    that must survive until the next real decision, months away.
 *  - It degrades gracefully with very few reviews, which is the regime this app
 *    lives in. Nobody is doing 200 finance cards a day.
 *  - Stability and difficulty are separate, so a concept that is intrinsically hard
 *    (sequence-of-returns risk) and one that is merely new (what an ETF is) do not
 *    get the same curve.
 *
 * Deliberately *not* full FSRS-6: its 21 fitted parameters need a large review
 * corpus to be worth anything, and with none to fit against they would be false
 * precision. This is the same shape with hand-set constants.
 */

/** How the learner did. Mapped from the UI's answer outcomes. */
export type Grade = 'forgot' | 'hard' | 'good' | 'easy'

export interface ReviewState {
  /** Days at which recall probability decays to RETENTION_TARGET. Higher = better known. */
  stability: number
  /** 1 (trivial) to 10 (brutal). Rises on lapses, falls slowly on easy recalls. */
  difficulty: number
  /** Total successful reviews. */
  reps: number
  /** Times it was forgotten after having been learned. */
  lapses: number
  /** Day key (YYYY-MM-DD) of the last review. */
  lastReview: string
  /** Day key this becomes due. */
  due: string
}

/**
 * The recall probability we schedule at. 0.9 is the FSRS default and a good
 * trade-off: lower means fewer reviews but more forgetting, higher means a
 * punishing queue. For a habit app, a punishing queue is what kills the habit.
 */
const RETENTION_TARGET = 0.9

/** Starting stability in days, by first-answer grade. */
const INITIAL_STABILITY: Record<Grade, number> = {
  forgot: 0.4,
  hard: 1.2,
  good: 3.2,
  easy: 8,
}

/** Starting difficulty, by first-answer grade. */
const INITIAL_DIFFICULTY: Record<Grade, number> = {
  forgot: 7.5,
  hard: 6.2,
  good: 5,
  easy: 3.5,
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))

const addDays = (dayKey: string, days: number): string => {
  const [y, m, d] = dayKey.split('-').map(Number)
  const date = new Date(Date.UTC(y, m - 1, d))
  date.setUTCDate(date.getUTCDate() + Math.max(1, Math.round(days)))
  return date.toISOString().slice(0, 10)
}

const daysApart = (a: string, b: string): number => {
  const toUtc = (s: string) => {
    const [y, m, d] = s.split('-').map(Number)
    return Date.UTC(y, m - 1, d)
  }
  return Math.max(0, Math.round((toUtc(b) - toUtc(a)) / 86_400_000))
}

/**
 * Probability the learner still recalls this, `elapsed` days after the last review.
 *
 * The power-law forgetting curve from the FSRS work, which fits human data
 * considerably better than the exponential curve SM-2 assumes — real memory has a
 * much fatter tail than exponential decay predicts.
 */
export function recallProbability(stability: number, elapsedDays: number): number {
  if (stability <= 0) return 0
  return Math.pow(1 + (19 / 81) * (elapsedDays / stability), -0.5)
}

/** First encounter with a concept. */
export function initialReview(grade: Grade, today: string): ReviewState {
  const stability = INITIAL_STABILITY[grade]
  return {
    stability,
    difficulty: INITIAL_DIFFICULTY[grade],
    reps: grade === 'forgot' ? 0 : 1,
    lapses: 0,
    lastReview: today,
    due: addDays(today, intervalFor(stability)),
  }
}

/** Days until recall probability falls to the retention target. */
function intervalFor(stability: number): number {
  // Inverse of recallProbability, solved for elapsed.
  return (stability * 81) / 19 * (Math.pow(RETENTION_TARGET, -2) - 1)
}

/**
 * Updates a concept's memory state after a review.
 *
 * The key behaviour: how much stability grows depends on how close the learner was
 * to forgetting. Reviewing something you already knew cold barely helps; reviewing
 * something you *just* managed to retrieve helps enormously. That is the spacing
 * effect, and scheduling on it is the whole point.
 */
export function review(state: ReviewState, grade: Grade, today: string): ReviewState {
  const elapsed = daysApart(state.lastReview, today)
  const retrievability = recallProbability(state.stability, elapsed)

  const difficulty = clamp(
    state.difficulty +
      { forgot: 1.6, hard: 0.6, good: -0.1, easy: -0.8 }[grade],
    1,
    10,
  )

  let stability: number
  if (grade === 'forgot') {
    // A lapse does not reset to zero — relearning is faster than learning. The
    // floor keeps a repeatedly-failed concept coming back tomorrow, not in a week.
    stability = Math.max(0.4, state.stability * 0.35 * Math.exp(-0.1 * state.lapses))
  } else {
    // Growth is damped by difficulty and amplified when retrievability was low.
    const gradeBonus = { hard: 0.6, good: 1, easy: 1.45 }[grade]
    const difficultyPenalty = 11 - difficulty
    const spacingBonus = Math.exp(1.2 * (1 - retrievability))
    const growth = 1 + (difficultyPenalty / 10) * gradeBonus * spacingBonus * 1.3
    stability = state.stability * growth
  }

  // A year is long enough for anything in this curriculum; beyond that, the app
  // should be re-teaching in context rather than quizzing.
  stability = clamp(stability, 0.4, 365)

  return {
    stability,
    difficulty,
    reps: grade === 'forgot' ? state.reps : state.reps + 1,
    lapses: grade === 'forgot' ? state.lapses + 1 : state.lapses,
    lastReview: today,
    due: addDays(today, intervalFor(stability)),
  }
}

/** True when this concept is at or past its due day. */
export function isDue(state: ReviewState, today: string): boolean {
  return state.due <= today
}

/**
 * Orders the review queue.
 *
 * Most-overdue first, with ties broken by lower recall probability, so the
 * concepts closest to being lost are rescued first. Capping the queue is the
 * caller's job — a queue of 60 is a queue nobody starts.
 */
export function dueQueue<T extends { review: ReviewState }>(items: T[], today: string): T[] {
  return items
    .filter((i) => isDue(i.review, today))
    .sort((a, b) => {
      const aOver = daysApart(a.review.due, today)
      const bOver = daysApart(b.review.due, today)
      if (aOver !== bOver) return bOver - aOver
      return (
        recallProbability(a.review.stability, daysApart(a.review.lastReview, today)) -
        recallProbability(b.review.stability, daysApart(b.review.lastReview, today))
      )
    })
}

/* ------------------------------------------------------------------------- *
 * Procedural review
 * ------------------------------------------------------------------------- */

/**
 * A second, deliberately different scheduler for *judgments* rather than facts.
 *
 * Item-level memory models like the one above schedule a specific card. That is
 * right for "what does APY mean" and wrong for "which of these debts do you pay
 * first" — the second is a procedure that has to transfer to situations the learner
 * has never seen, and a model that tracks recall of one phrasing does not measure
 * transfer at all.
 *
 * So procedures get a fixed expanding ladder with a *different surface scenario at
 * every rung*. The gaps stretch to months on purpose: the spacing literature is
 * clear that year-scale retention needs month-scale final gaps, and a ladder that
 * terminates at two weeks quietly guarantees the knowledge is gone by the time the
 * real decision arrives.
 */
export const PROCEDURE_LADDER = [1, 3, 10, 30, 90, 270] as const

export interface ProcedureState {
  /** Index into PROCEDURE_LADDER. */
  rung: number
  /** Scenario ids already used, so a repeat never reuses a surface. */
  seenScenarios: string[]
  lastReview: string
  due: string
}

const addDaysTo = (dayKey: string, days: number): string => {
  const [y, m, d] = dayKey.split('-').map(Number)
  const date = new Date(Date.UTC(y, m - 1, d))
  date.setUTCDate(date.getUTCDate() + Math.max(1, Math.round(days)))
  return date.toISOString().slice(0, 10)
}

export function initialProcedure(scenarioId: string, today: string): ProcedureState {
  return {
    rung: 0,
    seenScenarios: [scenarioId],
    lastReview: today,
    due: addDaysTo(today, PROCEDURE_LADDER[0]),
  }
}

/**
 * Advances or drops a rung.
 *
 * A miss steps back one rung rather than resetting: the learner demonstrably has
 * *some* of the procedure, and restarting a 270-day ladder from day 1 over a single
 * slip is both demotivating and unsupported by the evidence.
 */
export function reviewProcedure(
  state: ProcedureState,
  correct: boolean,
  scenarioId: string,
  today: string,
): ProcedureState {
  const rung = correct
    ? Math.min(PROCEDURE_LADDER.length - 1, state.rung + 1)
    : Math.max(0, state.rung - 1)

  return {
    rung,
    seenScenarios: state.seenScenarios.includes(scenarioId)
      ? state.seenScenarios
      : [...state.seenScenarios, scenarioId],
    lastReview: today,
    due: addDaysTo(today, PROCEDURE_LADDER[rung]),
  }
}

/**
 * Picks the next surface scenario for a procedure, preferring one the learner has
 * never seen. Falls back to the least recently used once the pool is exhausted —
 * a repeated scenario still beats skipping the review.
 */
export function nextScenario(state: ProcedureState, pool: string[]): string | null {
  if (pool.length === 0) return null
  const unseen = pool.filter((s) => !state.seenScenarios.includes(s))
  if (unseen.length > 0) return unseen[0]
  return pool[state.seenScenarios.length % pool.length]
}
