/**
 * Which call is today's, and when tomorrow's arrives.
 *
 * The day turns over at 06:00 local, not midnight. That is a product decision:
 * someone playing at 00:30 is finishing their evening, not starting a new day,
 * and handing them a fresh call at midnight breaks the "one a day" contract for
 * every night owl. The countdown on the receipt has to agree with this, or the
 * app appears to lie about when the next call lands.
 *
 * Everything here is pure and takes an explicit `now`, so the rotation is
 * testable without mocking the clock.
 */

/** The hour at which a new call unlocks, in the player's own timezone. */
export const ROLLOVER_HOUR = 6

/** Day 1. Fixed forever — call numbers are derived from it. */
export const EPOCH = '2026-09-01'

/** The local calendar day a given instant belongs to, accounting for rollover. */
export function compoundDay(now: Date = new Date()): string {
  const d = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  // Before the rollover hour the player is still on yesterday's call.
  if (now.getHours() < ROLLOVER_HOUR) d.setDate(d.getDate() - 1)

  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

/** Whole days between two day keys. */
function daysBetween(from: string, to: string): number {
  const utc = (s: string) => {
    const [y, m, d] = s.split('-').map(Number)
    return Date.UTC(y, m - 1, d)
  }
  return Math.round((utc(to) - utc(from)) / 86_400_000)
}

/**
 * The call number for a given day. Starts at 1 on the epoch and never repeats a
 * number, so "No.142" is a stable, shareable identifier forever.
 */
export function callNumber(day: string = compoundDay()): number {
  return daysBetween(EPOCH, day) + 1
}

/**
 * Which record in the library that number maps to.
 *
 * The library is smaller than the number of days that will pass, so it cycles.
 * Multiplying by a value coprime to the library size walks the whole library in
 * a scrambled order rather than marching through it 1,2,3 — which matters
 * because a predictable order lets someone read ahead, and because a plain
 * modulo would land the same call on the same weekday forever.
 */
export function callIndex(number: number, librarySize: number): number {
  if (librarySize <= 0) return 0
  const cycle = Math.floor((number - 1) / librarySize)
  const withinCycle = (number - 1) % librarySize
  // First pass through the library is in authored order: calls 1-10 are a
  // deliberate onboarding sequence and must not be shuffled. Later cycles are
  // offset so the sequence does not simply repeat.
  if (cycle === 0) return withinCycle
  return (withinCycle * 7 + cycle * 3) % librarySize
}

/** Milliseconds until the next call unlocks. */
export function msUntilNextCall(now: Date = new Date()): number {
  const next = new Date(now.getFullYear(), now.getMonth(), now.getDate(), ROLLOVER_HOUR, 0, 0, 0)
  if (now.getHours() >= ROLLOVER_HOUR) next.setDate(next.getDate() + 1)
  return next.getTime() - now.getTime()
}

/** `07:42:13`, for the countdown. Always three segments so it never reflows. */
export function formatCountdown(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':')
}

/** `SEP 13`, for the receipt. */
export function formatCallDate(day: string): string {
  const [y, m, d] = day.split('-').map(Number)
  return new Date(y, m - 1, d)
    .toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    .toUpperCase()
}

/**
 * The `count` compound days ending at `day`, oldest first.
 *
 * Used by the Tab's attendance strip. Dates are stepped in UTC deliberately:
 * a compound day is already a local-calendar key by the time it gets here, so
 * treating it as UTC midnight makes the arithmetic immune to the daylight-
 * saving shift that would otherwise drop or repeat a day twice a year.
 */
export function recentDays(count: number, day: string = compoundDay()): string[] {
  const [y, m, d] = day.split('-').map(Number)
  const end = Date.UTC(y, m - 1, d)
  return Array.from({ length: count }, (_, i) => {
    const t = new Date(end - (count - 1 - i) * 86_400_000)
    return `${t.getUTCFullYear()}-${String(t.getUTCMonth() + 1).padStart(2, '0')}-${String(
      t.getUTCDate(),
    ).padStart(2, '0')}`
  })
}

/**
 * Whether the set releases one question a day, or all at once.
 *
 * Kept as one constant because it is a live product disagreement, not a
 * settled fact. The pitch is "ten questions, one a day", and the daily rhythm
 * is what gives the product a reason to be opened tomorrow. Against that, all
 * three test readers objected to it unprompted — the content is already
 * written, they can tell, and one said plainly: "give me a reason for the
 * pacing or give me the ten." Flip this to false and the gate is gone.
 */
export const ONE_A_DAY = true

/**
 * The next question this player should answer.
 *
 * Player-relative, and that is the whole point. The previous scheme indexed
 * the library by the global day: authored order for the first ten days after
 * EPOCH, a shuffling stride forever after. Those ten days elapsed on
 * 2026-09-10, so from the eleventh day onward a brand-new player's first ever
 * screen was whichever question the stride happened to land on — in practice
 * a dial asking for fund fees on a portfolio they do not have. The onboarding
 * sequence the library was ordered for became unreachable for exactly the
 * people it was written for.
 *
 * Now the queue follows the player: the first question in authored order they
 * have not answered for real. Everyone starts at question one, whenever they
 * arrive.
 */
export function nextQuestionId(
  orderedIds: number[],
  answered: Set<number>,
): number | null {
  for (const id of orderedIds) if (!answered.has(id)) return id
  return null
}

/**
 * Whether today's question is still waiting, given the days already spent.
 *
 * A question answered earlier today closes the set until the next rollover.
 * Practice — replaying an answered question, or following a shared link —
 * never counts, so it can never consume a day.
 */
export function unlockedToday(answeredDays: string[], now: Date = new Date()): boolean {
  if (!ONE_A_DAY) return true
  return !answeredDays.includes(compoundDay(now))
}
