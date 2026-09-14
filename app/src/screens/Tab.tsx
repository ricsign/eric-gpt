/*
 * `tabSummary` is exported from this file rather than a sibling because it is
 * the definition of the number this screen exists to show, and the Tomorrow
 * screen reads the same function. It is pure, so the tests exercise it directly.
 */
/* eslint-disable react/only-export-components */
import { useMemo } from 'react'
import { callNumber, compoundDay, formatCallDate, recentDays } from '../lib/schedule'
import { daysBetween, moneyCompact } from '../lib/format'
import type { CallResult, Profile, Verdict } from '../calls/types'
import './Tab.css'

/** Same horizon every compute function projects to. */
const RETIRE_AT = 65

const VERDICT_LABEL: Record<Verdict, string> = {
  optimal: 'Optimal',
  short: 'Short',
  over: 'Over',
}

export interface TabSummary {
  /** What the player has actually banked at 65. Never falls. */
  total: number
  /** Calls answered for real. Practice runs are not counted. */
  played: number
  /** How many were the best play. */
  optimal: number
  /** 0-1. Share of calls played optimally. */
  ratio: number
  /** Consecutive days played, counted back from the most recent one. */
  streak: number
  /** What the misses cost at 65, as a positive number. */
  missed: number
  /** 0-1. Of everything the best plays were worth, the share actually taken. */
  capture: number
}

/**
 * The identity number.
 *
 * The Tab only moves on a good decision. A fumbled call adds nothing and takes
 * nothing away, which is the one piece of softening in the whole product and it
 * is deliberate: a running total that can fall is a number people stop opening,
 * and the sting of a bad call has already been delivered, in red, on the
 * outcome screen. What a miss costs is carried separately as `missed`, so the
 * screen can still show it without letting it eat the headline.
 */
export function tabSummary(all: CallResult[]): TabSummary {
  // Practice is dropped here rather than trusted to every caller. A replay of a
  // call the player already closed, or one opened from a shared link, must not
  // move this number — and "the caller filters it" is a comment, not a
  // guarantee: the one time it is forgotten, the Tab silently inflates and the
  // only number the product asks to be believed is wrong.
  const results = all.filter((r) => !r.practice)

  let total = 0
  let optimal = 0
  let missed = 0
  let captured = 0
  let available = 0

  for (const r of results) {
    if (r.verdict === 'optimal') {
      total += r.at65
      optimal += 1
    } else {
      // delta is signed: overshooting can project to *more* at 65 while still
      // being the wrong call, so only genuine shortfalls are counted as cost.
      missed += Math.max(0, -r.delta)
    }
    captured += r.at65
    // delta = this play minus the best play, so the best play is at65 - delta.
    available += r.at65 - r.delta
  }

  const played = results.length

  return {
    total,
    played,
    optimal,
    ratio: played > 0 ? optimal / played : 0,
    streak: streakOf(results),
    missed,
    // Capped at 1: an "over" call can be worth more at 65 than the best play
    // and still be wrong, and a capture rate above 100% would read as a score.
    capture: available > 0 ? Math.min(1, captured / available) : 0,
  }
}

/**
 * Consecutive days played, counted back from the most recent day.
 *
 * Deliberately not measured against today: this is pure, and a streak that
 * silently reset at 6am would make the number depend on when the screen was
 * opened rather than on what the player did. A missed day ends the streak the
 * moment the next call is answered, which is when it is true.
 */
function streakOf(results: CallResult[]): number {
  const days = [...new Set(results.map((r) => r.day))].sort().reverse()
  if (days.length === 0) return 0

  let streak = 1
  for (let i = 1; i < days.length; i++) {
    if (daysBetween(days[i], days[i - 1]) !== 1) break
    streak += 1
  }
  return streak
}

/**
 * The Tab.
 *
 * One hero number and the ledger that produced it. Every row is a call the
 * player closed, so a year of this is the Wrapped: a receipt for the year,
 * built from receipts.
 */
export function Tab({
  results,
  profile,
  onClose,
}: {
  /** Answers, oldest first. Practice runs are ignored, wherever they are filtered. */
  results: CallResult[]
  profile: Profile
  onClose: () => void
}) {
  const summary = useMemo(() => tabSummary(results), [results])
  // Newest first: the last thing they did is the thing they came to see. Same
  // filter as the summary, or the ledger would show rows the hero number did
  // not count and the screen would appear to have lost money.
  const rows = useMemo(() => results.filter((r) => !r.practice).reverse(), [results])
  const years = Math.max(0, RETIRE_AT - profile.age)

  // Attendance, not score. A fortnight is long enough that a gap is visible and
  // short enough that it stays a strip rather than a chart — and on day one it
  // is thirteen empty slots and one filled, which reads as a row with room in
  // it rather than as a screen that failed to load.
  const attendance = useMemo(() => {
    const played = new Set(results.filter((r) => !r.practice).map((r) => r.day))
    const today = compoundDay()
    return recentDays(14).map((day) => ({
      day,
      played: played.has(day),
      today: day === today,
    }))
  }, [results])

  return (
    <div className="tab scroll">
      <header className="tab-head">
        <h1 className="tab-title">The Tab</h1>
        <button className="tab-close data press" onClick={onClose}>
          Close
        </button>
      </header>

      <p className="data-sm">Banked at 65</p>
      {/* Accent means money captured. An empty tab in acid lime would be the
          product congratulating someone for nothing. */}
      <p className="tab-hero num" data-zero={summary.total === 0 ? '' : undefined}>
        {moneyCompact(summary.total)}
      </p>
      {/* Before the first call there is nothing true to say here, and 0/0 in a
          tile reads as a score of zero rather than as an empty ledger. */}
      {summary.played > 0 && (
        <>
          <p className="tab-sub data-sm num">
            {summary.optimal} optimal {summary.optimal === 1 ? 'call' : 'calls'} · {years} years to
            65
          </p>

          <div className="tile-row tab-stats">
            <div className="tile">
              <p className="tile-k">Streak</p>
              <p className="tile-v num">{summary.streak}</p>
            </div>
            <div className="tile">
              <p className="tile-k">Optimal</p>
              <p className="tile-v num">
                {summary.optimal}/{summary.played}
              </p>
            </div>
            <div className="tile">
              <p className="tile-k">Captured</p>
              <p className="tile-v num">{Math.round(summary.capture * 100)}%</p>
            </div>
          </div>
        </>
      )}

      {summary.missed > 0 && (
        <p className="tab-missed data-sm num">
          {moneyCompact(summary.missed)} left on the table
        </p>
      )}

      <section className="tab-attendance">
        <p className="data-sm">Last 14 days</p>
        <ol className="tab-days">
          {attendance.map((d) => (
            <li
              className="tab-day"
              key={d.day}
              data-played={d.played ? '' : undefined}
              data-today={d.today ? '' : undefined}
            />
          ))}
        </ol>
      </section>

      {rows.length === 0 ? (
        <p className="tab-empty">Lock in a call and it lands here.</p>
      ) : (
        <ol className="tab-rows">
          {rows.map((r, i) => (
            <li className="tab-row" key={`${r.day}-${r.callId}-${i}`}>
              <span className="tab-no data-sm num">No.{callNumber(r.day)}</span>
              <span className="tab-date data-sm num">{formatCallDate(r.day)}</span>
              <span className="tab-chip" data-v={r.verdict}>
                {VERDICT_LABEL[r.verdict]}
              </span>
              <span className="tab-delta num" data-v={r.verdict}>
                {r.verdict === 'optimal'
                  ? `+${moneyCompact(r.at65)}`
                  : `${r.delta < 0 ? '−' : '+'}${moneyCompact(Math.abs(r.delta))}`}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
