/*
 * `tabSummary` is exported from this file rather than a sibling because it is
 * the definition of what one answer turned out to be worth, and the suite
 * exercises it directly — nothing in this project can mount React.
 *
 * The screen underneath it no longer renders money at all. What it used to
 * lead with was a projection to 65 under the word "banked", which is a claim
 * no app without an account behind it gets to make, and which on the first day
 * read "$0" — the product congratulating someone for nothing. The arithmetic
 * is still correct and still under test, so it stays; what changed is that a
 * forty-year projection is no longer the thing this screen says about a person.
 */
/* eslint-disable react/only-export-components */
import { useMemo } from 'react'
import { ActionPicker } from '../ui/ActionPicker'
import { daysBetween } from '../lib/format'
import type { ActionState, Call, CallResult } from '../calls/types'
import './Tab.css'

/**
 * What the answers so far added up to.
 *
 * The field names are frozen by the suite rather than chosen: `played`,
 * `optimal` and `capture` are words this product no longer says out loud, and
 * none of them reaches a screen.
 */
export interface TabSummary {
  /** Projected value at 65 of the answers that were right. Never falls. */
  total: number
  /** Answers that counted. Practice runs never do. */
  played: number
  /** How many of them were right. */
  optimal: number
  /** 0-1. Share of the answers that were right. */
  ratio: number
  /** Consecutive days answered, counted back from the most recent one. */
  streak: number
  /** What the misses cost at 65, as a positive number. */
  missed: number
  /** 0-1. Of everything the right answers were worth, the share taken. */
  capture: number
}

/**
 * The running total, and the one piece of softening in the product.
 *
 * A wrong answer adds nothing and takes nothing away. A total that can fall is
 * a number people stop opening, and the cost of a wrong answer has already
 * been shown, once, on the screen where it could be checked against the
 * figures that produced it. What a miss cost is carried separately as
 * `missed`, so it can still be stated without eating anything.
 */
export function tabSummary(all: CallResult[]): TabSummary {
  // Practice is dropped here rather than trusted to every caller. A second run
  // at a question already answered, or one opened from a shared link, must not
  // move this number — and "the caller filters it" is a comment, not a
  // guarantee: the one time it is forgotten the total silently inflates.
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
      // delta is signed: an answer past the right one can project to *more* at
      // 65 while still being wrong, so only genuine shortfalls count as cost.
      missed += Math.max(0, -r.delta)
    }
    captured += r.at65
    // delta = this answer minus the right one, so the right one is at65 - delta.
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
    // Capped at 1: an answer past the right one can be worth more at 65 and
    // still be wrong, and a share above 100% would read as a score.
    capture: available > 0 ? Math.min(1, captured / available) : 0,
  }
}

/**
 * Consecutive days answered, counted back from the most recent day.
 *
 * Nothing renders this any more — a day streak is a retention trick, and days
 * are the wrong unit for a set that ends at ten. It survives because it is
 * pure, correct and covered, and because the schedule is still one a day: the
 * moment anything needs to know whether yesterday was used, this is the answer
 * rather than a second implementation of it.
 *
 * Deliberately not measured against today. A streak that silently reset at 6am
 * would depend on when the screen was opened rather than on what was done.
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
 * My progress.
 *
 * The set is ten questions, so this screen counts questions and errands, both
 * of which have an end. It replaces a hero dollar figure, a day streak and a
 * fourteen-day attendance strip: three different ways of measuring an infinite
 * daily habit, on a product that is finished after ten.
 *
 * The list is the whole set in order, not a ledger of what was answered. A row
 * nobody has reached still says what it will ask, because the same ten
 * questions are already listed on the home screen and hiding them here would
 * only make the page shorter than the thing it is describing.
 */
export function Tab({
  results,
  calls,
  actions,
  onClose,
  onToggleAction,
}: {
  /** Answers, oldest first. Practice runs are dropped here, wherever else they are. */
  results: CallResult[]
  /** The whole set, in the order it is asked in. */
  calls: Call[]
  /** Where each errand stands, by question id. Missing means not started. */
  actions: Record<number, ActionState>
  onClose: () => void
  onToggleAction: (callId: number, state: ActionState) => void
}) {
  // Distinct questions, not results: the denominator is the set, so answering
  // the same question twice must never read as two tenths of it.
  const answered = useMemo(
    () => new Set(results.filter((r) => !r.practice).map((r) => r.callId)),
    [results],
  )

  // Counted across the set rather than over `actions`, so a state left behind
  // by a question that has since left the library cannot inflate the headline.
  const done = useMemo(
    () => calls.filter((c) => answered.has(c.id) && actions[c.id] === 'done').length,
    [calls, actions, answered],
  )

  return (
    <div className="tab scroll">
      <header className="tab-head">
        <h1 className="tab-title">My progress</h1>
        <button className="tab-close data press" onClick={onClose}>
          Close
        </button>
      </header>

      <p className="data-sm">Where you are</p>
      {/* A count, not a currency. It is true on the first day, it means the
          same thing on the last one, and it has a finish line in it. */}
      <p className="tab-hero num">
        {answered.size} of {calls.length}
      </p>
      <p className="tab-note">
        answered
        {/* Only said once there is one. "0 things done" under a screen called
            progress is a scolding, and it is the reader's first day. */}
        {done > 0 && (
          <>
            {' · '}
            <span className="tab-note-done num">
              {done} {done === 1 ? 'thing' : 'things'} done
            </span>
          </>
        )}
      </p>

      {answered.size === 0 && (
        <p className="tab-empty">Answer one and it lands here.</p>
      )}

      <ol className="tab-set">
        {calls.map((call, i) => {
          const isAnswered = answered.has(call.id)
          const state = actions[call.id] ?? 'open'
          return (
            <li className="tab-item" key={call.id} data-answered={isAnswered ? '' : undefined}>
              <p className="tab-q">
                <span className="tab-no data-sm num">{i + 1}</span>
                <span className="tab-q-t">{call.question}</span>
              </p>

              {!isAnswered && <p className="tab-waiting data-sm">Not answered yet</p>}

              {isAnswered && (
                <>
                  {/* The instructions are only here while they are still worth
                      following. Once an errand is finished or set aside the
                      chips say so, and repeating the errand under them would
                      make the page grow as the work shrinks. */}
                  {state === 'open' && <p className="tab-do">{call.action}</p>}
                  <ActionPicker
                    callId={call.id}
                    state={state}
                    label={`The thing to do about question ${i + 1}`}
                    onChange={onToggleAction}
                  />
                </>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
