import { useMemo } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ActionPicker } from '../ui/ActionPicker'
import type { ActionState, Call, CallResult } from '../calls/types'
import './Rules.css'

/**
 * Everything you keep, and the only screen that has to work outside the app.
 *
 * This is the deliverable. A rule on its own is a thing to agree with and
 * forget by Thursday, so every rule here is followed by the one errand that
 * makes it real and a control for saying where you are with it. Answer the
 * ten questions and this page is what you are left holding.
 *
 * It prints as black type on white (see Rules.css), which is also what "save
 * as PDF" produces, because the moment it earns its keep is in a dealership or
 * ten minutes before signing a lease — with the phone in one hand and no
 * intention of opening an app. `onExport` lets a caller swap in an image
 * export without this screen knowing anything about canvases.
 *
 * Unfinished first. The reader did not come here to admire what is already
 * done, and burying the two open errands under six finished ones is how a
 * reference page turns back into a trophy cabinet.
 */

/**
 * The sort. Open work, then finished work, then the questions that were never
 * theirs to begin with — which sink rather than disappear, because a reader
 * who set five aside should still be able to find them and change their mind.
 */
const RANK: Record<ActionState, number> = { open: 0, done: 1, na: 2 }

export function Rules({
  results,
  calls,
  actions,
  onClose,
  onExport,
  onToggleAction,
}: {
  /** Answers, oldest first. Practice runs are dropped here, wherever else they are. */
  results: CallResult[]
  /** The whole set, in the order it is asked in. */
  calls: Call[]
  /** Where each errand stands, by question id. Missing means not started. */
  actions: Record<number, ActionState>
  onClose: () => void
  /** Defaults to the print path, which is also how the page is saved as a PDF. */
  onExport?: () => void
  onToggleAction: (callId: number, state: ActionState) => void
}) {
  const reduce = useReducedMotion()

  const answered = useMemo(
    () => new Set(results.filter((r) => !r.practice).map((r) => r.callId)),
    [results],
  )

  // Numbered by position in the set, not by the day it was answered. The
  // number is how a reader cross-references this page against the home screen,
  // and a date is only ever true of one of the two.
  const numbered = useMemo(
    () => calls.map((call, i) => ({ call, no: i + 1 })),
    [calls],
  )

  const rows = useMemo(
    () =>
      numbered
        .filter((r) => answered.has(r.call.id))
        .map((r) => ({ ...r, state: actions[r.call.id] ?? ('open' as ActionState) }))
        // Ties break on the set order, so two finished errands never swap
        // places under the reader between one render and the next.
        .sort((a, b) => RANK[a.state] - RANK[b.state] || a.no - b.no),
    [numbered, actions, answered],
  )

  const sealed = useMemo(
    () => numbered.filter((r) => !answered.has(r.call.id)),
    [numbered, answered],
  )

  const todo = rows.filter((r) => r.state === 'open').length

  return (
    <div className="rules scroll">
      {/* Printed pages leave the app and lose the app. Off the screen this is
          the only thing that says what the sheet of paper is. */}
      <p className="rules-mark">Napkin · ten money questions</p>

      <header className="rules-head">
        <h1 className="rules-title">My list</h1>
        <div className="rules-actions">
          <button
            className="rules-action data press"
            onClick={() => (onExport ? onExport() : window.print())}
          >
            Print
          </button>
          <button className="rules-action data press" onClick={onClose}>
            Close
          </button>
        </div>
      </header>

      <p className="rules-count data-sm num">
        {rows.length} {rows.length === 1 ? 'rule' : 'rules'} ·{' '}
        {todo > 0 ? `${todo} to do` : 'nothing waiting'}
      </p>

      {rows.length === 0 ? (
        <p className="rules-empty">Answer a question and its rule lands here.</p>
      ) : (
        <ol className="rules-list">
          {rows.map(({ call, no, state }) => (
            // The list re-sorts the moment an errand is marked, so the row has
            // to be seen moving. Without this it teleports, and a reader who
            // tapped "Done" on row two watches an unrelated row two appear
            // under their thumb.
            <motion.li
              className="rules-row"
              key={call.id}
              data-state={state}
              layout
              transition={
                reduce ? { duration: 0 } : { type: 'spring', visualDuration: 0.28, bounce: 0 }
              }
            >
              <p className="rules-q">
                <span className="rules-no data-sm num">{no}</span>
                <span className="rules-q-t">{call.question}</span>
              </p>

              {/* The rule is the part worth remembering; the errand under it is
                  the part that changes a week. Neither is much use alone. */}
              <p className="rules-text">{call.rule}</p>
              <p className="rules-do">{call.action}</p>

              <ActionPicker
                callId={call.id}
                state={state}
                label={`The thing to do about question ${no}`}
                onChange={onToggleAction}
              />
            </motion.li>
          ))}
        </ol>
      )}

      {sealed.length > 0 && (
        <section className="rules-sealed">
          <p className="rules-count data-sm num">{sealed.length} still to come</p>
          <ul className="rules-seal-list">
            {sealed.map(({ call, no }) => (
              <li className="rules-seal" key={call.id}>
                {/* The question, not just its subject. The whole set is listed
                    on the home screen, so withholding it here bought nothing
                    and left nine rows that said "housing" and "tax" at someone
                    trying to work out what they had signed up for. What is
                    still hidden is the rule, which is the part you get by
                    running into it. */}
                <p className="rules-q">
                  <span className="rules-no data-sm num">{no}</span>
                  <span className="rules-q-t">{call.question}</span>
                </p>
                <span className="rules-seal-bar" aria-hidden="true" />
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="rules-foot data-sm">Estimates, not advice.</p>
    </div>
  )
}
