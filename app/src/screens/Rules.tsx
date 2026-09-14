import { useMemo } from 'react'
import { callNumber, formatCallDate } from '../lib/schedule'
import type { Call, CallResult } from '../calls/types'
import './Rules.css'

/**
 * Everything the player has earned, one line each.
 *
 * This is the usefulness payload — the page someone reopens in a dealership or
 * before signing a lease — so it is the one surface that has to survive leaving
 * the app. It prints as black-on-white type (see Rules.css), which is also what
 * a "save as PDF" produces, and `onExport` lets a caller swap in an image
 * export without this screen knowing anything about canvases.
 *
 * A rule is earned by running into it, not by getting it right. The call that
 * cost you $188,000 is the one you remember.
 */
export function Rules({
  results,
  calls,
  onClose,
  onExport,
}: {
  /** Real answers, oldest first. Practice runs must be filtered out upstream. */
  results: CallResult[]
  calls: Call[]
  onClose: () => void
  /** Defaults to the print path, which is also how the page is saved as a PDF. */
  onExport?: () => void
}) {
  const earned = useMemo(() => earnedRules(results, calls), [results, calls])

  return (
    <div className="rules scroll">
      <header className="rules-head">
        <h1 className="rules-title">Rules</h1>
        <div className="rules-actions">
          <button
            className="rules-action data press"
            onClick={() => (onExport ? onExport() : window.print())}
          >
            Export
          </button>
          <button className="rules-action data press" onClick={onClose}>
            Close
          </button>
        </div>
      </header>

      <p className="rules-count data-sm num">
        {earned.length} {earned.length === 1 ? 'rule' : 'rules'} earned
      </p>

      {earned.length === 0 ? (
        <p className="rules-empty">No rules yet.</p>
      ) : (
        <ol className="rules-list">
          {earned.map((r) => (
            <li className="rules-row" key={r.callId}>
              <div className="rules-meta data-sm num">
                <span className="rules-no">No.{r.no}</span>
                <span>{r.date}</span>
                <span className="rules-tag">{r.domain}</span>
              </div>
              <p className="rules-text">{r.rule}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

interface EarnedRule {
  callId: number
  no: number
  date: string
  rule: string
  domain: string
}

/**
 * One row per rule, newest first.
 *
 * Deduplicated by call, because the library cycles: past day ten the same call
 * comes round again, and a Rules page that listed "Take the match before
 * anything else" four times would be a log, not a reference. The most recent
 * encounter wins, so the number and date point at the last time they met it.
 */
function earnedRules(results: CallResult[], calls: Call[]): EarnedRule[] {
  const byId = new Map(calls.map((c) => [c.id, c]))
  const latest = new Map<number, CallResult>()

  for (const r of results) {
    const seen = latest.get(r.callId)
    if (!seen || r.day >= seen.day) latest.set(r.callId, r)
  }

  return [...latest.values()]
    .sort((a, b) => (a.day === b.day ? b.callId - a.callId : b.day.localeCompare(a.day)))
    .flatMap((r) => {
      // A result whose call is no longer in the library is dropped rather than
      // rendered blank: the registry is the only source of a rule's wording.
      const call = byId.get(r.callId)
      if (!call) return []
      return [
        {
          callId: r.callId,
          no: callNumber(r.day),
          date: formatCallDate(r.day),
          rule: call.rule,
          domain: call.domain,
        },
      ]
    })
}
