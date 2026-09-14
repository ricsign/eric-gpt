import { useEffect, useState } from 'react'
import { formatCountdown, msUntilNextCall } from '../lib/schedule'
import type { CallResult } from '../calls/types'
import './Tomorrow.css'

/**
 * The end of one question, and the only screen that sells the next one.
 *
 * The hook is the situation, never an exhortation. "The mechanic calls with a
 * number" is a reason to come back; "come back tomorrow!" is a reason to
 * delete the app. The same string is what a reminder would say.
 *
 * What this screen used to lead with was a running dollar total labelled
 * "Your tab at 65", which on day one read "$0" — the product congratulating
 * someone for nothing, in a unit it had no business claiming. Progress
 * through a set of ten is a number that is true on day one and means the same
 * thing on day ten.
 */
export function Tomorrow({
  teaser,
  nextTitle,
  results,
  onTab,
  onRules,
  onHome,
}: {
  /** The next question's situation line, used as the hook. */
  teaser: string
  nextTitle: string
  results: CallResult[]
  onTab: () => void
  onRules: () => void
  onHome: () => void
}) {
  const [remaining, setRemaining] = useState(() => msUntilNextCall())
  const answered = new Set(results.map((r) => r.callId)).size
  const finished = nextTitle === '' && teaser === ''

  useEffect(() => {
    const id = window.setInterval(() => setRemaining(msUntilNextCall()), 1000)
    return () => window.clearInterval(id)
  }, [])

  return (
    <div className="tomorrow">
      {finished ? (
        <>
          <p className="data-sm">That is all ten</p>
          <p className="tomorrow-teaser">
            You have been through the whole set. What you keep is on your list.
          </p>
        </>
      ) : (
        <>
          <p className="data-sm">Next question in</p>
          <p className="tomorrow-clock num">{formatCountdown(remaining)}</p>
          <p className="tomorrow-teaser">{teaser || nextTitle}</p>
        </>
      )}

      <div className="tomorrow-tab">
        <p className="tile-k">Your progress</p>
        <p className="tomorrow-tab-v num">{answered} of 10</p>
        <p className="data-sm">answered</p>
      </div>

      <nav className="tomorrow-nav">
        <button className="tomorrow-link press" onClick={onRules}>
          My list
        </button>
        <button className="tomorrow-link press" onClick={onTab}>
          Progress
        </button>
        <button className="tomorrow-link press" onClick={onHome}>
          All questions
        </button>
      </nav>
    </div>
  )
}
