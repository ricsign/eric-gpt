import { useEffect, useState } from 'react'
import { tabSummary } from '../screens/Tab'
import { formatCountdown, msUntilNextCall } from '../lib/schedule'
import { moneyCompact } from '../lib/format'
import type { CallResult } from '../calls/types'
import './Tomorrow.css'

/**
 * The end of the loop, and the only screen that sells the next one.
 *
 * The hook is the scenario, never an exhortation. "A $2,400 repair on a $5,000
 * car" is a reason to come back; "come back tomorrow!" is a reason to delete the
 * app. The same string is what the notification says.
 */
export function Tomorrow({
  teaser,
  nextTitle,
  results,
  onTab,
  onRules,
}: {
  teaser: string
  nextTitle: string
  results: CallResult[]
  onTab: () => void
  onRules: () => void
}) {
  const [remaining, setRemaining] = useState(() => msUntilNextCall())
  const summary = tabSummary(results)

  useEffect(() => {
    const id = window.setInterval(() => setRemaining(msUntilNextCall()), 1000)
    return () => window.clearInterval(id)
  }, [])

  return (
    <div className="tomorrow">
      <p className="data-sm">Next call in</p>
      <p className="tomorrow-clock num">{formatCountdown(remaining)}</p>

      <p className="tomorrow-teaser">{teaser || nextTitle}</p>

      <div className="tomorrow-tab">
        <p className="tile-k">Your tab at 65</p>
        <p className="tomorrow-tab-v num">{moneyCompact(summary.total)}</p>
        <p className="data-sm num">
          {summary.played} {summary.played === 1 ? 'call' : 'calls'} · {summary.streak} day streak
        </p>
      </div>

      <nav className="tomorrow-nav">
        <button className="tomorrow-link press" onClick={onTab}>
          The Tab
        </button>
        <button className="tomorrow-link press" onClick={onRules}>
          Rules
        </button>
      </nav>
    </div>
  )
}
