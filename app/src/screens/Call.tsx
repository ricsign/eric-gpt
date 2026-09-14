import { useEffect, useMemo, useState } from 'react'
import { BlockBar } from '../ui/BlockBar'
import { COMPUTE } from '../calls/compute'
import { resolveFact, stepCount, type Call, type Profile } from '../calls/types'
import { money, moneyCompact } from '../lib/format'
import { formatCountdown, msUntilNextCall } from '../lib/schedule'
import { haptic } from '../lib/haptics'
import './Call.css'

/**
 * The call screen.
 *
 * Everything on it below the headline recomputes on the same frame as the drag.
 * That is not a performance nicety — it is the entire feel of the product, and
 * the reason the lesson lands without a word of explanation. The player drags,
 * watches the accent blocks stop filling while their monthly cost keeps
 * climbing, and has understood that the match has a ceiling in about four
 * seconds.
 *
 * There is no explanation on this screen, by design. The rule is only revealed
 * after they have committed to an answer.
 */
export function CallScreen({
  call,
  callNo,
  profile,
  playedCount,
  onLockIn,
}: {
  call: Call
  /** The global call number — No.142 — not the library index. */
  callNo: number
  profile: Profile
  /** Real plays recorded for this call, or null when there is no live count. */
  playedCount: number | null
  onLockIn: (value: number) => void
}) {
  const [value, setValue] = useState(call.variable.start)
  const [touched, setTouched] = useState(false)
  const [remaining, setRemaining] = useState(() => msUntilNextCall())

  // The countdown only needs to be right to the second, and only while this
  // screen is mounted. Started in render it would outlive every visit — the
  // player passes through here once a day for as long as they keep the app
  // open, and each pass would leave another timer ticking against a screen
  // that no longer exists.
  useEffect(() => {
    const id = window.setInterval(() => setRemaining(msUntilNextCall()), 1000)
    return () => window.clearInterval(id)
  }, [])

  const compute = COMPUTE[call.compute]
  const outcome = useMemo(
    () => compute(value, profile),
    [compute, value, profile],
  )

  const blocks = Math.min(15, stepCount(call.variable))

  return (
    <div className="call">
      <header className="call-head">
        <span className="data-sm">No.{callNo}</span>
        {playedCount !== null ? (
          <span className="data-sm num">{playedCount.toLocaleString('en-US')} played</span>
        ) : (
          <span className="data-sm num">closes {formatCountdown(remaining)}</span>
        )}
      </header>

      <h1 className="call-title">{call.title}</h1>

      <div className="tile-row call-facts">
        {call.fixed.map((f) => (
          <div className="tile" key={f.k}>
            <p className="tile-k">{f.k}</p>
            <p className="tile-v num">{resolveFact(f.v, profile, money)}</p>
          </div>
        ))}
      </div>

      <div className="call-control">
        <BlockBar
          value={value}
          onChange={(v) => {
            setValue(v)
            if (!touched) setTouched(true)
          }}
          min={call.variable.min}
          max={call.variable.max}
          step={call.variable.step}
          blocks={blocks}
          tint={outcome.blockTint}
          label={call.variable.label}
        />

        <output className="call-readout num">
          {value}
          <span className="call-unit">{call.variable.unit}</span>
        </output>
        <p className="call-readout-label data-sm">{call.variable.label}</p>
      </div>

      <div className="tile-row call-consequences">
        <div className="tile">
          <p className="tile-k">Costs you</p>
          <p className="tile-v num">{money(outcome.cost)}<span className="tile-unit">/mo</span></p>
        </div>
        <div className="tile" data-good={outcome.benefit > 0 || undefined}>
          <p className="tile-k">You capture</p>
          <p className="tile-v num">{money(outcome.benefit)}<span className="tile-unit">/yr</span></p>
        </div>
      </div>

      <div className="call-hero tile">
        <p className="tile-k">At 65 this is worth</p>
        <p className="call-hero-v num">{moneyCompact(outcome.at65)}</p>
      </div>

      <button
        className="call-lock press"
        disabled={!touched}
        onClick={() => {
          haptic('heavy')
          onLockIn(value)
        }}
      >
        {touched ? 'Lock it in' : 'Move the bar'}
      </button>
    </div>
  )
}
