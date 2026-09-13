import { useMemo } from 'react'
import { Histogram } from '../ui/Histogram'
import { COMPUTE } from '../calls/compute'
import { blendCrowd, percentileOf } from '../lib/crowd'
import { judge, optimalValue, stepCount, type Call, type Profile } from '../calls/types'
import { moneyCompact } from '../lib/format'
import { haptic } from '../lib/haptics'
import './Outcome.css'

const VERDICT_COPY = {
  optimal: 'Optimal play',
  short: 'Money left behind',
  over: 'Overshot',
} as const

/**
 * What that answer was worth.
 *
 * The wrong answer shows a loss in red, unsoftened. That is deliberate: it is
 * the strongest viral lever and the honest one — a product that tells you your
 * call cost you nothing is a product that teaches nothing. The one concession
 * is that the loss is always framed against the optimal play on the same call,
 * never against the player's real finances, which they have not disclosed.
 *
 * The crowd histogram frequently shows the popular answer is not the optimal
 * one. That tension is the most screenshot-worthy thing in the app, so it gets
 * room rather than a footnote.
 */
export function Outcome({
  call,
  callNo,
  value,
  profile,
  localCrowd,
  onReceipt,
}: {
  call: Call
  callNo: number
  value: number
  profile: Profile
  /** Locally observed results for this call, same length as the seed. */
  localCrowd: number[] | undefined
  onReceipt: () => void
}) {
  const compute = COMPUTE[call.compute]
  const verdict = judge(value, call.optimal)
  const best = optimalValue(call.optimal)

  const mine = useMemo(() => compute(value, profile), [compute, value, profile])
  const theirs = useMemo(() => compute(best, profile), [compute, best, profile])

  const delta = mine.at65 - theirs.at65
  const steps = stepCount(call.variable)

  const crowd = useMemo(
    () => blendCrowd(call.crowd, localCrowd ?? new Array(steps).fill(0)),
    [call.crowd, localCrowd, steps],
  )

  const percentile = useMemo(
    () => percentileOf(value, crowd, call.variable.min, call.variable.step, best),
    [value, crowd, call.variable.min, call.variable.step, best],
  )

  return (
    <div className="outcome scroll">
      <span className="outcome-chip" data-v={verdict}>
        {VERDICT_COPY[verdict]}
      </span>

      <p className="outcome-hero num" data-v={verdict}>
        {verdict === 'optimal' ? '' : delta < 0 ? '−' : '+'}
        {moneyCompact(Math.abs(delta))}
      </p>

      {/* One line. If a second is needed, the call is too complicated. */}
      <p className="outcome-line">
        {verdict === 'optimal'
          ? call.rule
          : `${moneyCompact(Math.abs(delta))} ${
              delta < 0 ? 'walked past' : 'over-committed'
            } by 65, against the best play on this call.`}
      </p>

      <div className="tile-row outcome-compare">
        <div className="tile">
          <p className="tile-k">Your call</p>
          <p className="tile-v num">{moneyCompact(mine.at65)}</p>
        </div>
        <div className="tile" data-best>
          <p className="tile-k">Best play</p>
          <p className="tile-v num">{moneyCompact(theirs.at65)}</p>
        </div>
      </div>

      <section className="outcome-crowd">
        <p className="data-sm">Where everyone landed</p>
        <Histogram
          distribution={crowd}
          min={call.variable.min}
          step={call.variable.step}
          unit={call.variable.unit}
          value={value}
          optimal={best}
          verdict={verdict}
        />
        <p className="outcome-percentile data-sm num">
          Better than {percentile.toFixed(0)}% of players
        </p>
      </section>

      <section className="outcome-rule">
        <p className="data-sm">Rule filed · No.{callNo}</p>
        <p className="outcome-rule-text">{call.rule}</p>
      </section>

      <button
        className="outcome-print press"
        onClick={() => {
          haptic('medium')
          onReceipt()
        }}
      >
        Print my receipt
      </button>

      <p className="outcome-assumptions data-sm">{call.assumptions}</p>
    </div>
  )
}
