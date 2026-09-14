/*
 * The pure geometry helpers live beside the component that consumes them,
 * because they are the definition of where a bar sits rather than shared
 * utilities, and the tests exercise them directly. Same arrangement — and same
 * reason — as `tabSummary` in screens/Tab.tsx. The cost is fast refresh in dev.
 */
/* eslint-disable react/only-export-components */
import { useMemo, type CSSProperties } from 'react'
import { modeOf } from '../lib/crowd'
import type { Verdict } from '../calls/types'
import './Histogram.css'

/**
 * Where everyone landed.
 *
 * The whole point of this chart is the gap between the tallest bar and the
 * marker: on most calls the popular answer is not the optimal one, and seeing
 * the crowd pile up two steps short of free money is the thing worth
 * screenshotting. So the mode is lifted a shade out of the crowd, the optimum
 * gets a hard accent rule, and the legend underneath names both values — if
 * they differ, the line says so in four words and the eye has already found it.
 *
 * Deliberately not a chart library. This is one flex row of divs; an SVG chart
 * with axes and ticks would bring a visual language that is not this product's.
 */
export function Histogram({
  distribution,
  min,
  step,
  unit,
  value,
  optimal,
  verdict,
}: {
  /** Bucket weights, index 0 at `min`. Any scale — only ratios are read. */
  distribution: number[]
  min: number
  step: number
  unit: string
  /** The player's answer. */
  value: number
  /**
   * The best play, as a single point. May sit between two buckets when the call
   * accepts a range; non-finite when the call has no upper bound, in which case
   * there is no honest place to put the marker and it is left off.
   */
  optimal: number
  verdict: Verdict
}) {
  const n = distribution.length
  const max = min + Math.max(0, n - 1) * step

  const peak = useMemo(
    () => distribution.reduce((m, w) => (Number.isFinite(w) && w > m ? w : m), 0),
    [distribution],
  )

  const mode = useMemo(() => modeOf(distribution, min, step), [distribution, min, step])
  const hasCrowd = peak > 0

  const youIndex = bucketOf(value, min, step, n)
  const modeIndex = hasCrowd ? bucketOf(mode, min, step, n) : -1

  // Fractional on purpose: a range optimum (3-6 months) resolves to 4.5 and the
  // marker belongs between two bars, not snapped onto one of them.
  const hasOptimal = Number.isFinite(optimal)
  const optimalOffset = offsetPercent(optimal, min, step, n)

  if (n === 0) return null

  return (
    <figure
      className="hist"
      role="img"
      aria-label={[
        // Spoken to someone who cannot see the bars, so it says what the
        // picture says and nothing the product cannot back up: the chart is a
        // research prior, not a count of other people using the app.
        hasCrowd
          ? `Crowd distribution. Most people answered ${tick(mode, unit)}.`
          : 'No crowd data yet.',
        hasOptimal ? `The best answer is ${tick(optimal, unit)}.` : '',
        `You answered ${tick(value, unit)}.`,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <div className="hist-plot">
        <div className="hist-bars" aria-hidden="true">
          {distribution.map((w, i) => {
            const ratio = peak > 0 && Number.isFinite(w) && w > 0 ? w / peak : 0
            const role = i === youIndex ? 'you' : i === modeIndex ? 'mode' : 'crowd'
            return (
              <span
                key={i}
                className="hist-bar"
                data-role={role}
                // Only the player's own bar carries the verdict: it is the only
                // one that turns red, and fifty copies of the attribute would
                // imply the rest of the chart cared about it.
                data-v={role === 'you' ? verdict : undefined}
                style={barStyle(ratio, i)}
              />
            )
          })}
        </div>

        {hasOptimal && (
          <div className="hist-optimal" style={{ left: `${optimalOffset}%` }} aria-hidden="true">
            <i className="hist-optimal-cap" />
          </div>
        )}
      </div>

      <div className="hist-axis data-sm num" aria-hidden="true">
        <span>{tick(min, unit)}</span>
        <span>{tick(max, unit)}</span>
      </div>

      <figcaption className="hist-legend data-sm num">
        {hasCrowd && (
          <span className="hist-key" data-role="mode">
            Most {tick(mode, unit)}
          </span>
        )}
        <span className="hist-key" data-role="you" data-v={verdict}>
          You {tick(value, unit)}
        </span>
        {hasOptimal && (
          <span className="hist-key" data-role="best">
            Best {tick(optimal, unit)}
          </span>
        )}
      </figcaption>
    </figure>
  )
}

/**
 * Ratio drives the height in CSS so the empty-bucket floor can stay a pixel
 * value: a 1% bucket must still read as a tick on the axis, or a distribution
 * with one big spike looks like it has holes punched in it.
 *
 * `--i` staggers the rise and is capped, because a 50-bucket chart animating at
 * 12ms a bar would take longer to draw than anyone looks at it.
 */
function barStyle(ratio: number, index: number): CSSProperties {
  // Deliberately not named --r: that is the global corner-radius token, and a
  // local override of it would hand any future child of a bar a radius of 0.42.
  return { '--fill': ratio, '--i': Math.min(index, 24) } as CSSProperties
}

/** Nearest bucket to a value, or -1 if it falls outside the distribution. */
export function bucketOf(value: number, min: number, step: number, n: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(step) || step <= 0) return -1
  const i = Math.round((value - min) / step)
  return i >= 0 && i < n ? i : -1
}

/** Horizontal position of a value, as a percentage of the plot's width. */
export function offsetPercent(value: number, min: number, step: number, n: number): number {
  if (n <= 0 || !Number.isFinite(value) || step <= 0) return 0
  // Bars are centred in their slot, so bucket k sits at (k + 0.5) / n.
  const k = (value - min) / step
  return Math.min(100, Math.max(0, ((k + 0.5) / n) * 100))
}

/** `6%`, `4.5mo`, `500`. Two decimals at most, trailing zeros never printed. */
export function tick(v: number, unit: string): string {
  if (!Number.isFinite(v)) return '—'
  return `${Math.round(v * 100) / 100}${unit}`
}
