import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useReducedMotion } from 'motion/react'
import { compareCurve, type CurveComparison, type StrokePoint } from '../lib/curve'
import { haptic } from '../lib/haptics'
import { moneyCompact } from '../lib/format'
import './CurveDraw.css'

export interface CurvePoint {
  /** 0..1 along the horizontal domain. */
  t: number
  /** The real value at that point. */
  value: number
}

export interface CurveDrawProps {
  /** The truth, sampled across the domain. */
  truth: CurvePoint[]
  /** Axis labels. */
  xLabel: string
  yMax: number
  /** Years, months, whatever the x axis counts. Used in the readout. */
  domainLabel: string
  /** Fires once the user has committed a prediction and the reveal has run. */
  onCommit: (result: CurveResult) => void
  height?: number
}

export interface CurveResult extends CurveComparison {
  /** True when they used the "not sure" escape hatch. */
  assisted: boolean
  /** The stroke as drawn, so the share card can redraw it exactly. */
  stroke: StrokePoint[]
}

/** How much of the width must be drawn before a prediction counts. */
const MIN_COVERAGE = 0.88

/**
 * Draw where you think it goes.
 *
 * This is the product's central interaction, and it is doing four jobs at once:
 *
 *  - **Generation.** Committing to a prediction before seeing the answer is the
 *    highest-leverage move in the learning-science literature, and a gesture
 *    commits far more willingly than a number field does.
 *  - **Measurement.** The gap between the drawn line and the real curve is a
 *    direct, quantitative instrument for exponential-growth bias — the specific
 *    misconception with the largest documented behavioural effect in this domain.
 *  - **Feeling.** Being told the number teaches you the number. Watching the truth
 *    sweep away from a line you drew with your own finger teaches you that your
 *    intuition is the thing that was wrong, which is the part that transfers.
 *  - **Sharing.** The resulting image contains a hand-drawn line and a percentage.
 *    No balance, no salary, no net worth — so the money taboo that kills every
 *    other finance share artifact does not apply to it.
 *
 * Deliberately a gesture and never an un-skippable numeric entry: demanding
 * arithmetic on the second screen from an anxious, low-numeracy user is the
 * highest-churn design available.
 */
export function CurveDraw({
  truth,
  xLabel,
  yMax,
  domainLabel,
  onCommit,
  height = 260,
}: CurveDrawProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(340)
  const [stroke, setStroke] = useState<StrokePoint[]>([])
  const [drawing, setDrawing] = useState(false)
  const [committed, setCommitted] = useState(false)
  const [assisted, setAssisted] = useState(false)
  const reduced = useReducedMotion()

  const PAD = { top: 14, right: 14, bottom: 26, left: 14 }
  const innerW = width - PAD.left - PAD.right
  const innerH = height - PAD.top - PAD.bottom

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(220, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const toScreen = useCallback(
    (t: number, v: number) => ({
      x: PAD.left + t * innerW,
      y: PAD.top + innerH - (Math.min(v, yMax) / yMax) * innerH,
    }),
    [PAD.left, PAD.top, innerW, innerH, yMax],
  )

  const truthPath = useMemo(
    () =>
      truth
        .map((p, i) => {
          const s = toScreen(p.t, p.value)
          return `${i === 0 ? 'M' : 'L'}${s.x.toFixed(1)},${s.y.toFixed(1)}`
        })
        .join(' '),
    [truth, toScreen],
  )

  const strokePath = useMemo(() => {
    if (stroke.length < 2) return ''
    return stroke
      .map((p, i) => {
        const x = PAD.left + p.x * innerW
        const y = PAD.top + (1 - p.y) * innerH
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(' ')
  }, [stroke, PAD.left, PAD.top, innerW, innerH])

  const coverage = stroke.length ? stroke[stroke.length - 1].x : 0
  const canCommit = coverage >= MIN_COVERAGE && !committed

  /** Records a point, enforcing left-to-right monotonicity. */
  const record = (clientX: number, clientY: number) => {
    const rect = wrapRef.current?.getBoundingClientRect()
    if (!rect) return

    const x = Math.min(1, Math.max(0, (clientX - rect.left - PAD.left) / innerW))
    const y = Math.min(1, Math.max(0, 1 - (clientY - rect.top - PAD.top) / innerH))

    setStroke((s) => {
      // A prediction is a function of time: going backwards would make it ambiguous,
      // so leftward movement extends the current point rather than adding one.
      if (s.length && x <= s[s.length - 1].x) {
        const next = [...s]
        next[next.length - 1] = { x: next[next.length - 1].x, y }
        return next
      }
      return [...s, { x, y }]
    })
  }

  const commit = (viaAssist = false) => {
    const comparison = compareCurve(
      stroke,
      truth.map((p) => p.value),
      yMax,
    )
    if (!comparison) return

    haptic('medium')
    setCommitted(true)
    setAssisted(viaAssist)
    onCommit({ ...comparison, assisted: viaAssist, stroke })
  }

  /** The escape hatch: a straight line from origin to the end of the domain. */
  const notSure = () => {
    const linear = Array.from({ length: 24 }, (_, i) => {
      const t = i / 23
      // A straight line to the sum of contributions is the canonical wrong answer,
      // so the escape hatch still records a real, analysable prediction.
      return { x: t, y: t * 0.45 }
    })
    setStroke(linear)
    haptic('light')
    window.setTimeout(() => commit(true), 260)
  }

  const endStroke = () => {
    setDrawing(false)
    if (coverage >= MIN_COVERAGE) haptic('light')
  }

  return (
    <div className="curve">
      <div
        className="curve-canvas"
        ref={wrapRef}
        style={{ height }}
        onPointerDown={(e) => {
          if (committed) return
          e.currentTarget.setPointerCapture(e.pointerId)
          setDrawing(true)
          setStroke([])
          record(e.clientX, e.clientY)
        }}
        onPointerMove={(e) => {
          if (!drawing || committed) return
          // Only claim the gesture once it is actually a drawing gesture, so a
          // vertical flick still scrolls the page.
          if (e.cancelable) e.preventDefault()
          record(e.clientX, e.clientY)
        }}
        onPointerUp={endStroke}
        onPointerCancel={endStroke}
      >
        <svg width={width} height={height} className="curve-svg" aria-hidden="true">
          {/* Gridlines give the gesture something to aim at without implying values. */}
          {[0.25, 0.5, 0.75, 1].map((f) => (
            <line
              key={f}
              className="curve-grid"
              x1={PAD.left}
              x2={width - PAD.right}
              y1={PAD.top + innerH * (1 - f)}
              y2={PAD.top + innerH * (1 - f)}
            />
          ))}

          <line
            className="curve-axis"
            x1={PAD.left}
            x2={width - PAD.right}
            y1={PAD.top + innerH}
            y2={PAD.top + innerH}
          />

          {strokePath && <path d={strokePath} className="curve-stroke" />}

          {committed && (
            <path
              d={truthPath}
              className={`curve-truth ${reduced ? '' : 'curve-truth--draw'}`}
            />
          )}

          {/* The origin dot: where every curve starts, so the gesture has an anchor. */}
          <circle cx={PAD.left} cy={PAD.top + innerH} r="4" className="curve-origin" />

          <text x={PAD.left} y={height - 6} className="curve-axis-label">
            now
          </text>
          <text x={width - PAD.right} y={height - 6} className="curve-axis-label" textAnchor="end">
            {xLabel}
          </text>
          <text x={PAD.left + 4} y={PAD.top + 12} className="curve-axis-label">
            {moneyCompact(yMax)}
          </text>
        </svg>

        {!stroke.length && !committed && (
          <p className="curve-prompt">
            Draw where you think it goes
            <span>with your finger, left to right</span>
          </p>
        )}

        {stroke.length > 0 && !committed && coverage < MIN_COVERAGE && (
          <p className="curve-nudge">Keep going — all the way to {xLabel}</p>
        )}
      </div>

      {!committed && (
        <div className="curve-actions">
          <button className="curve-commit" disabled={!canCommit} onClick={() => commit(false)}>
            {canCommit ? "That's my guess" : 'Draw your line first'}
          </button>
          <button className="curve-unsure" onClick={notSure}>
            I'm not sure
          </button>
        </div>
      )}

      {committed && (
        <p className="curve-key">
          <span className="curve-key-item curve-key-item--yours">
            <i /> your line{assisted ? ' (a straight guess)' : ''}
          </span>
          <span className="curve-key-item curve-key-item--truth">
            <i /> what actually happens over {domainLabel}
          </span>
        </p>
      )}
    </div>
  )
}
