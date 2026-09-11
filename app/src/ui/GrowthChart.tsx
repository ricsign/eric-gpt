import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useReducedMotion } from 'motion/react'
import type { GrowthPoint } from '../lib/finance'
import { moneyCompact } from '../lib/format'
import { haptic } from '../lib/haptics'
import './GrowthChart.css'

export interface GrowthChartProps {
  points: GrowthPoint[]
  /** Draws a second, dimmed curve behind the main one — the "if you waited" path. */
  comparison?: GrowthPoint[]
  comparisonLabel?: string
  height?: number
  /** Horizontal rules at these balances, e.g. a $1M goal. */
  milestones?: { value: number; label: string }[]
  /** Disable the scrub interaction for small inline sparkline uses. */
  interactive?: boolean
  /**
   * Whether the chart owns the headline figure.
   *
   * Off when the surrounding screen already shows it — two copies of one number
   * disagreeing while a count-up is in flight reads as a bug, not as emphasis.
   * The readout still appears while scrubbing, because at that moment it is
   * showing something the headline is not: the value at the year under the finger.
   */
  showReadout?: boolean
}

const PAD = { top: 18, right: 12, bottom: 22, left: 12 }

/**
 * The compound-growth chart.
 *
 * Three deliberate choices, each in service of the one idea the app exists to teach:
 *
 *  1. **The area is split, not solid.** The lower band is money the user put in; the
 *     upper band is money that growth created. The visual gap between them opening up
 *     over time *is* the lesson — a single-colour area chart hides it completely.
 *  2. **Linear y-axis, always.** A log axis makes exponential growth look like a
 *     straight line, which is precisely the intuition being corrected. The dramatic
 *     hockey stick is not a distortion here; it is the finding.
 *  3. **Scrubbing beats a legend.** Dragging across the curve answers "what about at
 *     40?" directly, which is the question every person actually has.
 */
export function GrowthChart({
  points,
  comparison,
  comparisonLabel = 'If you wait',
  height = 220,
  milestones = [],
  interactive = true,
  showReadout = true,
}: GrowthChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(340)
  const [scrub, setScrub] = useState<number | null>(null)
  const reduced = useReducedMotion()
  const lastHapticIndex = useRef(-1)

  // Measure rather than assume: the chart appears at several widths (card, sheet,
  // share image) and an SVG viewBox alone would distort the stroke weights.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      setWidth(Math.max(200, entry.contentRect.width))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const geom = useMemo(() => {
    if (points.length < 2) return null

    const innerW = width - PAD.left - PAD.right
    const innerH = height - PAD.top - PAD.bottom

    const maxYear = points[points.length - 1].year
    // Both curves share a scale, otherwise the comparison is a lie.
    const maxValue = Math.max(
      points[points.length - 1].balance,
      comparison?.[comparison.length - 1]?.balance ?? 0,
      ...milestones.map((m) => m.value),
      1,
    )

    const x = (year: number) => PAD.left + (year / maxYear) * innerW
    const y = (value: number) => PAD.top + innerH - (value / maxValue) * innerH

    // One sample per ~2px of width: enough for a smooth curve, few enough that the
    // path string stays short and the scrub lookup stays cheap.
    const step = Math.max(1, Math.floor(points.length / (innerW / 2)))
    const sampled = points.filter((_, i) => i % step === 0 || i === points.length - 1)

    const line = (pts: GrowthPoint[], key: 'balance' | 'contributed') =>
      pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.year).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ')

    const baseline = y(0)

    return {
      innerW,
      innerH,
      maxYear,
      maxValue,
      x,
      y,
      sampled,
      baseline,
      /** Outline of the total balance. */
      balanceLine: line(sampled, 'balance'),
      /** Everything under the balance curve. */
      balanceArea: `${line(sampled, 'balance')} L${x(maxYear).toFixed(1)},${baseline} L${PAD.left},${baseline} Z`,
      /** The contributions band, drawn on top so it masks the lower part of the area. */
      contributedArea: `${line(sampled, 'contributed')} L${x(maxYear).toFixed(1)},${baseline} L${PAD.left},${baseline} Z`,
      comparisonLine: comparison
        ? line(
            comparison.filter((_, i) => i % step === 0 || i === comparison.length - 1),
            'balance',
          )
        : null,
    }
  }, [points, comparison, width, height, milestones])

  const pointAt = useCallback(
    (clientX: number): number | null => {
      const el = wrapRef.current
      if (!el || !geom) return null
      const rect = el.getBoundingClientRect()
      const rel = (clientX - rect.left - PAD.left) / geom.innerW
      const idx = Math.round(Math.min(1, Math.max(0, rel)) * (points.length - 1))
      return idx
    },
    [geom, points.length],
  )

  const onScrubMove = (e: React.PointerEvent) => {
    if (!interactive) return
    const idx = pointAt(e.clientX)
    if (idx == null) return
    setScrub(idx)

    // A tick as the readout crosses each year — the same feedback a picker gives.
    const year = Math.floor(points[idx].year)
    if (year !== lastHapticIndex.current) {
      lastHapticIndex.current = year
      haptic('selection')
    }
  }

  if (!geom) return <div ref={wrapRef} style={{ height }} />

  const active = scrub != null ? points[scrub] : null
  const last = points[points.length - 1]
  const shown = active ?? last

  return (
    <div className="growthchart" ref={wrapRef}>
      {/* Live readout. Sits above the chart so a finger on the curve never covers it.
          When the parent owns the headline, only the split chips are shown. */}
      <div className="growthchart-readout" aria-live="polite">
        {(showReadout || active) && (
          <div className="growthchart-readout-main">
            <span className="growthchart-readout-value num">{moneyCompact(shown.balance)}</span>
            <span className="growthchart-readout-year">
              {active
                ? `at year ${Math.round(shown.year)}`
                : `after ${Math.round(last.year)} years`}
            </span>
          </div>
        )}
        <div className="growthchart-readout-split">
          <span className="growthchart-chip growthchart-chip--in">
            <i />
            {moneyCompact(shown.contributed)} in
          </span>
          <span className="growthchart-chip growthchart-chip--growth">
            <i />
            {moneyCompact(shown.growth)} growth
          </span>
        </div>
      </div>

      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="growthchart-svg"
        role="img"
        aria-label={`Projected balance reaching ${moneyCompact(last.balance)} after ${Math.round(
          last.year,
        )} years, of which ${moneyCompact(last.contributed)} is money you put in and ${moneyCompact(
          last.growth,
        )} is growth.`}
        onPointerDown={(e) => {
          if (!interactive) return
          e.currentTarget.setPointerCapture(e.pointerId)
          onScrubMove(e)
        }}
        onPointerMove={(e) => {
          if (e.buttons === 0 && e.pointerType === 'mouse') return
          if (scrub == null && e.pointerType !== 'mouse') return
          onScrubMove(e)
        }}
        onPointerUp={() => {
          setScrub(null)
          lastHapticIndex.current = -1
        }}
        onPointerCancel={() => setScrub(null)}
      >
        <defs>
          <linearGradient id="growthFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--growth)" stopOpacity="0.34" />
            <stop offset="100%" stopColor="var(--growth)" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {milestones.map((m) => (
          <g key={m.label}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={geom.y(m.value)}
              y2={geom.y(m.value)}
              className="growthchart-milestone-line"
            />
            <text x={PAD.left + 2} y={geom.y(m.value) - 5} className="growthchart-milestone-label">
              {m.label}
            </text>
          </g>
        ))}

        {/* Growth band: the full area, with the contributions band painted over it. */}
        <path d={geom.balanceArea} fill="url(#growthFill)" />
        <path d={geom.contributedArea} className="growthchart-contributed" />

        {geom.comparisonLine && (
          <path d={geom.comparisonLine} className="growthchart-comparison" />
        )}

        <path
          d={geom.balanceLine}
          className={`growthchart-line ${reduced ? '' : 'growthchart-line--draw'}`}
        />

        {active && (
          <g className="growthchart-cursor">
            <line
              x1={geom.x(active.year)}
              x2={geom.x(active.year)}
              y1={PAD.top}
              y2={geom.baseline}
            />
            <circle cx={geom.x(active.year)} cy={geom.y(active.balance)} r="5" />
            <circle
              cx={geom.x(active.year)}
              cy={geom.y(active.contributed)}
              r="3"
              className="growthchart-cursor-in"
            />
          </g>
        )}

        {/* Year axis: first, middle, last. More labels than that is noise at this size. */}
        {[0, Math.round(geom.maxYear / 2), Math.round(geom.maxYear)].map((yr, i) => (
          <text
            key={yr}
            x={geom.x(yr)}
            y={height - 6}
            className="growthchart-axis"
            textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}
          >
            {yr === 0 ? 'now' : `${yr}y`}
          </text>
        ))}
      </svg>

      {geom.comparisonLine && (
        <p className="growthchart-comparison-key">
          <span className="growthchart-dash" aria-hidden="true" />
          {comparisonLabel}
        </p>
      )}

      {interactive && (
        <p className="growthchart-hint">
          {scrub == null ? 'Drag across the curve to scrub through the years' : ' '}
        </p>
      )}
    </div>
  )
}
