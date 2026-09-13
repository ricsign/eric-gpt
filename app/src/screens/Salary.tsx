import { useRef, useState } from 'react'
import { haptic } from '../lib/haptics'
import { money } from '../lib/format'
import { DEFAULT_PROFILE, type Profile } from '../calls/types'
import './Salary.css'

/**
 * First run. One question, then the game.
 *
 * No splash, no login wall, no tutorial, no email, no age, no goals
 * questionnaire. This single number personalises every figure in the product
 * forever, and asking for anything else would trade the thing that makes the
 * cold open work — that you are playing within four seconds — for data nobody
 * has earned the right to yet.
 *
 * The escape hatch is deliberately prominent. A player who skips still gets a
 * fully working game on the median salary, and can set their own later.
 */
export function Salary({ onDone }: { onDone: (p: Profile) => void }) {
  const [salary, setSalary] = useState(DEFAULT_PROFILE.salary)
  const [dragging, setDragging] = useState(false)
  const trackRef = useRef<HTMLDivElement>(null)
  const lastTick = useRef(salary)

  const MIN = 15_000
  const MAX = 400_000
  const STEP = 1_000

  /**
   * Log scale. Salary spans more than an order of magnitude, and on a linear
   * track every value under $60k — which is most people — is crammed into the
   * first eighth of the bar and unreachable with a thumb.
   */
  const toFraction = (v: number) =>
    (Math.log(v) - Math.log(MIN)) / (Math.log(MAX) - Math.log(MIN))

  const toValue = (f: number) => {
    const raw = Math.exp(
      Math.min(1, Math.max(0, f)) * (Math.log(MAX) - Math.log(MIN)) + Math.log(MIN),
    )
    return Math.min(MAX, Math.max(MIN, Math.round(raw / STEP) * STEP))
  }

  const setFromPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect()
    if (!rect) return
    const next = toValue((clientX - rect.left) / rect.width)
    if (next === salary) return
    // One tick per $1,000 crossed, not per pointermove event.
    if (next !== lastTick.current) {
      haptic('selection')
      lastTick.current = next
    }
    setSalary(next)
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    const big = 10_000
    const map: Record<string, number> = {
      ArrowRight: STEP,
      ArrowUp: STEP,
      ArrowLeft: -STEP,
      ArrowDown: -STEP,
      PageUp: big,
      PageDown: -big,
    }
    if (e.key === 'Home') return setSalary(MIN)
    if (e.key === 'End') return setSalary(MAX)
    const d = map[e.key]
    if (d === undefined) return
    e.preventDefault()
    setSalary((s) => Math.min(MAX, Math.max(MIN, s + d)))
  }

  return (
    <div className="salary">
      <p className="salary-kicker data-sm">Compound · first run</p>

      <h1 className="salary-q">What do you make a year?</h1>

      <output className="salary-value num" data-dragging={dragging || undefined}>
        {money(salary)}
      </output>

      <div
        className="salary-track"
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="Annual salary"
        aria-valuemin={MIN}
        aria-valuemax={MAX}
        aria-valuenow={salary}
        aria-valuetext={money(salary)}
        onKeyDown={onKeyDown}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          setDragging(true)
          setFromPointer(e.clientX)
        }}
        onPointerMove={(e) => {
          if (!dragging) return
          if (e.cancelable) e.preventDefault()
          setFromPointer(e.clientX)
        }}
        onPointerUp={() => {
          setDragging(false)
          haptic('light')
        }}
        onPointerCancel={() => setDragging(false)}
      >
        <div className="salary-fill" style={{ width: `${toFraction(salary) * 100}%` }} />
      </div>

      <p className="salary-note data-sm">
        Stays on this device. Never uploaded.
      </p>

      <div className="salary-actions">
        <button
          className="salary-go press"
          onClick={() => {
            haptic('medium')
            onDone({ ...DEFAULT_PROFILE, salary })
          }}
        >
          Start
        </button>
        <button
          className="salary-skip"
          onClick={() => {
            haptic('light')
            onDone(DEFAULT_PROFILE)
          }}
        >
          Skip, use {money(DEFAULT_PROFILE.salary)}
        </button>
      </div>
    </div>
  )
}
