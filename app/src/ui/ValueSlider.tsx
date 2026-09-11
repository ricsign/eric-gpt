import { useCallback, useEffect, useRef, useState } from 'react'
import { haptic } from '../lib/haptics'
import './ValueSlider.css'

export interface ValueSliderProps {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  /** How the big number above the track is rendered. */
  format: (v: number) => string
  label: string
  /** Small text under the track — the assumption, the unit, the caveat. */
  hint?: string
  /**
   * Compresses the scale so the low end gets more of the track. Money inputs span
   * orders of magnitude and a linear slider makes every value under $10k
   * unreachable on a phone.
   */
  curve?: 'linear' | 'log'
  disabled?: boolean
}

/**
 * The app's primary numeric input.
 *
 * A slider rather than a text field, for three reasons that matter here:
 * a keyboard covers half a phone screen and breaks the flow; every one of these
 * inputs is an estimate rather than a known figure, and a slider says so; and
 * dragging a value while the result updates live is the interaction that teaches
 * the relationship, which a typed number never does.
 *
 * The detent haptic on each step is what makes it feel like a physical control.
 */
export function ValueSlider({
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
  label,
  hint,
  curve = 'linear',
  disabled = false,
}: ValueSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const lastDetent = useRef<number>(value)

  /** Value -> 0..1 position on the track. */
  const toPosition = useCallback(
    (v: number) => {
      const clamped = Math.min(max, Math.max(min, v))
      if (curve === 'log') {
        // Shift by 1 so a min of 0 is representable.
        const lo = Math.log(min + 1)
        const hi = Math.log(max + 1)
        return (Math.log(clamped + 1) - lo) / (hi - lo)
      }
      return (clamped - min) / (max - min)
    },
    [min, max, curve],
  )

  /** 0..1 position -> value, snapped to the step. */
  const toValue = useCallback(
    (p: number) => {
      const clamped = Math.min(1, Math.max(0, p))
      const raw =
        curve === 'log'
          ? Math.exp(clamped * (Math.log(max + 1) - Math.log(min + 1)) + Math.log(min + 1)) - 1
          : min + clamped * (max - min)
      return Math.min(max, Math.max(min, Math.round(raw / step) * step))
    },
    [min, max, step, curve],
  )

  const position = toPosition(value)

  const handlePointer = useCallback(
    (clientX: number) => {
      const track = trackRef.current
      if (!track) return
      const rect = track.getBoundingClientRect()
      const next = toValue((clientX - rect.left) / rect.width)
      if (next !== value) {
        // One tick per step crossed, not per pointermove event.
        if (next !== lastDetent.current) {
          haptic('selection')
          lastDetent.current = next
        }
        onChange(next)
      }
    },
    [onChange, toValue, value],
  )

  // Keyboard support: a slider that only responds to a finger is not an input.
  const onKeyDown = (e: React.KeyboardEvent) => {
    const big = (max - min) / 10
    const map: Record<string, number> = {
      ArrowRight: step,
      ArrowUp: step,
      ArrowLeft: -step,
      ArrowDown: -step,
      PageUp: big,
      PageDown: -big,
    }
    if (e.key === 'Home') return onChange(min)
    if (e.key === 'End') return onChange(max)
    const delta = map[e.key]
    if (delta === undefined) return
    e.preventDefault()
    onChange(Math.min(max, Math.max(min, Math.round((value + delta) / step) * step)))
  }

  useEffect(() => {
    lastDetent.current = value
  }, [value])

  return (
    <div className="vslider" data-disabled={disabled || undefined}>
      <div className="vslider-head">
        <span className="vslider-label">{label}</span>
        <output className="vslider-value num" data-dragging={dragging || undefined}>
          {format(value)}
        </output>
      </div>

      <div
        className="vslider-track"
        ref={trackRef}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={format(value)}
        aria-disabled={disabled || undefined}
        onKeyDown={onKeyDown}
        onPointerDown={(e) => {
          if (disabled) return
          e.currentTarget.setPointerCapture(e.pointerId)
          setDragging(true)
          handlePointer(e.clientX)
        }}
        onPointerMove={(e) => {
          if (!dragging || disabled) return
          handlePointer(e.clientX)
        }}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
      >
        <div className="vslider-fill" style={{ width: `${position * 100}%` }} />
        <div className="vslider-thumb" style={{ left: `${position * 100}%` }} />
      </div>

      {hint && <p className="vslider-hint">{hint}</p>}
    </div>
  )
}
