import { animate, useMotionValue, useReducedMotion } from 'motion/react'
import { useEffect, useRef } from 'react'

export interface NumberRollProps {
  value: number
  /** How to render the current animated value. Defaults to a rounded integer. */
  format?: (n: number) => string
  /** Seconds. Longer than a spring on purpose — the count-up *is* the content. */
  duration?: number
  className?: string
  /** Announced to screen readers instead of the rolling digits. */
  ariaLabel?: string
}

/**
 * A number that counts to its value instead of snapping to it.
 *
 * Used for every projection in the app, because the count-up is doing real
 * pedagogical work: watching $500/month become $1.3M over four seconds communicates
 * the shape of exponential growth in a way the final figure alone never does.
 *
 * The DOM text is written directly from the motion value, so a four-second count
 * costs zero React renders. Assistive tech gets the final value immediately via
 * aria-label rather than a stream of meaningless intermediate numbers.
 */
export function NumberRoll({
  value,
  format = (n) => Math.round(n).toLocaleString('en-US'),
  duration = 1.1,
  className = '',
  ariaLabel,
}: NumberRollProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const mv = useMotionValue(0)
  const reduced = useReducedMotion()

  useEffect(() => {
    const node = ref.current
    if (!node) return

    if (reduced) {
      node.textContent = format(value)
      mv.set(value)
      return
    }

    const controls = animate(mv, value, {
      duration,
      // Decelerating ease: fast at first, settling into the final figure. A spring
      // would overshoot, and a number that briefly shows more money than it should
      // is not a mistake worth making in a finance app.
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => {
        node.textContent = format(v)
      },
    })

    return () => controls.stop()
  }, [value, duration, format, mv, reduced])

  return (
    <span
      ref={ref}
      className={`num ${className}`}
      role="text"
      aria-label={ariaLabel ?? format(value)}
    >
      {format(0)}
    </span>
  )
}
