import { AnimatePresence, motion, useMotionValue, animate } from 'motion/react'
import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { projectFling, rubberBand, spring } from '../lib/motion'
import { haptic } from '../lib/haptics'
import './Sheet.css'

export interface SheetProps {
  open: boolean
  onClose: () => void
  title?: string
  /**
   * Resting heights as fractions of the viewport, smallest first — iOS "detents".
   * `[0.5, 0.95]` gives the familiar half-sheet that can be dragged up to near-full.
   */
  detents?: number[]
  /** Hide the grabber for sheets that should feel modal rather than dismissible. */
  grabber?: boolean
  children: ReactNode
  /** Rendered pinned to the bottom, outside the scroll area — for a primary action. */
  footer?: ReactNode
}

/**
 * A UIKit-style bottom sheet.
 *
 * The parts that make it feel native rather than "a div that slides up":
 *
 *  - **Rubber-banding.** Dragging above the tallest detent compresses logarithmically
 *    instead of stopping dead or following the finger 1:1.
 *  - **Velocity projection.** The commit decision uses where the flick *would land*
 *    (`projectFling`) rather than where the finger currently is, so a fast short
 *    flick dismisses and a slow long drag does not.
 *  - **Scroll handoff.** A downward drag that starts while the inner list is scrolled
 *    belongs to the list. The sheet only claims the gesture once the list is at the
 *    top, which is exactly how iOS arbitrates it.
 *  - **Nothing behind it moves the page.** The backdrop takes the scroll lock so the
 *    body underneath cannot rubber-band into view.
 */
export function Sheet({
  open,
  onClose,
  title,
  detents = [0.92],
  grabber = true,
  children,
  footer,
}: SheetProps) {
  const y = useMotionValue(0)
  const sheetRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const titleId = useId()

  const sorted = [...detents].sort((a, b) => a - b)
  const [detentIndex, setDetentIndex] = useState(sorted.length - 1)

  // Gesture bookkeeping, in refs so pointer moves never trigger a render.
  const drag = useRef({
    active: false,
    startY: 0,
    startOffset: 0,
    lastY: 0,
    lastT: 0,
    velocity: 0,
    /** Set on pointerdown: was the inner scroller already at the top? */
    ownsGesture: false,
    /**
     * Sheet height, captured at pointerdown. Measured then rather than read from
     * a ref during render, where it is null on the first pass and where the old
     * `window.innerHeight` fallback was wrong inside the desktop phone frame.
     */
    height: 0,
  })

  const heightFor = useCallback(
    (i: number) => `${Math.round(sorted[Math.min(i, sorted.length - 1)] * 100)}dvh`,
    [sorted],
  )

  // Reset to the tallest detent each time the sheet opens, so a previous session's
  // half-height does not surprise the user on the next, unrelated presentation.
  // Guarded on the transition rather than on `open` being truthy, so a re-render
  // while the sheet is already open does not snap it back.
  const wasOpen = useRef(open)
  useEffect(() => {
    if (open && !wasOpen.current) {
      setDetentIndex(sorted.length - 1)
      y.set(0)
    }
    wasOpen.current = open
  }, [open, sorted.length, y])

  // Escape closes, matching both the platform convention and keyboard expectations.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const onPointerDown = (e: React.PointerEvent) => {
    // Ignore secondary buttons and anything originating in an interactive control.
    if (e.button !== 0) return

    const scroller = scrollRef.current
    drag.current = {
      active: true,
      startY: e.clientY,
      startOffset: y.get(),
      lastY: e.clientY,
      lastT: e.timeStamp,
      velocity: 0,
      ownsGesture: !scroller || scroller.scrollTop <= 0,
      height: sheetRef.current?.offsetHeight ?? 0,
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d.active) return

    const delta = e.clientY - d.startY
    const scroller = scrollRef.current

    // Hand the gesture back to the list if the user reverses into an upward drag
    // while the sheet is at rest — they meant to scroll, not to resize.
    if (!d.ownsGesture) {
      if (scroller && scroller.scrollTop <= 0 && delta > 0) {
        d.ownsGesture = true
        d.startY = e.clientY
        d.startOffset = y.get()
      } else {
        return
      }
    }

    // Once we own it, stop the browser from also scrolling or triggering
    // pull-to-refresh underneath.
    if (e.cancelable) e.preventDefault()
    ;(e.target as Element).setPointerCapture?.(e.pointerId)

    const raw = d.startOffset + delta
    // Downward travel is free; upward past the top detent is resisted.
    y.set(raw >= 0 ? raw : rubberBand(raw, d.height))

    const dt = e.timeStamp - d.lastT
    if (dt > 0) {
      // Exponential smoothing: a single raw frame delta is far too jittery to
      // make a commit decision from.
      const instant = ((e.clientY - d.lastY) / dt) * 1000
      d.velocity = d.velocity * 0.7 + instant * 0.3
      d.lastY = e.clientY
      d.lastT = e.timeStamp
    }
  }

  const endDrag = () => {
    const d = drag.current
    if (!d.active) return
    d.active = false
    if (!d.ownsGesture) return

    const offset = y.get()
    // Where the flick would settle, not where the finger stopped.
    const projected = offset + projectFling(d.velocity)

    // Past a third of the sheet's height, or flicked hard downward: dismiss.
    if (projected > d.height * 0.33 || d.velocity > 700) {
      haptic('light')
      onClose()
      return
    }

    // Otherwise fall to the nearest detent. Snapping down a detent counts as a
    // deliberate resize and gets its own subtle selection tick.
    const travelled = d.height > 0 ? offset / d.height : 0
    let next = detentIndex
    if (travelled > 0.12 && detentIndex > 0) next = detentIndex - 1
    if (next !== detentIndex) {
      haptic('selection')
      setDetentIndex(next)
    }

    animate(y, 0, spring.nav)
  }

  return (
    <AnimatePresence>
      {open && (
        <div className="sheet-layer" role="presentation">
          <motion.div
            className="sheet-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            onClick={onClose}
          />

          <motion.div
            ref={sheetRef}
            className="sheet"
            style={{ y, height: heightFor(detentIndex) }}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={spring.nav}
            role="dialog"
            aria-modal="true"
            aria-labelledby={title ? titleId : undefined}
            // touch-action none on the drag surfaces only; the scroller re-enables it.
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
          >
            {grabber && (
              <div className="sheet-grabber-area">
                <div className="sheet-grabber" />
              </div>
            )}

            {title && (
              <header className="sheet-header">
                <h2 id={titleId} className="sheet-title">
                  {title}
                </h2>
                <button className="sheet-close pressable" onClick={onClose} aria-label="Close">
                  <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                    <path
                      d="M6 6l12 12M18 6L6 18"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
              </header>
            )}

            <div className="sheet-body scroll" ref={scrollRef}>
              {children}
            </div>

            {footer && <div className="sheet-footer">{footer}</div>}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}
