/*
 * The geometry below is exported from this file rather than a sibling because
 * it is the control's definition of what a finger position means — it belongs
 * next to the handler that calls it, and it is what the tests exercise, since
 * this suite cannot mount React.
 */
/* eslint-disable react/only-export-components */
import { useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { haptic } from '../lib/haptics'
import './BlockBar.css'

/* -------------------------------------------------------------------------- */
/* Geometry — pure, exported, and the only thing that decides what a finger     */
/* position means. Kept free of React so it can be tested exhaustively.         */
/* -------------------------------------------------------------------------- */

/**
 * Tolerance for the float noise in `span / step`. A range of 0–15 in steps of
 * 0.1 divides evenly in decimal and not at all in binary, and a lattice that
 * loses its last position to a 1e-16 remainder makes the top of the bar
 * unreachable.
 */
const FUZZ = 1e-9

/** Kills accumulated float error so a value lands *on* its step, not near it. */
function clean(n: number): number {
  if (!Number.isFinite(n)) return n
  // 10 decimals is far finer than any variable this product drags (cents at
  // worst) and coarse enough to absorb every error a few multiplications add.
  return Number(n.toFixed(10))
}

interface Lattice {
  span: number
  /** Index of the last whole step. */
  lastK: number
  /** True when `max` does not sit on a step and therefore needs its own slot. */
  tail: boolean
  /** Index of the topmost position, `tail` included. */
  lastIndex: number
  step: number
}

/**
 * The set of values this control can produce: `min + k*step` for every whole k,
 * plus `max` itself.
 *
 * Native range inputs drop a `max` that no step lands on — a 0–10 slider in
 * steps of 3 simply cannot be set to 10. Here the ends of the bar are the two
 * most meaningful answers in the call (contribute nothing / contribute
 * everything), so the remainder gets its own short position rather than being
 * truncated away.
 */
function lattice(min: number, max: number, step: number): Lattice {
  const span = max - min
  if (!Number.isFinite(span) || span <= 0) {
    return { span: 0, lastK: 0, tail: false, lastIndex: 0, step: 1 }
  }
  // A nonsense step degrades to "ends only" rather than to NaN or a hang.
  const s = Number.isFinite(step) && step > 0 ? step : span
  const q = span / s
  const slack = FUZZ * Math.max(1, q)
  const lastK = Math.floor(q + slack)
  const tail = q - lastK > slack
  return { span, lastK, tail, lastIndex: lastK + (tail ? 1 : 0), step: s }
}

/** Number of discrete positions on the bar, both ends included. */
export function positionCount(min: number, max: number, step: number): number {
  return lattice(min, max, step).lastIndex + 1
}

/** Nearest legal index for a distance `raw` above `min`. */
function indexAt(raw: number, L: Lattice): number {
  const k = Math.min(L.lastK, Math.max(0, Math.round(raw / L.step)))
  // The tail slot wins only once the position is genuinely closer to `max`.
  if (L.tail && Math.abs(L.span - raw) < Math.abs(k * L.step - raw)) return L.lastIndex
  return k
}

function valueAtIndex(i: number, min: number, max: number, L: Lattice): number {
  if (L.tail && i >= L.lastIndex) return max
  return clean(min + Math.min(i, L.lastK) * L.step)
}

/**
 * Maps a 0–1 position along the bar to the nearest legal value.
 *
 * Nearest-neighbour, not floor: quantizing down would make the top position
 * half as wide as the others and the bottom one twice as wide, so the value
 * would visibly lag the thumb near the ends. Because it is a pure function of
 * the fraction, the same finger position always yields the same value — the
 * anti-jitter guarantee the whole drag depends on.
 */
export function positionToValue(fraction: number, min: number, max: number, step: number): number {
  const L = lattice(min, max, step)
  if (L.span <= 0) return min
  const f = Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0
  return valueAtIndex(indexAt(f * L.span, L), min, max, L)
}

/** Where a value sits along the bar, 0–1. Inverse of `positionToValue`. */
export function valueToFraction(value: number, min: number, max: number): number {
  const span = max - min
  if (!Number.isFinite(span) || span <= 0 || !Number.isFinite(value)) return 0
  return Math.min(1, Math.max(0, (value - min) / span))
}

/**
 * Moves `value` by whole positions along the lattice — the keyboard's step.
 *
 * Index arithmetic rather than `value + delta*step` so that arrowing off the
 * tail position lands on the last whole step instead of overshooting past it.
 */
export function stepValue(
  value: number,
  positions: number,
  min: number,
  max: number,
  step: number,
): number {
  const L = lattice(min, max, step)
  if (L.span <= 0) return min
  const clamped = Math.min(max, Math.max(min, Number.isFinite(value) ? value : min))
  const from = indexAt(clamped - min, L)
  const to = Math.min(L.lastIndex, Math.max(0, from + Math.trunc(positions)))
  return valueAtIndex(to, min, max, L)
}

/**
 * Whether a visual block is filled at this value.
 *
 * Rounded, not ceiled: a full bar must mean "at the top", and with fewer blocks
 * than steps `ceil` would paint every block solid while a tenth of the range
 * remained. The block count is a readout of the value, not a promise that every
 * step moves a block.
 */
export function blockState(
  blockIndex: number,
  blockCount: number,
  value: number,
  min: number,
  max: number,
): 'filled' | 'empty' {
  if (!Number.isFinite(blockCount) || blockCount <= 0) return 'empty'
  const filled = Math.round(valueToFraction(value, min, max) * blockCount)
  return blockIndex >= 0 && blockIndex < filled ? 'filled' : 'empty'
}

/**
 * The step positions a single block stands for, inclusive.
 *
 * The bar draws at most fifteen blocks whatever the control's range, so on a
 * call with twenty-five steps one block covers nearly two of them. Compute
 * functions author their tints against step positions — "everything before
 * month twelve is free" — which is the only unit they know, so the block index
 * has to be translated before it reaches them. Without this the boundary lands
 * wherever the ratio happens to put it: the promo call's free window was being
 * painted across eighty percent of a bar that is half free.
 *
 * Returns `[lo, hi]` with `lo <= hi`. A block narrower than the gap between two
 * positions collapses to the nearest single one.
 */
export function blockPositions(
  blockIndex: number,
  blockCount: number,
  positions: number,
): [number, number] {
  const last = Math.max(0, positions - 1)
  if (blockCount <= 0) return [0, 0]
  const span = last / blockCount
  const lo = Math.ceil(blockIndex * span)
  const hi = Math.floor((blockIndex + 1) * span)
  if (lo > hi) {
    const near = Math.round((blockIndex + 0.5) * span)
    return [Math.min(last, near), Math.min(last, near)]
  }
  return [Math.min(last, lo), Math.min(last, hi)]
}

/** Ranked worst-first, so a block spanning a boundary can be resolved. */
const TINT_RANK = { loss: 0, plain: 1, accent: 2 } as const

/**
 * One tint for a block that may straddle a boundary: the worst of what it covers.
 *
 * A block half inside a penalty zone is drawn as penalty. The alternative is a
 * control that paints part of a cliff in the safe colour, which is the one
 * mistake a teaching control must never make.
 */
export function blockTintFor(
  blockIndex: number,
  blockCount: number,
  positions: number,
  tint: (position: number) => 'accent' | 'plain' | 'loss',
): 'accent' | 'plain' | 'loss' {
  const [lo, hi] = blockPositions(blockIndex, blockCount, positions)
  let worst: 'accent' | 'plain' | 'loss' = 'accent'
  for (let p = lo; p <= hi; p++) {
    const t = tint(p)
    if (TINT_RANK[t] < TINT_RANK[worst]) worst = t
  }
  return worst
}

/* -------------------------------------------------------------------------- */
/* The control                                                                 */
/* -------------------------------------------------------------------------- */

export interface BlockBarProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  /** Visual segment count. Independent of the number of steps. */
  blocks?: number
  /**
   * Fill colour by STEP POSITION — 0 is `min`, not the index of a drawn block.
   * The bar translates, because it draws fewer blocks than most calls have
   * steps and only the call knows where its boundaries are.
   */
  tint?: (position: number) => 'accent' | 'plain' | 'loss'
  disabled?: boolean
  /** Accessible name. The screen's own caption, not a second one. */
  label: string
  /** Spoken value. Falls back to the bare number. */
  format?: (value: number) => string
}

/**
 * The drag control. This is the product.
 *
 * The whole bar is the hit area — there is no thumb to find, you put your
 * finger down anywhere and the value is already there. Every input route
 * (touch, mouse, pen, keyboard) goes through one pointer path and one commit
 * function, so they cannot drift apart in feel or in what they emit.
 *
 * `onChange` is called straight out of the pointermove handler. No debounce, no
 * rAF: the dependent figures on the call screen have to move *with* the thumb,
 * because that simultaneity is the entire lesson.
 */
export function BlockBar({
  value,
  onChange,
  min,
  max,
  step,
  blocks = 15,
  tint,
  disabled = false,
  label,
  format,
}: BlockBarProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  /**
   * Track geometry, measured once per drag.
   *
   * Reading the rect on every move would force a layout flush inside the
   * handler — against a screen whose numbers are all rewriting on the same
   * frame, that is exactly where the jank would come from. Caching also makes
   * the mapping immune to anything that reflows mid-drag, which is the other
   * half of the no-jitter guarantee.
   */
  const geom = useRef<{ left: number; width: number } | null>(null)
  const pointerId = useRef<number | null>(null)
  /**
   * Last value handed to `onChange`; gates both re-emits and haptic ticks.
   *
   * A drag cannot compare against the `value` prop: React may not have
   * re-rendered this component by the time the next pointermove arrives, and a
   * stale prop would re-fire the tick haptic for a crossing already made. It is
   * re-seeded from the prop whenever a gesture starts, which is the only moment
   * the prop is guaranteed current.
   */
  const emitted = useRef(value)

  const count = Math.max(1, Math.round(blocks))
  // How many legal stops the control actually has, which is usually more than
  // the number of blocks drawn for it.
  const positions = positionCount(min, max, step)

  const commit = (next: number, tick: boolean) => {
    if (next === emitted.current) return
    emitted.current = next
    if (tick) haptic('selection')
    onChange(next)
  }

  const valueAt = (clientX: number) => {
    const g = geom.current
    if (!g || g.width <= 0) return value
    return positionToValue((clientX - g.left) / g.width, min, max, step)
  }

  const end = (el: HTMLElement, id: number) => {
    if (pointerId.current !== id) return
    pointerId.current = null
    geom.current = null
    // Written imperatively, and deliberately not rendered as a JSX prop: a
    // value change mid-drag re-renders this component, and React would put the
    // attribute back to whatever the last render said. The blocks must lose
    // their fill transition for the whole gesture, not for one frame of it.
    delete el.dataset.dragging
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    // Ignore a second finger: two pointers fighting over one value is worse
    // than the second one doing nothing.
    if (disabled || !e.isPrimary || pointerId.current !== null) return
    if (e.pointerType === 'mouse' && e.button !== 0) return
    const track = trackRef.current
    if (!track) return

    emitted.current = value
    const r = track.getBoundingClientRect()
    geom.current = { left: r.left, width: r.width }
    pointerId.current = e.pointerId

    const host = e.currentTarget
    try {
      // Capture, so a finger that slides off the bar — or off the screen —
      // keeps driving the value and still delivers its pointerup.
      host.setPointerCapture(e.pointerId)
    } catch {
      /* pointer already gone; the handlers below no-op */
    }
    host.dataset.dragging = 'true'
    // Arrow keys should work on the thing you just touched.
    host.focus({ preventScroll: true })

    // One firm haptic for the grab, whether or not the value moved; the
    // per-step ticks start from the next crossing. A tap is therefore a jump
    // plus exactly one commit haptic, rather than a tick and a commit 80ms
    // apart, which on a short press reads as a stutter.
    haptic('light')
    commit(valueAt(e.clientX), false)
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (pointerId.current !== e.pointerId) return
    commit(valueAt(e.clientX), true)
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    const host = e.currentTarget
    if (pointerId.current !== e.pointerId) return
    try {
      host.releasePointerCapture(e.pointerId)
    } catch {
      /* already released */
    }
    end(host, e.pointerId)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return
    // Key events are discrete, so React has already flushed the last one.
    emitted.current = value

    // A tenth of the bar, rounded to whole positions, minimum one.
    const page = Math.max(1, Math.round((positionCount(min, max, step) - 1) / 10))

    let next: number
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        next = stepValue(value, 1, min, max, step)
        break
      case 'ArrowLeft':
      case 'ArrowDown':
        next = stepValue(value, -1, min, max, step)
        break
      case 'PageUp':
        next = stepValue(value, page, min, max, step)
        break
      case 'PageDown':
        next = stepValue(value, -page, min, max, step)
        break
      case 'Home':
        next = min
        break
      case 'End':
        next = max
        break
      default:
        return
    }
    // Stops arrows scrolling the shell and PageUp paging it.
    e.preventDefault()
    commit(next, true)
  }

  return (
    <div
      className="blockbar"
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-orientation="horizontal"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={format ? format(value) : String(value)}
      aria-disabled={disabled || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onLostPointerCapture={(e) => end(e.currentTarget, e.pointerId)}
      onKeyDown={onKeyDown}
    >
      <div className="blockbar-track" ref={trackRef}>
        {Array.from({ length: count }, (_, i) => (
          <span
            key={i}
            className="blockbar-block"
            data-state={blockState(i, count, value, min, max)}
            data-tint={tint ? blockTintFor(i, count, positions, tint) : 'plain'}
          />
        ))}
      </div>
    </div>
  )
}
