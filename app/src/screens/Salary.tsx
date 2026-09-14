import { useRef, useState } from 'react'
import { haptic } from '../lib/haptics'
import { clamp, money } from '../lib/format'
import { DEFAULT_PROFILE, type Profile } from '../calls/types'
import './Salary.css'

/**
 * Making the figures yours. Optional, and reached only after the first answer.
 *
 * This used to be the cold open: a stranger arriving from a shared link was
 * asked for her income by an unnamed thing, before she had been told the
 * category, the benefit or the time cost. Twenty-one reviewers put that near
 * the top of why the product made no sense. The ask has not got softer — it
 * has got later. By the time anyone sees this screen they have watched a
 * question recompute on the sample numbers, so they know exactly what a real
 * number would buy them, which is the only honest reason to hand one over.
 *
 * Age is here because the app was quietly wrong without it. Every "by 65"
 * figure is a compounding horizon, and the old build hardcoded 30 for
 * everybody — so a 52-year-old was told she had thirty-five years of growth
 * ahead of her, which overstates the projection by roughly a factor of three.
 * That is not a missing nicety, it is a wrong answer presented confidently.
 */

const PAY_MIN = 15_000
const PAY_MAX = 400_000
const PAY_STEP = 1_000

const AGE_MIN = 18
const AGE_MAX = 70
const AGE_STEP = 1

/* The visible label and the slider's accessible name are the same words, and
   they are written once so they cannot drift apart. */
const PAY_LABEL = 'What you make a year'
const AGE_LABEL = 'Your age'

/**
 * Rough pay bands, and the figure each one stands for.
 *
 * A reader put the objection plainly: "I am not typing my real pay, privacy
 * line or not. Give me a rough band to tap and I would do it." A band is a
 * far smaller confession than a number, and it costs the maths almost
 * nothing — every figure downstream is already an estimate, and landing on
 * the middle of the right band moves them by a few percent, well inside the
 * noise of a forty-year projection. Anyone who wants to be exact still has
 * the dial underneath.
 *
 * `floor` is the bottom of the band and `value` the midpoint it sets. The top
 * band is open-ended, so its figure is a representative one rather than a
 * midpoint of anything.
 */
const BANDS = [
  { label: 'Under $30K', floor: PAY_MIN, value: 22_000 },
  { label: '$30–50K', floor: 30_000, value: 40_000 },
  { label: '$50–75K', floor: 50_000, value: 62_000 },
  { label: '$75–120K', floor: 75_000, value: 97_000 },
  { label: '$120K+', floor: 120_000, value: 150_000 },
] as const

/** Log for pay, straight for age. See `Dial`. */
type Scale = 'log' | 'linear'

/**
 * One dial. Both of this screen's inputs are the same control, because they
 * are the same gesture and a person should only have to learn it once.
 *
 * The whole bar is the hit area — there is no thumb to find, you put your
 * thumb where you want the value, exactly as the question control behaves.
 */
function Dial({
  label,
  caption,
  value,
  min,
  max,
  step,
  page,
  scale,
  format,
  onChange,
}: {
  label: string
  /** The words under the figure, and the tail of what a screen reader says. */
  caption: string
  value: number
  min: number
  max: number
  step: number
  /** The PageUp/PageDown jump: crossing the range without 300 key presses. */
  page: number
  scale: Scale
  format: (v: number) => string
  onChange: (v: number) => void
}) {
  const [dragging, setDragging] = useState(false)
  const trackRef = useRef<HTMLDivElement>(null)
  const lastTick = useRef(value)

  const lnMin = Math.log(min)
  const lnSpan = Math.log(max) - lnMin

  /**
   * Pay spans more than an order of magnitude, and on a linear track every
   * value under $60k — which is most people — is crammed into the first
   * eighth of the bar and unreachable with a thumb. Age spans 18 to 70 and
   * needs no such help; a log scale there would only make the last decade of
   * a working life harder to hit than the first.
   */
  const toFraction = (v: number) =>
    scale === 'log' ? (Math.log(v) - lnMin) / lnSpan : (v - min) / (max - min)

  const toValue = (f: number) => {
    const c = clamp(f, 0, 1)
    const raw = scale === 'log' ? Math.exp(c * lnSpan + lnMin) : min + c * (max - min)
    return clamp(Math.round(raw / step) * step, min, max)
  }

  const commit = (next: number) => {
    if (next === value) return
    // One tick per step crossed, not per pointermove event. A drag across the
    // track fires dozens of moves, and ticking on each one is a continuous
    // buzz rather than the feel of counting past notches.
    if (next !== lastTick.current) {
      haptic('selection')
      lastTick.current = next
    }
    onChange(next)
  }

  const setFromPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect()
    if (!rect) return
    commit(toValue((clientX - rect.left) / rect.width))
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    const by: Record<string, number> = {
      ArrowRight: step,
      ArrowUp: step,
      ArrowLeft: -step,
      ArrowDown: -step,
      PageUp: page,
      PageDown: -page,
    }

    let next: number | null = null
    if (e.key === 'Home') next = min
    else if (e.key === 'End') next = max
    else if (e.key in by) next = clamp(value + by[e.key], min, max)
    if (next === null) return

    // Arrows and the Page keys scroll this screen otherwise, which slides the
    // control out from under the person operating it.
    e.preventDefault()
    commit(next)
  }

  return (
    <div className="salary-dial">
      <output className="salary-value num" data-dragging={dragging || undefined}>
        {format(value)}
      </output>
      <p className="salary-value-label data-sm">{caption}</p>

      <div
        className="salary-track"
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        // The figure and its caption are one reading on screen, so they are
        // one reading aloud. A slider that announced a bare "62000" would drop
        // the half that says what the number is.
        aria-valuetext={`${format(value)} ${caption}`}
        onKeyDown={onKeyDown}
        onPointerDown={(e) => {
          // Capture, so a thumb that slides off the bar — or off the screen —
          // keeps driving the value instead of dropping the drag.
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
        <div className="salary-fill" style={{ width: `${toFraction(value) * 100}%` }} />
      </div>
    </div>
  )
}

export function Salary({
  initial,
  onDone,
  onBack,
}: {
  initial: Profile
  onDone: (p: Profile) => void
  onBack: () => void
}) {
  // Clamped on the way in. A profile saved by an older build, or edited by
  // hand in local storage, must not render a fill wider than its own track.
  const [salary, setSalary] = useState(() => clamp(initial.salary, PAY_MIN, PAY_MAX))
  const [age, setAge] = useState(() => clamp(initial.age, AGE_MIN, AGE_MAX))

  /**
   * The sample figures are only "the sample figures" the first time.
   *
   * Someone who has already given us a number and comes back to change it
   * would be told a flat lie by the first-run line — and this is the one
   * screen in the product that asks to be trusted with something.
   */
  const untouched =
    initial.salary === DEFAULT_PROFILE.salary && initial.age === DEFAULT_PROFILE.age

  /**
   * Which band holds the current figure, whichever control put it there.
   *
   * Derived rather than remembered: if a tapped band were stored as its own
   * state, dragging the dial afterwards would leave the wrong chip lit, and
   * the two controls would visibly disagree about one value.
   */
  const band = BANDS.findLast((b) => salary >= b.floor)

  return (
    <div className="salary">
      <div className="salary-body scroll">
        <h1 className="salary-q">
          {untouched
            ? `Those figures were on ${money(initial.salary)}. Make them yours?`
            : `Your figures are on ${money(initial.salary)}. Change them?`}
        </h1>

        <section className="salary-field">
          <h2 className="salary-k data-sm">{PAY_LABEL}</h2>

          <div className="salary-bands">
            {BANDS.map((b) => (
              <button
                key={b.label}
                className="salary-band press"
                aria-pressed={b === band}
                onClick={() => {
                  haptic('selection')
                  setSalary(b.value)
                }}
              >
                {b.label}
              </button>
            ))}
          </div>

          <p className="salary-hint">
            A band is close enough. Drag the bar below if you want it exact.
          </p>

          <Dial
            label={PAY_LABEL}
            caption="before tax"
            value={salary}
            min={PAY_MIN}
            max={PAY_MAX}
            step={PAY_STEP}
            page={10_000}
            scale="log"
            format={money}
            onChange={setSalary}
          />
        </section>

        <section className="salary-field">
          <h2 className="salary-k data-sm">{AGE_LABEL}</h2>

          <Dial
            label={AGE_LABEL}
            caption="years old"
            value={age}
            min={AGE_MIN}
            max={AGE_MAX}
            step={AGE_STEP}
            page={5}
            scale="linear"
            format={String}
            onChange={setAge}
          />

          <p className="salary-why">
            Every “by 65” figure depends on this. Without it we would guess 30, and we
            would rather ask.
          </p>
        </section>

        <p className="salary-note">
          These stay on your phone. We never send them anywhere.
        </p>
      </div>

      <div className="salary-actions">
        <button
          className="salary-go press"
          onClick={() => {
            haptic('medium')
            // Spread, so anything the parent is carrying on the profile that
            // this screen does not ask about survives the round trip.
            onDone({ ...initial, salary, age })
          }}
        >
          Use my numbers
        </button>
        <button
          className="salary-skip press"
          onClick={() => {
            haptic('light')
            onBack()
          }}
        >
          Not now
        </button>
      </div>
    </div>
  )
}
