import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Screen } from '../ui/Screen'
import { useNav } from '../ui/NavStack'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { ValueSlider } from '../ui/ValueSlider'
import { GrowthChart } from '../ui/GrowthChart'
import { NumberRoll } from '../ui/NumberRoll'
import { useStore } from '../state/store'
import { lessonById } from '../data/lessons'
import type { Beat, PracticeItem } from '../data/lesson-types'
import { money, moneyCompact, percent } from '../lib/format'
import { haptic } from '../lib/haptics'
import { spring } from '../lib/motion'
import { Mechanism } from './Mechanism'
import './LessonPlayer.css'

/**
 * Runs one lesson, one beat per screen.
 *
 * The invariant worth protecting: the learner cannot reach the explanation without
 * first committing to an answer. That single ordering constraint is what separates
 * this from a slideshow with a quiz on the end — reading first produces recognition
 * and the confident feeling of having known it all along, which is precisely the
 * illusion this format exists to prevent.
 */
export function LessonPlayer({ lessonId }: { lessonId: string }) {
  const { state, completeLesson, gradeConcept, commit } = useStore()
  const nav = useNav()
  const lesson = lessonById(lessonId)
  const beats = useMemo(() => lesson?.build(state.profile) ?? [], [lesson, state.profile])

  const [index, setIndex] = useState(0)
  const [score, setScore] = useState({ correct: 0, total: 0 })

  if (!lesson) return null

  const beat = beats[index]
  const isLast = index === beats.length - 1
  const progress = (index + 1) / beats.length

  const advance = () => {
    if (isLast) return
    haptic('light')
    setIndex((i) => i + 1)
  }

  const finish = () => {
    completeLesson(lesson.id, score.correct, Math.max(1, score.total))
    for (const concept of lesson.concepts) {
      // A lesson finished with most answers right starts the concept on a longer
      // first interval than one scraped through.
      gradeConcept(concept, score.total === 0 || score.correct / score.total >= 0.75 ? 'good' : 'hard')
    }
    nav.pop()
  }

  return (
    <Screen
      title={lesson.title}
      inlineTitle
      left={
        <button className="navbtn navbtn--plain" onClick={() => nav.pop()}>
          Close
        </button>
      }
      noTabBar
      footer={
        <LessonFooter
          beat={beat}
          isLast={isLast}
          onAdvance={advance}
          onFinish={finish}
          onCommit={(when, then, worth) => {
            commit({ lessonId: lesson.id, when, then, worth })
            haptic('success')
            finish()
          }}
        />
      }
    >
      <div className="lp-progress" aria-hidden="true">
        <div className="lp-progress-fill" style={{ transform: `scaleX(${progress})` }} />
      </div>

      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={index}
          className="lp-beat"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={spring.smooth}
        >
          <BeatView
            beat={beat}
            onScore={(correct) =>
              setScore((s) => ({ correct: s.correct + (correct ? 1 : 0), total: s.total + 1 }))
            }
          />
        </motion.div>
      </AnimatePresence>
    </Screen>
  )
}

/* ------------------------------------------------------------------------- */

function BeatView({ beat, onScore }: { beat: Beat; onScore: (correct: boolean) => void }) {
  switch (beat.kind) {
    case 'anchor':
      return (
        <div className="lp-anchor">
          <p className="lp-kicker">The situation</p>
          <p className="lp-anchor-body selectable">{beat.body}</p>
          {beat.note && <p className="lp-note">{beat.note}</p>}
        </div>
      )

    case 'probe':
      return <Probe beat={beat} onScore={onScore} />

    case 'reveal':
      return <Reveal beat={beat} />

    case 'mechanism':
      return (
        <div className="lp-mechanism">
          <p className="lp-kicker">Why</p>
          <p className="lp-mechanism-sentence selectable">{beat.sentence}</p>
          <Mechanism visual={beat.visual} />
          <p className="lp-mechanism-detail selectable">{beat.detail}</p>
        </div>
      )

    case 'worked':
      return <Worked beat={beat} />

    case 'practice':
      return <Practice beat={beat} onScore={onScore} />

    case 'rule':
      return (
        <div className="lp-rule">
          <p className="lp-kicker">Keep this</p>
          <h3 className="lp-rule-name">{beat.name}</h3>
          <p className="lp-rule-statement selectable">{beat.statement}</p>
          <div className="lp-rule-example selectable">{beat.example}</div>
        </div>
      )

    case 'action':
      return (
        <div className="lp-action">
          <p className="lp-kicker">One thing</p>
          <p className="lp-action-text selectable">
            <span className="lp-action-when">When {beat.when},</span> {beat.then}.
          </p>
          <p className="lp-note">
            Tied to something that will actually happen, rather than to remembering. That is the
            difference between an intention and a plan.
          </p>
        </div>
      )
  }
}

/* ---- Prediction probe ------------------------------------------------------ */

function Probe({
  beat,
  onScore,
}: {
  beat: Extract<Beat, { kind: 'probe' }>
  onScore: (correct: boolean) => void
}) {
  const [guess, setGuess] = useState(beat.mode === 'estimate' ? beat.start : 0)
  const [locked, setLocked] = useState(false)
  const [picked, setPicked] = useState<number | null>(null)

  const fmt = (v: number) => {
    if (beat.mode !== 'estimate') return `${v}`
    if (beat.unit === 'usd') return moneyCompact(v)
    if (beat.unit === 'percent') return `${v.toFixed(1)}%`
    if (beat.unit === 'years') return `${v.toFixed(1)} years`
    return `${Math.round(v)}`
  }

  if (beat.mode === 'choice') {
    return (
      <div className="lp-probe">
        <p className="lp-kicker">Your call, before we say anything</p>
        <h3 className="lp-probe-question">{beat.question}</h3>

        <ul className="lp-probe-options">
          {beat.options.map((o, i) => (
            <li key={o.label}>
              <button
                className="lp-probe-option"
                data-state={
                  picked === null ? undefined : o.correct ? 'correct' : picked === i ? 'wrong' : 'dimmed'
                }
                disabled={picked !== null}
                onClick={() => {
                  haptic(o.correct ? 'success' : 'error')
                  setPicked(i)
                  onScore(!!o.correct)
                }}
              >
                {o.label}
              </button>
              {picked !== null && picked === i && o.misconception && (
                <p className="lp-probe-why">{o.misconception}</p>
              )}
            </li>
          ))}
        </ul>

        {picked !== null && <p className="lp-probe-because selectable">{beat.because}</p>}
      </div>
    )
  }

  const error = Math.abs(guess - beat.answer) / Math.max(1, Math.abs(beat.answer))
  const close = error <= beat.tolerance

  return (
    <div className="lp-probe">
      <p className="lp-kicker">Your call, before we say anything</p>
      <h3 className="lp-probe-question">{beat.question}</h3>

      <div className="lp-probe-slider">
        <ValueSlider
          label="Your estimate"
          value={guess}
          onChange={setGuess}
          min={beat.min}
          max={beat.max}
          step={Math.max(1, Math.round((beat.max - beat.min) / 200))}
          format={fmt}
          curve={beat.unit === 'usd' ? 'log' : 'linear'}
          disabled={locked}
        />
      </div>

      {!locked ? (
        <Button
          block
          size="lg"
          onClick={() => {
            haptic(close ? 'success' : 'warning')
            setLocked(true)
            onScore(close)
          }}
        >
          Lock it in
        </Button>
      ) : (
        <motion.div
          className="lp-probe-result"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={spring.nav}
        >
          <div className="lp-probe-compare">
            <div>
              <p className="lp-probe-compare-label">You said</p>
              <p className="lp-probe-compare-value num">{fmt(guess)}</p>
            </div>
            <div>
              <p className="lp-probe-compare-label">Actually</p>
              <p className="lp-probe-compare-value num lp-probe-compare-value--real">
                <NumberRoll value={beat.answer} format={fmt} duration={1.3} />
              </p>
            </div>
          </div>

          <p className="lp-probe-verdict" data-close={close || undefined}>
            {close
              ? 'Within range — you already have the intuition for this one.'
              : guess < beat.answer
                ? `You were low by ${percent(error, 0)}.`
                : `You were high by ${percent(error, 0)}.`}
          </p>

          <p className="lp-probe-because selectable">{beat.because}</p>
        </motion.div>
      )}
    </div>
  )
}

/* ---- Gap reveal ------------------------------------------------------------ */

function Reveal({ beat }: { beat: Extract<Beat, { kind: 'reveal' }> }) {
  const chart = beat.chart?.()
  return (
    <div className="lp-reveal">
      <h3 className="lp-reveal-headline num">{beat.headline}</h3>
      <p className="lp-reveal-body selectable">{beat.body}</p>

      {chart && (
        <GrowthChart
          points={chart.points}
          comparison={chart.comparison}
          comparisonLabel={chart.comparisonLabel}
          height={210}
        />
      )}

      {beat.contrast && (
        <div className="lp-contrast">
          {beat.contrast.map((c) => (
            <div key={c.label} className="lp-contrast-item" data-tone={c.tone}>
              <p className="lp-contrast-label">{c.label}</p>
              <p className="lp-contrast-value num">{c.value}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ---- Worked example -------------------------------------------------------- */

function Worked({ beat }: { beat: Extract<Beat, { kind: 'worked' }> }) {
  const [filled, setFilled] = useState<Record<number, boolean>>({})

  return (
    <div className="lp-worked">
      <p className="lp-kicker">Step by step</p>
      <p className="lp-worked-setup selectable">{beat.setup}</p>

      <ol className="lp-worked-steps">
        {beat.steps.map((s, i) => {
          const hidden = s.blankable && !filled[i]
          return (
            <li key={s.label} className="lp-worked-step">
              <span className="lp-worked-label">{s.label}</span>
              {hidden ? (
                <button
                  className="lp-worked-blank"
                  onClick={() => {
                    haptic('light')
                    setFilled((f) => ({ ...f, [i]: true }))
                  }}
                >
                  tap to check
                </button>
              ) : (
                <span className="lp-worked-value num">{s.value}</span>
              )}
            </li>
          )
        })}
      </ol>

      <p className="lp-worked-conclusion selectable">{beat.conclusion}</p>
    </div>
  )
}

/* ---- Practice -------------------------------------------------------------- */

function Practice({
  beat,
  onScore,
}: {
  beat: Extract<Beat, { kind: 'practice' }>
  onScore: (correct: boolean) => void
}) {
  const [answers, setAnswers] = useState<Record<number, number>>({})

  return (
    <div className="lp-practice">
      <p className="lp-kicker">Try it</p>
      {beat.items.map((item, qi) => (
        <PracticeQuestion
          key={item.prompt}
          item={item}
          picked={answers[qi]}
          onPick={(oi, correct) => {
            setAnswers((a) => ({ ...a, [qi]: oi }))
            onScore(correct)
          }}
        />
      ))}
    </div>
  )
}

function PracticeQuestion({
  item,
  picked,
  onPick,
}: {
  item: PracticeItem
  picked: number | undefined
  onPick: (optionIndex: number, correct: boolean) => void
}) {
  return (
    <div className="lp-q">
      {item.transfer && <span className="lp-q-tag">New situation</span>}
      <p className="lp-q-prompt selectable">{item.prompt}</p>

      <ul className="lp-q-options">
        {item.options.map((o, oi) => (
          <li key={o.label}>
            <button
              className="lp-q-option"
              data-state={
                picked === undefined
                  ? undefined
                  : o.correct
                    ? 'correct'
                    : picked === oi
                      ? 'wrong'
                      : 'dimmed'
              }
              disabled={picked !== undefined}
              onClick={() => {
                haptic(o.correct ? 'success' : 'error')
                onPick(oi, !!o.correct)
              }}
            >
              {o.label}
            </button>
            {picked !== undefined && (picked === oi || o.correct) && (
              <p className="lp-q-why">{o.why}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ---- Footer ---------------------------------------------------------------- */

function LessonFooter({
  beat,
  isLast,
  onAdvance,
  onFinish,
  onCommit,
}: {
  beat: Beat
  isLast: boolean
  onAdvance: () => void
  onFinish: () => void
  onCommit: (when: string, then: string, worth: number | null) => void
}) {
  const { state } = useStore()

  if (beat.kind === 'action') {
    const worth = beat.worth?.(state.profile) ?? null
    return (
      <div className="lp-footer-actions">
        {worth != null && worth > 0 && (
          <p className="lp-footer-worth">
            Worth about <strong className="num">{money(worth)}</strong> on your numbers.
          </p>
        )}
        {beat.options.map((o) => (
          <Button
            key={o.label}
            block
            size={o.commits ? 'lg' : 'md'}
            variant={o.commits ? 'primary' : 'plain'}
            onClick={() => (o.commits ? onCommit(beat.when, beat.then, worth) : onFinish())}
            feedback={o.commits ? null : 'light'}
          >
            {o.label}
          </Button>
        ))}
      </div>
    )
  }

  return (
    <Button block size="lg" onClick={isLast ? onFinish : onAdvance} icon={undefined}>
      Continue
      <Icon name="chevron" size={14} stroke />
    </Button>
  )
}
