import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Screen } from '../ui/Screen'
import { BackButton } from '../ui/NavStack'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { useNav } from '../ui/NavStack'
import { useStore } from '../state/store'
import { DRILLS } from '../data/drills'
import { drillIndex, drillNumber, MAX_ATTEMPTS } from '../lib/daily'
import { dayKey } from '../lib/format'
import { haptic } from '../lib/haptics'
import { spring } from '../lib/motion'
import './Drill.css'

/**
 * One question a day.
 *
 * A teaching surface, not a growth loop. It once had a Wordle-style glyph to
 * share, and that was cut on a simple information-theoretic argument: three
 * attempts at a four-option question has three possible outcomes, so the glyph
 * carries almost no signal, the modal result is a perfect score, and it only
 * reads as a brag to someone who already recognises the format. Wordle's grid
 * works because it encodes five letters across six rows. This does not, and
 * shipping it would have been cargo-culting the shape without the substance.
 * The share artifact is the Gap Card from a drawn curve instead.
 *
 * What remains is the part that was always doing the work:
 *
 *  - **A wrong answer teaches immediately.** The option you picked explains why
 *    it was tempting, before you try again. For most people the near-miss is the
 *    thing they remember, so that is where the content goes.
 *  - **The reveal shows the arithmetic either way**, so losing is still worth the
 *    sixty seconds.
 *  - **No streak and no cadence pressure.** Missing a day costs nothing, because
 *    there is no daily-repetition skill here to protect.
 */
export function DrillScreen() {
  const { state, recordDrillAttempt } = useStore()
  const nav = useNav()
  const today = dayKey()
  const number = drillNumber(today)
  const drill = useMemo(() => DRILLS[drillIndex(number, DRILLS.length)], [number])

  const record = state.drills[number]
  const attempts = record?.attempts ?? []
  const solved = record?.solved ?? false
  const exhausted = attempts.length >= MAX_ATTEMPTS
  const finished = solved || exhausted

  // Which options the player has already ruled out this session.
  const [tried, setTried] = useState<number[]>([])
  const [shakeIndex, setShakeIndex] = useState<number | null>(null)

  const pick = (i: number) => {
    if (finished || tried.includes(i)) return
    const correct = drill.options[i].correct === true

    haptic(correct ? 'success' : 'error')
    recordDrillAttempt(number, correct)
    setTried((t) => [...t, i])
    if (!correct) {
      setShakeIndex(i)
      window.setTimeout(() => setShakeIndex(null), 420)
    }
  }

  const remaining = MAX_ATTEMPTS - attempts.length

  return (
    <Screen
      title={`Drill #${number}`}
      inlineTitle
      left={<BackButton label="Today" />}
      noTabBar
    >
      <div className="drill">
        <p className="drill-eyebrow">
          {finished
            ? 'Everyone gets the same question today'
            : `${remaining} ${remaining === 1 ? 'try' : 'tries'} left`}
        </p>

        <h2 className="drill-question selectable">{drill.question}</h2>

        <ul className="drill-options">
          {drill.options.map((option, i) => {
            const isTried = tried.includes(i)
            const revealCorrect = finished && option.correct
            return (
              <li key={option.label}>
                <motion.button
                  className="drill-option"
                  data-state={
                    revealCorrect
                      ? 'correct'
                      : isTried && option.correct
                        ? 'correct'
                        : isTried
                          ? 'wrong'
                          : finished
                            ? 'dimmed'
                            : undefined
                  }
                  disabled={finished || isTried}
                  onClick={() => pick(i)}
                  animate={
                    shakeIndex === i
                      ? { x: [0, -8, 7, -5, 3, 0] }
                      : { x: 0 }
                  }
                  transition={shakeIndex === i ? { duration: 0.42 } : spring.tap}
                >
                  <span className="drill-option-label">{option.label}</span>
                  {(isTried || revealCorrect) && (
                    <span className="drill-option-mark" aria-hidden="true">
                      {option.correct ? <Icon name="check" size={18} /> : '×'}
                    </span>
                  )}
                </motion.button>

                {/* The teaching moment: why the option you just picked was tempting. */}
                <AnimatePresence initial={false}>
                  {(isTried || (finished && option.correct)) && (
                    <motion.p
                      className="drill-why"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={spring.smooth}
                    >
                      {option.why}
                    </motion.p>
                  )}
                </AnimatePresence>
              </li>
            )
          })}
        </ul>

        <AnimatePresence>
          {finished && (
            <motion.div
              className="drill-result"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={spring.nav}
            >
              <div className="drill-reveal">
                <p className="drill-reveal-label">The arithmetic</p>
                <p className="drill-reveal-body selectable">{drill.reveal}</p>
              </div>

              <Button block size="lg" variant="secondary" onClick={() => nav.pop()}>
                Done
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Screen>
  )
}
