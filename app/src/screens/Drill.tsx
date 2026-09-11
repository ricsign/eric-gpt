import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Screen } from '../ui/Screen'
import { BackButton } from '../ui/NavStack'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { useStore } from '../state/store'
import { DRILLS } from '../data/drills'
import { drillIndex, drillNumber, glyph, MAX_ATTEMPTS, shareResult } from '../lib/daily'
import { dayKey } from '../lib/format'
import { haptic } from '../lib/haptics'
import { spring } from '../lib/motion'
import './Drill.css'

/**
 * The Daily Drill.
 *
 * One question, the same one for everyone, three attempts. The design points that
 * matter:
 *
 *  - **A wrong answer teaches immediately.** The option you picked explains why it
 *    was tempting, before you try again. For most players the near-miss is the
 *    thing they remember, so that is where the content goes.
 *  - **The reveal shows the arithmetic either way.** Losing still has to be worth
 *    the sixty seconds, or people stop coming.
 *  - **Sharing discloses nothing.** Squares say how many tries, never which option.
 *    A result that cannot spoil the puzzle is a result people paste into group chats.
 */
export function DrillScreen() {
  const { state, recordDrillAttempt } = useStore()
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
  const [shareState, setShareState] = useState<'idle' | 'shared' | 'copied' | 'failed'>('idle')

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

  const onShare = async () => {
    haptic('light')
    const result = await shareResult({
      number,
      attempts,
      solved,
      streak: state.streak.current,
    })
    setShareState(result)
    if (result !== 'failed') haptic('success')
  }

  const remaining = MAX_ATTEMPTS - attempts.length

  return (
    <Screen
      title={`Drill #${number}`}
      inlineTitle
      left={<BackButton label="Today" />}
      right={
        state.streak.current > 0 ? (
          <span className="drill-streak">
            <Icon name="flame" size={15} />
            {state.streak.current}
          </span>
        ) : undefined
      }
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
              <div className="drill-verdict">
                <span className="drill-glyph" aria-hidden="true">
                  {glyph({ number, attempts, solved, streak: state.streak.current })}
                </span>
                <span className="drill-score num">
                  {solved ? `${attempts.length}/${MAX_ATTEMPTS}` : `X/${MAX_ATTEMPTS}`}
                </span>
              </div>

              <div className="drill-reveal">
                <p className="drill-reveal-label">The arithmetic</p>
                <p className="drill-reveal-body selectable">{drill.reveal}</p>
              </div>

              <Button
                block
                size="lg"
                variant="secondary"
                icon={<Icon name="share" size={18} />}
                onClick={onShare}
                feedback={null}
              >
                {shareState === 'copied'
                  ? 'Copied'
                  : shareState === 'shared'
                    ? 'Shared'
                    : 'Share your result'}
              </Button>

              <p className="drill-share-note">
                Your result is three squares and a streak. It contains no dollar amount, and it
                does not give the answer away.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Screen>
  )
}
