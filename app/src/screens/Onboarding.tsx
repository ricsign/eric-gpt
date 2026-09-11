import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Button } from '../ui/Button'
import { ValueSlider } from '../ui/ValueSlider'
import { GrowthChart } from '../ui/GrowthChart'
import { NumberRoll } from '../ui/NumberRoll'
import { useStore } from '../state/store'
import { growthSeries, futureValue } from '../lib/finance'
import { money, moneyCompact, percent } from '../lib/format'
import { haptic } from '../lib/haptics'
import './Onboarding.css'

/**
 * Onboarding.
 *
 * Four questions, no account, no bank link, nothing leaves the device.
 *
 * That is not minimalism for its own sake. Requiring a signup before the first
 * computation is an entry toll on a product whose whole claim is that it is a
 * neutral utility, and bank-link flows lose a large share of users at the connect
 * step — a price not worth paying to avoid four sliders.
 *
 * The last screen is the payoff, not a "you're all set". The first thing the user
 * sees is their own number, computed from what they just told us.
 */

type Step = 'intro' | 'age' | 'saving' | 'debt' | 'jurisdiction' | 'payoff'

const ORDER: Step[] = ['intro', 'age', 'saving', 'debt', 'jurisdiction', 'payoff']

export function Onboarding() {
  const { state, setProfile, dispatch } = useStore()
  const [step, setStep] = useState<Step>('intro')
  const [age, setAge] = useState(28)
  const [income, setIncome] = useState(65_000)
  const [monthly, setMonthly] = useState(300)
  const [invested, setInvested] = useState(0)
  const [debt, setDebt] = useState(0)

  const index = ORDER.indexOf(step)
  const go = (next: Step) => {
    haptic('light')
    setStep(next)
  }

  const projection = useMemo(
    () =>
      growthSeries({
        principal: invested,
        monthly,
        annualRate: state.profile.assumedReturn,
        years: Math.max(10, 65 - age),
      }),
    [invested, monthly, age, state.profile.assumedReturn],
  )

  const end = projection[projection.length - 1]

  const finish = (jurisdiction: 'US' | 'other') => {
    setProfile({ age, income, monthly, invested, debtBalance: debt, targetAge: 65 })
    dispatch({ type: 'setJurisdiction', value: jurisdiction })
    go('payoff')
  }

  return (
    <div className="onb">
      {/* Progress runs across the top, skipping the intro and the payoff. */}
      {index > 0 && index < ORDER.length - 1 && (
        <div className="onb-progress" aria-hidden="true">
          {ORDER.slice(1, -1).map((s, i) => (
            <span key={s} data-done={i <= index - 1 || undefined} />
          ))}
        </div>
      )}

      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={step}
          className="onb-step scroll"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          // Tween, not a spring: with mode="wait" the two halves run in series,
          // and a spring's settling time doubles into a visible stall.
          transition={{ duration: 0.16, ease: [0.32, 0.72, 0, 1] }}
        >
          {step === 'intro' && (
            <div className="onb-intro">
              <div className="onb-mark" aria-hidden="true">
                <svg viewBox="0 0 120 72">
                  {/* The product's whole thesis, as a mark: a curve that bends. */}
                  <path
                    d="M4 68 C 34 66, 58 58, 76 40 S 104 8, 116 4"
                    fill="none"
                    stroke="var(--growth)"
                    strokeWidth="5"
                    strokeLinecap="round"
                  />
                  <circle cx="116" cy="4" r="6" fill="var(--growth)" />
                </svg>
              </div>

              <h1 className="onb-title">Compound</h1>
              <p className="onb-lede">
                Two minutes a day. One question everyone gets, and a set of tools that run on
                your own numbers.
              </p>

              <ul className="onb-promises">
                <li>
                  <strong>No account.</strong> Nothing to sign up for, nothing to remember.
                </li>
                <li>
                  <strong>No bank connection.</strong> Four sliders, not a login to your bank.
                </li>
                <li>
                  <strong>Nothing leaves this device.</strong> Your numbers are stored in this
                  browser and sent nowhere.
                </li>
                <li>
                  <strong>No affiliate links, ever.</strong> We are not paid to point you at a
                  card, a fund, or a bank.
                </li>
              </ul>

              <Button block size="lg" onClick={() => go('age')}>
                Start
              </Button>
              <button className="onb-skip" onClick={() => finish('US')}>
                Skip — just show me the app
              </button>
            </div>
          )}

          {step === 'age' && (
            <Question
              title="Where are you starting from?"
              sub="Rough is fine. You can change any of this later."
              onNext={() => go('saving')}
            >
              <ValueSlider
                label="Your age"
                value={age}
                onChange={setAge}
                min={16}
                max={75}
                format={(v) => `${v}`}
              />
              <ValueSlider
                label="Annual income, before tax"
                value={income}
                onChange={setIncome}
                min={0}
                max={400_000}
                step={1_000}
                curve="log"
                format={money}
                hint="Used to size an employer match. Never sent anywhere."
              />
            </Question>
          )}

          {step === 'saving' && (
            <Question
              title="What are you putting away?"
              sub="Everything already invested, and what you add each month."
              onNext={() => go('debt')}
            >
              <ValueSlider
                label="Already saved or invested"
                value={invested}
                onChange={setInvested}
                min={0}
                max={1_000_000}
                step={500}
                curve="log"
                format={money}
              />
              <ValueSlider
                label="Added each month"
                value={monthly}
                onChange={setMonthly}
                min={0}
                max={5_000}
                step={25}
                format={money}
                hint="If the answer is zero, that is a fine place to start from."
              />

              {/* The payoff starts here, not at the end — the chart updates as they
                  drag, so the relationship is felt before it is ever explained. */}
              <div className="onb-preview">
                <p className="onb-preview-label">At 65, on a {percent(state.profile.assumedReturn)} assumption</p>
                <p className="onb-preview-value num">{moneyCompact(
                  futureValue({
                    principal: invested,
                    monthly,
                    annualRate: state.profile.assumedReturn,
                    years: Math.max(10, 65 - age),
                  }).balance,
                )}</p>
              </div>
            </Question>
          )}

          {step === 'debt' && (
            <Question
              title="Anything charging you interest?"
              sub="Cards, personal loans — the expensive kind. Leave it at zero if not."
              onNext={() => go('jurisdiction')}
            >
              <ValueSlider
                label="Highest-rate balance"
                value={debt}
                onChange={setDebt}
                min={0}
                max={60_000}
                step={100}
                curve="log"
                format={money}
                hint="This decides which tools matter most for you — it is not a judgement."
              />
            </Question>
          )}

          {step === 'jurisdiction' && (
            <div className="onb-question">
              <h2 className="onb-q-title">Which country's rules apply to you?</h2>
              <p className="onb-q-sub">
                Contribution limits, tax brackets and account types are completely different
                between countries. We ask rather than guess, because showing you US numbers with
                a caveat would be worse than showing you nothing.
              </p>

              <div className="onb-choices">
                <button className="onb-choice pressable" onClick={() => finish('US')}>
                  <span className="onb-choice-label">United States</span>
                  <span className="onb-choice-sub">Everything is available</span>
                </button>
                <button className="onb-choice pressable" onClick={() => finish('other')}>
                  <span className="onb-choice-label">Somewhere else</span>
                  <span className="onb-choice-sub">
                    You get the maths — compounding, debt, fees, behaviour. We will hide the
                    US-specific tools rather than show you the wrong numbers.
                  </span>
                </button>
              </div>
            </div>
          )}

          {step === 'payoff' && (
            <div className="onb-payoff">
              <p className="onb-payoff-label">On what you just told us</p>
              <h2 className="onb-payoff-value">
                <NumberRoll
                  value={end.balance}
                  duration={1.8}
                  format={moneyCompact}
                  ariaLabel={`${moneyCompact(end.balance)} at 65`}
                />
              </h2>
              <p className="onb-payoff-sub">
                at 65, of which <strong>{moneyCompact(end.contributed)}</strong> is money you put
                in and <strong>{moneyCompact(end.growth)}</strong> is growth that nobody paid for.
              </p>

              {/* showReadout off: the headline above already is the number, and two
                  copies disagreeing mid-count-up looks like a bug. */}
              <GrowthChart points={projection} height={250} showReadout={false} />

              <p className="onb-disclaimer">
                An illustration on a {percent(state.profile.assumedReturn)} assumption you can
                change, not a prediction or a recommendation. Real returns vary and can be
                negative.
              </p>

              <Button
                block
                size="lg"
                onClick={() => {
                  haptic('success')
                  dispatch({ type: 'completeOnboarding' })
                }}
              >
                Take me in
              </Button>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function Question({
  title,
  sub,
  children,
  onNext,
}: {
  title: string
  sub: string
  children: React.ReactNode
  onNext: () => void
}) {
  return (
    <div className="onb-question">
      <h2 className="onb-q-title">{title}</h2>
      <p className="onb-q-sub">{sub}</p>
      <div className="onb-fields">{children}</div>
      <Button block size="lg" onClick={onNext}>
        Next
      </Button>
    </div>
  )
}
